# services/async_azure_blob_service.py
import asyncio
import os
from typing import Optional, Tuple, Dict, Any, List
from azure.storage.blob.aio import BlobServiceClient, ContainerClient, BlobClient
from azure.storage.blob import ContentSettings
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
import uuid
import logging
import re

logger = logging.getLogger(__name__)


class AsyncAzureBlobService:
    """Async service for handling Azure Blob Storage operations."""
    
    _thread_pools = {}  # Class-level thread pool cache
    
    def __init__(
        self, 
        connection_string: Optional[str] = None,
        base_folder: str = "Templates"
    ):
        """
        Initialize Async Azure Blob Service.
        
        Args:
            connection_string: Azure Storage connection string
            base_folder: Base folder for all blobs
        """
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("Azure Storage connection string is required")
        
        self.base_folder = base_folder.strip("/")
        self.container_prefix = "customer"
        
        # Use shared thread pool instead of creating a new one
        self.thread_pool = self._get_shared_thread_pool()
    
    @classmethod
    def _get_shared_thread_pool(cls):
        """Get or create a shared thread pool."""
        import threading
        thread_id = threading.current_thread().ident
        
        if thread_id not in cls._thread_pools:
            import concurrent.futures
            cls._thread_pools[thread_id] = concurrent.futures.ThreadPoolExecutor(
                max_workers=10,
                thread_name_prefix=f"azure_blob_{thread_id}"
            )
        
        return cls._thread_pools[thread_id]
    
    @classmethod
    async def cleanup(cls):
        """Cleanup all thread pools (call on application shutdown)."""
        for pool in cls._thread_pools.values():
            pool.shutdown(wait=True)
        cls._thread_pools.clear()
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - only close Azure client, not thread pool."""
        if hasattr(self, 'blob_service_client'):
            await self.blob_service_client.close()
    
    def sanitize_container_name(self, customer_id: str) -> str:
        """Sanitize container name."""
        customer_id_str = str(customer_id)
        sanitized = re.sub(r'[^a-z0-9-]', '', customer_id_str.lower())
        sanitized = sanitized.replace('_', '-')
        sanitized = re.sub(r'-+', '-', sanitized)
        sanitized = sanitized.strip('-')
        
        if len(sanitized) < 3:
            sanitized = f"cust{sanitized}".rjust(3, '0')[:63]
        elif len(sanitized) > 63:
            sanitized = sanitized[:63].rstrip('-')
        
        return sanitized
    
    def get_container_name(self, customer_id: str) -> str:
        """Generate container name for customer."""
        customer_id_str = str(customer_id)
        if customer_id_str.isdigit():
            return f"customer-{customer_id_str}"
        return self.sanitize_container_name(f"{self.container_prefix}-{customer_id_str}")
    
    async def ensure_container_exists(self, customer_id: str) -> str:
        """Create container if it doesn't exist."""
        container_name = self.get_container_name(customer_id)
        
        async with BlobServiceClient.from_connection_string(self.connection_string) as blob_service_client:
            try:
                container_client = blob_service_client.get_container_client(container_name)
                
                try:
                    await container_client.get_container_properties()
                    logger.info(f"✅ Container exists: {container_name}")
                except ResourceNotFoundError:
                    logger.info(f"📦 Creating container: {container_name}")
                    await container_client.create_container()
                    logger.info(f"✅ Container created: {container_name}")
                
                return container_name
                
            except Exception as e:
                logger.error(f"❌ Container error {container_name}: {e}")
                raise
    
    async def upload_file(
        self,
        customer_id: str,
        file_content: bytes,
        filename: str,
        category: str = "",
        target_path: str = "",
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> dict:
        """Upload a file to Azure Blob Storage."""
        actual_base_folder = self.base_folder
        
        # Get container name (sync operation, run in thread pool)
        container_name = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self.get_container_name,
            customer_id
        )
        
        async with BlobServiceClient.from_connection_string(self.connection_string) as blob_service_client:
            try:
                container_client = blob_service_client.get_container_client(container_name)
                
                # Check/create container
                try:
                    await container_client.get_container_properties()
                except ResourceNotFoundError:
                    await container_client.create_container()
                
                # Generate file ID
                file_id = str(uuid.uuid4())
                
                # Build blob path
                path_components = []
                
                if actual_base_folder:
                    path_components.append(actual_base_folder.strip("/"))
                
                if category:
                    path_components.append(category.strip("/"))
                
                if target_path:
                    path_parts = [p.strip("/") for p in target_path.split("/") if p.strip("/")]
                    path_components.extend(path_parts)
                
                clean_filename = filename.replace("/", "_").replace("\\", "_")
                path_components.append(f"{file_id}_{clean_filename}")
                
                blob_name = "/".join(path_components)
                
                # Upload blob
                blob_client = container_client.get_blob_client(blob_name)
                
                content_settings = ContentSettings(
                    content_type=content_type or "application/octet-stream"
                )
                
                await blob_client.upload_blob(
                    data=file_content,
                    content_settings=content_settings,
                    metadata=metadata,
                    overwrite=True
                )
                
                logger.info(f"✅ Uploaded: {blob_name}")
                
                return {
                    "file_id": file_id,
                    "filename": filename,
                    "blob_name": blob_name,
                    "blob_url": blob_client.url,
                    "container": container_name,
                    "content_type": content_type,
                    "size": len(file_content),
                    "full_file_path": blob_name,
                    "category": category,
                    "path": target_path,
                    "base_folder": actual_base_folder,
                    "customer_id": customer_id
                }
                
            except Exception as e:
                logger.error(f"❌ Upload failed {filename}: {e}")
                raise


# Simplified factory function
_async_blob_service_cache = {}

async def get_async_blob_service(base_folder: str = "Templates") -> AsyncAzureBlobService:
    """Get async AzureBlobService instance."""
    if base_folder not in _async_blob_service_cache:
        _async_blob_service_cache[base_folder] = AsyncAzureBlobService(
            base_folder=base_folder
        )
    
    return _async_blob_service_cache[base_folder]