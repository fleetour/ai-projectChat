# services/azure_blob_service.py
import os
from typing import Optional, Tuple, BinaryIO
from azure.storage.blob import BlobServiceClient, ContentSettings, BlobClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
import uuid
import logging

logger = logging.getLogger(__name__)


class AzureBlobService:
    """Service for handling Azure Blob Storage operations."""
    
    def __init__(
        self, 
        connection_string: Optional[str] = None,
        base_folder: str = "Projects"
    ):
        """
        Initialize Azure Blob Service.
        
        Args:
            connection_string: Azure Storage connection string
            base_folder: Base folder for all blobs (default: "Projects")
        """
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("Azure Storage connection string is required")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_prefix = "customer"
        self.base_folder = base_folder.strip("/")  # Remove trailing slashes
    
    def sanitize_container_name(self, customer_id: str) -> str:
        """
        Sanitize container name to meet Azure requirements.
        Azure container names must be lowercase, 3-63 chars, only letters, numbers, hyphens.
        """
        # Convert to string first
        customer_id_str = str(customer_id)
        
        # Remove any non-alphanumeric characters except hyphens
        sanitized = re.sub(r'[^a-z0-9-]', '', customer_id_str.lower())
        
        # Replace underscores with hyphens
        sanitized = sanitized.replace('_', '-')
        
        # Remove consecutive hyphens
        sanitized = re.sub(r'-+', '-', sanitized)
        
        # Ensure it starts and ends with alphanumeric
        sanitized = sanitized.strip('-')
        
        # Ensure length is 3-63 characters
        if len(sanitized) < 3:
            # Pad with zeros if too short
            sanitized = f"cust{sanitized}".rjust(3, '0')[:63]
        elif len(sanitized) > 63:
            # Truncate if too long
            sanitized = sanitized[:63].rstrip('-')
        
        return sanitized
    
    def get_container_name(self, customer_id: str) -> str:
        """Generate container name for customer following Azure rules."""
        # Convert to string first
        customer_id_str = str(customer_id)
        
        # If customer_id is just a number like "1", create a proper name
        if customer_id_str.isdigit():
            return f"customer-{customer_id_str}"
        else:
            # Sanitize the customer_id
            return self.sanitize_container_name(f"{self.container_prefix}-{customer_id_str}")
    
    def ensure_container_exists(self, customer_id: str) -> str:
        """Create container if it doesn't exist and return container name."""
        # Convert to string
        customer_id_str = str(customer_id)
        container_name = self.get_container_name(customer_id_str)
        
        logger.info(f"🔄 Creating/checking container: {container_name}")
        
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            
            try:
                # Try to get properties to check if container exists
                container_client.get_container_properties()
                logger.info(f"✅ Container already exists: {container_name}")
                return container_name
            except Exception as get_error:
                # Container doesn't exist, create it
                logger.info(f"📦 Creating new container: {container_name}")
                container_client.create_container()
                logger.info(f"✅ Container created: {container_name}")
                return container_name
                
        except Exception as e:
            logger.error(f"❌ Failed to create/access container {container_name}: {e}")
            raise
    
    def upload_file(
        self,
        customer_id: str,
        file_content: bytes,
        filename: str,
        target_project: str,
        target_path: str = "",
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None,
        base_folder: Optional[str] = None  # Optional override for base folder
    ) -> dict:
        """
        Upload a file to Azure Blob Storage.
        
        Args:
            customer_id: Customer identifier (string or int)
            file_content: File content as bytes
            filename: Original filename
            target_project: Project name
            target_path: Path within project (optional)
            content_type: MIME type of the file
            metadata: Additional metadata for the blob
            base_folder: Override default base folder (optional)
        
        Returns:
            dict: Upload information including blob URL and path
        """
        try:
            # Ensure container exists
            container_name = self.ensure_container_exists(customer_id)
            container_client = self.blob_service_client.get_container_client(container_name)
            
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Use provided base_folder or default
            actual_base_folder = base_folder or self.base_folder
            
            # Build blob path: {base_folder}/{target_project}/{target_path}/{file_id}_{filename}
            path_components = []
            
            # Add base folder if specified
            if actual_base_folder:
                path_components.append(actual_base_folder.strip("/"))
            
            # Add target_project if provided
            if target_project:
                path_components.append(target_project.strip("/"))
            
            # Add target_path if provided
            if target_path:
                # Handle nested paths
                path_parts = [p.strip("/") for p in target_path.split("/") if p.strip("/")]
                path_components.extend(path_parts)
            
            # Add filename with UUID
            clean_filename = filename.replace("/", "_").replace("\\", "_")  # Sanitize filename
            path_components.append(f"{file_id}_{clean_filename}")
            
            # Join all parts
            blob_name = "/".join(path_components)
            
            logger.info(f"📤 Uploading to blob: {blob_name}")
            logger.info(f"   Base folder: {actual_base_folder}")
            logger.info(f"   Project: {target_project}")
            logger.info(f"   Path: {target_path}")
            
            # Create blob client
            blob_client = container_client.get_blob_client(blob_name)
            
            # Prepare content settings
            content_settings = ContentSettings(
                content_type=content_type or "application/octet-stream"
            )
            
            # Upload blob
            blob_client.upload_blob(
                data=file_content,
                content_settings=content_settings,
                metadata=metadata,
                overwrite=True
            )
            
            logger.info(f"✅ File uploaded to {blob_name} in container {container_name}")
            
            return {
                "file_id": file_id,
                "filename": filename,
                "blob_name": blob_name,
                "blob_url": blob_client.url,
                "container": container_name,
                "content_type": content_type,
                "size": len(file_content),
                "full_file_path": blob_name,
                "project": target_project,
                "path": target_path,
                "base_folder": actual_base_folder
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to upload file {filename}: {e}", exc_info=True)
            raise
    
    def download_file(self, customer_id: str, blob_name: str) -> Tuple[bytes, dict]:
        """
        Download a file from Azure Blob Storage.
        
        Args:
            customer_id: Customer identifier
            blob_name: Full blob path within container
        
        Returns:
            Tuple[bytes, dict]: File content and blob properties
        """
        try:
            container_name = self.get_container_name(customer_id)
            container_client = self.blob_service_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            # Check if blob exists
            if not blob_client.exists():
                raise ResourceNotFoundError(f"Blob {blob_name} not found in container {container_name}")
            
            # Download blob
            download_stream = blob_client.download_blob()
            content = download_stream.readall()
            
            # Get blob properties
            properties = blob_client.get_blob_properties()
            
            blob_info = {
                "filename": os.path.basename(blob_name),
                "content_type": properties.content_settings.content_type,
                "size": properties.size,
                "last_modified": properties.last_modified,
                "metadata": properties.metadata,
                "blob_url": blob_client.url,
                "blob_name": blob_name
            }
            
            logger.info(f"✅ File downloaded from {blob_name}")
            
            return content, blob_info
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to download blob {blob_name}: {e}")
            raise


# Factory function with configurable base folder
_blob_service_instances = {}

def get_blob_service(base_folder: Optional[str] = None) -> AzureBlobService:
    """
    Get or create AzureBlobService instance with optional base folder.
    
    Args:
        base_folder: Base folder for blobs (default: "Projects")
    
    Returns:
        AzureBlobService instance
    """
    global _blob_service_instances
    
    # Use provided base_folder or default
    actual_base_folder = base_folder or "Projects"
    
    # Create instance key
    instance_key = f"blob_service_{actual_base_folder}"
    
    if instance_key not in _blob_service_instances:
        _blob_service_instances[instance_key] = AzureBlobService(
            base_folder=actual_base_folder
        )
    
    return _blob_service_instances[instance_key]


# Singleton instance (optional)
_blob_service_instance = None

def get_blob_service() -> AzureBlobService:
    """Get or create singleton AzureBlobService instance."""
    global _blob_service_instance
    if _blob_service_instance is None:
        _blob_service_instance = AzureBlobService()
    return _blob_service_instance