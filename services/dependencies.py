# dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt as pyjwt
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Security scheme for Bearer tokens
security = HTTPBearer()

# JWT Configuration (from config.py)
from config import SECRET_KEY, ALGORITHM

def decode_jwt_token(token: str):
    """Helper function to decode JWT token using PyJWT."""
    try:
        payload = pyjwt.decode(
            token, 
            SECRET_KEY, 
            algorithms=[ALGORITHM],
            audience="projects-gpt.api"
        )
        return payload
    except Exception as e:
        logger.error(f"Token has expired or invalid: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired or invalid"
        )
    
    
async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Extract user_id from JWT token.
    Looks for: sub, user_id, id, userId
    """
    token = credentials.credentials
    payload = decode_jwt_token(token)
    
    # Try different possible field names
    user_id = (
        payload.get("nameid")
    )
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found in token"
        )
    
    return str(user_id)

async def get_customer_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Extract customer_id from JWT token.
    Looks for: customer_id, customerId, tenant_id, tenantId
    """
    token = credentials.credentials
    payload = decode_jwt_token(token)
    
    # Try different possible field names
    customer_id = (
        payload.get("customer_id")
    )
    
    if not customer_id:
        logger.warning("No customer_id found in JWT token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Customer ID not found in token"
        )
    
    return str(customer_id)


# Optional: Get token payload for advanced use cases
async def get_token_payload(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict:
    """
    Get the complete JWT payload.
    Useful if you need additional claims.
    """
    token = credentials.credentials
    return decode_jwt_token(token)