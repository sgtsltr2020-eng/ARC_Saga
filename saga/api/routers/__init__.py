"""
SAGA API Routers
================

FastAPI routers for the SAGA Local Server API.
"""

from saga.api.routers.approval import router as approval_router

__all__ = ["approval_router"]
