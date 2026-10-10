from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.routes import token
from app.api.routes import rag, documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("backend-main")

app = FastAPI(title="AI Voice Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}", extra={
        "method": request.method,
        "path": request.url.path,
        "client": request.client.host if request.client else "unknown"
    })
    response = await call_next(request)
    logger.info(f"Request completed: {request.method} {request.url.path} - Status: {response.status_code}", extra={
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code
    })
    return response

app.include_router(token.router, prefix="")
app.include_router(rag.router)
app.include_router(documents.router)

logger.info("Backend application startup complete")

@app.get("/health")
async def health():
    logger.info("Health check request received")
    return {"status": "ok"}
