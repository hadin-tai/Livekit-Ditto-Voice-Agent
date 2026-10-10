from fastapi import APIRouter
from app.schemas.rag import RagQueryRequest, RagQueryResponse
from app.services.pinecone_service import pinecone_service
import logging

router = APIRouter()
logger = logging.getLogger("rag-routes")


@router.post("/api/rag/search", response_model=RagQueryResponse)
async def rag_search(request: RagQueryRequest):
    logger.info(f"Processing RAG search request for user {request.user_id}, document {request.document_id}")
    answer = await pinecone_service.chat(
        question=request.question,
        user_id=request.user_id,
        document_id=request.document_id
    )
    return RagQueryResponse(answer=answer)
