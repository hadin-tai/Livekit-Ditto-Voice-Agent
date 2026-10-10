import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.document_registry import document_registry
from app.services.pinecone_service import pinecone_service

router = APIRouter()
logger = logging.getLogger("documents_api")

# Temporary storage directory
TEMP_DIR = Path(__file__).parent.parent.parent.parent / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    logger.info(f"Received file upload: {file.filename}")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )

    temp_file_path = TEMP_DIR / f"{uuid.uuid4()}_{file.filename}"

    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        logger.info(f"Saved temp file: {temp_file_path}")

        # Generate document ID
        document_id = str(uuid.uuid4())

        # Upload to Pinecone
        await pinecone_service.upload_file(
            file_path=str(temp_file_path),
            user_id="default_user",
            document_id=document_id,
        )

        # Save metadata locally
        document_registry.save_document(
            document_id=document_id,
            document_name=file.filename,
        )

        logger.info("Document uploaded successfully")

        return {
            "success": True,
            "document_id": document_id,
            "document_name": file.filename,
        }

    except Exception as e:
        logger.exception(f"Failed to upload document: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to upload document"
        )

    finally:
        # Always clean up temporary file
        if temp_file_path.exists():
            temp_file_path.unlink()
            logger.info("Temp file cleaned up")


@router.get("/api/documents")
async def list_documents():
    logger.info("Listing documents")

    try:
        docs = document_registry.list_documents()

        logger.info(f"Found {len(docs)} documents")

        return docs

    except Exception as e:
        logger.exception(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list documents"
        )


@router.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    logger.info(f"Deleting document: {document_id}")

    try:
        # Delete from Pinecone
        await pinecone_service.delete_file(
            document_id=document_id
        )

        # Delete from registry
        success = document_registry.delete_document(document_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )

        return {
            "success": True,
            "message": "Document deleted successfully",
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Failed to delete document: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete document"
        )