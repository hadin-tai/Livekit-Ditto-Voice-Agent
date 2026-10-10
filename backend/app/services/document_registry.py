import json
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger("document_registry")

# Define storage path
STORAGE_DIR = Path(__file__).parent.parent.parent / "storage"
DOCUMENTS_FILE = STORAGE_DIR / "documents.json"


class DocumentRegistry:
    def __init__(self):
        # Ensure storage directory exists
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        # Initialize file if it doesn't exist
        if not DOCUMENTS_FILE.exists():
            with open(DOCUMENTS_FILE, "w") as f:
                json.dump([], f)
            logger.info(f"Created new document registry at {DOCUMENTS_FILE}")

    def _load_documents(self) -> List[dict]:
        with open(DOCUMENTS_FILE, "r") as f:
            return json.load(f)

    def _save_documents(self, documents: List[dict]):
        with open(DOCUMENTS_FILE, "w") as f:
            json.dump(documents, f, indent=2)

    def save_document(self, document_id: str, document_name: str) -> dict:
        logger.info(f"Saving document to registry: {document_name} (ID: {document_id})")
        documents = self._load_documents()
        new_doc = {
            "document_id": document_id,
            "document_name": document_name,
            "uploaded_at": datetime.now().isoformat()
        }
        documents.append(new_doc)
        self._save_documents(documents)
        logger.info(f"Document saved successfully")
        return new_doc

    def list_documents(self) -> List[dict]:
        logger.info("Listing all documents from registry")
        docs = self._load_documents()
        # Return only the fields needed for frontend
        return [
            {
                "document_id": doc["document_id"],
                "document_name": doc["document_name"]
            }
            for doc in docs
        ]

    def get_document(self, document_id: str) -> Optional[dict]:
        logger.info(f"Looking up document with ID: {document_id}")
        documents = self._load_documents()
        for doc in documents:
            if doc["document_id"] == document_id:
                return doc
        return None

    def delete_document(self, document_id: str) -> bool:
        logger.info(f"Deleting document from registry: {document_id}")
        documents = self._load_documents()
        filtered = [doc for doc in documents if doc["document_id"] != document_id]
        if len(filtered) == len(documents):
            logger.warning(f"Document {document_id} not found in registry")
            return False
        self._save_documents(filtered)
        logger.info(f"Document {document_id} deleted successfully from registry")
        return True


# Singleton instance
document_registry = DocumentRegistry()