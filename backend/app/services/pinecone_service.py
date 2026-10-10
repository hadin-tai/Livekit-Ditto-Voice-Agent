import logging
import time
import re
from pinecone import Pinecone
from app.config import settings

logger = logging.getLogger("pinecone_service")


class PineconeService:
    def __init__(self):
        logger.info("PineconeService initialized (lazy mode)")
        self._pc = None
        self._assistant = None

    def _initialize(self):
        if self._pc is None:
            if not settings.PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY environment variable is not set")
            logger.info("Initializing Pinecone client")
            self._pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self._assistant = self._get_or_create_assistant()
            logger.info("Pinecone client initialized successfully")

    @property
    def pc(self):
        self._initialize()
        return self._pc

    @property
    def assistant(self):
        self._initialize()
        return self._assistant

    def _get_or_create_assistant(self):
        logger.info(f"Getting or creating assistant: {settings.ASSISTANT_NAME}")
        try:
            assistant = self._pc.assistant.describe_assistant(
                assistant_name=settings.ASSISTANT_NAME
            )
            logger.info(f"Assistant {settings.ASSISTANT_NAME} already exists")
            return self._pc.assistant.Assistant(assistant_name=settings.ASSISTANT_NAME)
        except Exception:
            logger.info(f"Creating new assistant: {settings.ASSISTANT_NAME}")
            instructions = """You are a high precision enterprise document assistant.

Rules:
1. Answer only from uploaded document.
2. Search headers, footers, notes, appendix, tiny text.
3. Important answers may appear only once.
4. Never guess.
5. If not found, say Not found in document.
6. Give same answer for repeated same query.
7. Prefer exact wording.
8. Be concise and accurate."""
            self._pc.assistant.create_assistant(
                assistant_name=settings.ASSISTANT_NAME,
                instructions=instructions
            )
            logger.info(f"Assistant {settings.ASSISTANT_NAME} created successfully")
            return self._pc.assistant.Assistant(assistant_name=settings.ASSISTANT_NAME)

    async def upload_file(
        self,
        file_path: str,
        user_id: str,
        document_id: str
    ) -> str:
        self._initialize()
        logger.info(f"Uploading file: {file_path} for user {user_id}, document {document_id}")
        try:
            response = self._assistant.upload_file(
                file_path=file_path,
                metadata={
                    "type": "pdf",
                    "user_id": user_id,
                    "document_id": document_id,
                    "workspace_id": "default"
                }
            )
            # Extract file ID from response
            pinecone_file_id = response.id if hasattr(response, 'id') else str(response)
            logger.info(f"File {file_path} uploaded successfully, Pinecone file ID: {pinecone_file_id}")
            return pinecone_file_id
        except Exception as e:
            logger.exception(f"Failed to upload file {file_path}")
            raise

    def _expand_queries(self, question: str):
        logger.info(f"Expanding query: {question}")

        q = question.strip()

        queries = [
            q,
            q.lower(),
            re.sub(r"[^a-zA-Z0-9 ]", " ", q).strip(),
            " ".join(q.split()[:6]),
        ]

        cleaned = []

        for item in queries:
            item = item.strip()

            if item and item not in cleaned:
                cleaned.append(item)

        logger.info(f"Expanded queries: {cleaned}")

        return cleaned[:4]

    def _extract_answer(self, response):
        logger.info("Extracting answer from response")

        try:
            if hasattr(response, "message"):

                if hasattr(response.message, "content"):
                    return str(response.message.content).strip()

                return str(response.message).strip()

            elif hasattr(response, "messages"):

                last_msg = response.messages[-1]

                if isinstance(last_msg, dict):
                    return str(last_msg.get("content", "")).strip()

                elif hasattr(last_msg, "content"):
                    return str(last_msg.content).strip()

                return str(last_msg).strip()

            return str(response).strip()

        except Exception:
            return str(response).strip()

    def _is_not_found(self, text: str):
        not_found_phrases = [
            "not found in document",
            "not specified in the document",
            "not mentioned in the document",
            "no information found",
            "unable to find",
            "not available in the document"
        ]
        text_lower = text.lower()
        for phrase in not_found_phrases:
            if phrase in text_lower:
                return True
        return False

    async def retrieve_context(
        self,
        question: str,
        user_id: str,
        document_id: str,
        top_k: int = 25,
        snippet_size: int = 1400
    ):
        self._initialize()
        logger.info(f"Retrieving context for question: {question}")
        expanded_queries = self._expand_queries(question)
        try:
            context = self._assistant.context(
                query=question,
                top_k=top_k,
                snippet_size=snippet_size,
                filter={
                    "user_id": user_id,
                    "document_id": document_id
                }
            )
            logger.info("Context retrieved successfully")
            return context
        except Exception as e:
            logger.exception("Failed to retrieve context")
            raise

    async def delete_file(
        self,
        document_id: str
    ):
        self._initialize()
        logger.info(f"Deleting file with document ID: {document_id} from Pinecone")
        try:
            # List files and find the one with matching document_id in metadata
            # Note: Pinecone Assistant's file deletion may vary by API version
            self._assistant.delete_file(document_id)
            logger.info(f"File {document_id} deleted from Pinecone successfully")
        except Exception as e:
            logger.warning(f"Failed to delete file from Pinecone (may not exist): {e}")

    async def chat(
        self,
        question: str,
        user_id: str,
        document_id: str
    ):
        self._initialize()

        start_time = time.time()

        logger.info(
            f"Chat request for user={user_id}, document={document_id}: {question}"
        )
        files = self.assistant.list_files()
        logger.info(f"Files in Pinecone : {files}")

        try:
            expanded_queries = self._expand_queries(question)

            final_prompt = f"""
    Answer using uploaded document only.

    Question:
    {question}

    Rules:
    1. Search all retrieved context carefully.
    2. Search headers, tables, notes, appendix, tiny text.
    3. Prefer exact wording.
    4. If answer exists once, return it.
    5. If truly absent say: Not found in document.
    6. Do not guess.
    """

            # ==================================
            # PASS 1
            # ==================================
            logger.info("Executing Pass 1 retrieval")

            response = self._assistant.chat(
                messages=[
                    {
                        "role": "user",
                        "content": final_prompt
                    }
                ],
                temperature=0.0,
                include_highlights=True,
                filter={
                    "user_id": user_id,
                    "document_id": document_id
                },
                context_options={
                    "top_k": 25,
                    "snippet_size": 1400,
                    "queries": expanded_queries
                }
            )

            answer = self._extract_answer(response)

            logger.info(f"Pass 1 answer: {answer}")

            # ==================================
            # PASS 2
            # ==================================
            if self._is_not_found(answer):

                logger.info(
                    "Pass 1 returned not found. Executing Pass 2 retrieval."
                )

                response = self._assistant.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": final_prompt
                        }
                    ],
                    temperature=0.0,
                    include_highlights=True,
                    filter={
                        "user_id": user_id,
                        "document_id": document_id
                    },
                    context_options={
                        "top_k": 40,
                        "snippet_size": 800,
                        "queries": expanded_queries
                    }
                )

                answer = self._extract_answer(response)

                logger.info(f"Pass 2 answer: {answer}")

            duration = time.time() - start_time

            logger.info(
                f"Chat completed successfully in {duration:.2f}s"
            )

            return answer

        except Exception as e:

            duration = time.time() - start_time

            logger.exception(
                f"Pinecone chat error after {duration:.2f}s: {str(e)}"
            )

            raise


pinecone_service = PineconeService()
