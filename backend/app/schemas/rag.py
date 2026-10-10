from pydantic import BaseModel


class RagQueryRequest(BaseModel):
    question: str
    user_id: str
    document_id: str


class RagQueryResponse(BaseModel):
    answer: str
