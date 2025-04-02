from pydantic import BaseModel, Field, field_validator

class MergedRetrievalResult(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)
    score: float
    source: str

    @field_validator('metadata', mode='before')
    def validate_metadata(cls, v):
        return v or {}