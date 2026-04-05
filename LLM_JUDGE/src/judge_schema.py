from pydantic import BaseModel, Field, field_validator, model_validator


class JudgeResult(BaseModel):
    malicious_intent_preserved: bool
    realistic_for_context: bool
    non_trivial_mutation: bool
    overall_quality_score: int = Field(ge=1, le=5)
    keep: bool
    reason: str

    @field_validator("reason")
    @classmethod
    def reason_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reason cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_keep_logic(self):
        expected_keep = (
            self.malicious_intent_preserved
            and self.realistic_for_context
            and self.non_trivial_mutation
            and self.overall_quality_score >= 4
        )
        if self.keep != expected_keep:
            self.keep = expected_keep
        return self