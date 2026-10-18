from pydantic import BaseModel, Field


class DiagnoseRequest(BaseModel):
    repository: str | None = Field(default=None, max_length=300)
    run_id: str | None = Field(default=None, max_length=200)
    log: str = Field(min_length=1, max_length=500_000)
    diff: str | None = Field(default=None, max_length=100_000)


class Evidence(BaseModel):
    source: str
    snippet: str
    score: float


class CandidateCause(BaseModel):
    title: str
    rationale: str
    recommended_checks: list[str]
    confidence: str = "heuristic"


class DiagnoseResponse(BaseModel):
    repository: str | None
    run_id: str | None
    extracted_signals: list[str]
    candidate_causes: list[CandidateCause]
    evidence: list[Evidence]
    limitations: list[str]
