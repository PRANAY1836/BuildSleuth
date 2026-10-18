from fastapi import FastAPI
from app.diagnosis import diagnose
from app.models import DiagnoseRequest, DiagnoseResponse

app = FastAPI(
    title="BuildSleuth API",
    description="Evidence-grounded baseline for investigating CI failures.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "buildsleuth", "version": app.version}


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose_endpoint(request: DiagnoseRequest) -> DiagnoseResponse:
    signals, causes, evidence = diagnose(request.log, request.diff)
    return DiagnoseResponse(
        repository=request.repository,
        run_id=request.run_id,
        extracted_signals=signals,
        candidate_causes=causes,
        evidence=evidence,
        limitations=[
            "This version uses deterministic heuristics and lightweight lexical retrieval, not an LLM-backed RAG pipeline.",
            "Candidate causes are hypotheses; verify them against the repository and full CI context.",
            "No live GitHub Actions API integration is enabled in this starter.",
        ],
    )
