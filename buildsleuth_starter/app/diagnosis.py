from __future__ import annotations

import re
from app.models import CandidateCause, Evidence
from app.retrieval import retrieve

RULES = [
    (re.compile(r"ModuleNotFoundError|No module named|ImportError", re.I), "Missing or unresolved Python dependency", "The log contains a Python import error.", ["Check the missing package is declared in project dependencies.", "Confirm the CI job installs dependencies in the active environment.", "Check Python and package version compatibility."]),
    (re.compile(r"ResolutionImpossible|Could not find a version that satisfies|dependency conflict", re.I), "Dependency resolution conflict", "The log indicates package constraints could not be resolved.", ["Inspect dependency and lockfile changes.", "Verify runtime and package version compatibility.", "Reproduce the install command in a clean environment."]),
    (re.compile(r"AssertionError|assert .*==|FAILED .*test", re.I), "Failing test or changed behavior", "The log contains a test failure signal.", ["Start with the first failing assertion and compare expected versus actual output.", "Inspect changed code and relevant fixtures.", "Rerun the single failing test with verbose output."]),
    (re.compile(r"KeyError|missing environment variable|environment variable .* not set", re.I), "Missing key or configuration", "The log suggests a missing key or required configuration value.", ["Check required configuration names in CI settings.", "Verify secrets are configured without printing their values.", "Compare local and CI environment assumptions."]),
    (re.compile(r"timed out|timeout|ConnectionError|Temporary failure in name resolution", re.I), "Network, availability, or timeout issue", "The log contains a timeout or connectivity signal.", ["Check external service availability and network access.", "Review timeout and retry settings.", "Determine whether the test depends on a flaky external service."]),
    (re.compile(r"flake8|ruff|black|eslint|prettier|lint", re.I), "Linting or formatting failure", "The log references a linter or formatter.", ["Run the same tool version locally.", "Inspect reported files and line numbers.", "Check that CI and local configuration are aligned."]),
]


def diagnose(log: str, diff: str | None = None) -> tuple[list[str], list[CandidateCause], list[Evidence]]:
    signals: list[str] = []
    causes: list[CandidateCause] = []
    for pattern, title, rationale, checks in RULES:
        matches = pattern.findall(log)
        if matches:
            signals.append(title)
            causes.append(CandidateCause(title=title, rationale=rationale, recommended_checks=checks))
    if diff and diff.strip():
        signals.append("Code diff supplied")
    context = log + "\n" + (diff or "")
    ranked = retrieve(context, limit=3)
    evidence = [Evidence(source=doc.source, snippet=doc.text, score=round(score, 3)) for doc, score in ranked]
    if not causes:
        causes.append(CandidateCause(
            title="No known failure signature matched",
            rationale="The baseline rules did not recognize a common failure signature; this is not evidence that the log has no root cause.",
            recommended_checks=["Inspect the earliest error in the log rather than only the final summary.", "Include the failing step, relevant diff, and runtime details.", "Add a regression fixture if this failure pattern recurs."],
        ))
    return signals, causes, evidence
