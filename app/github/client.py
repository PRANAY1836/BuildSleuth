"""Stage 7: Least-privilege GitHub REST API client with SSRF protection and log/diff fetching."""

from __future__ import annotations

import logging
import httpx
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class GitHubClient:
    """Secure GitHub API integration with SSRF validation and optional token auth."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.base_url = "https://api.github.com"

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token and self.token.strip():
            headers["Authorization"] = f"Bearer {self.token.strip()}"
        return headers

    def validate_repository_url(self, repo_slug_or_url: str) -> str:
        """Validate and extract owner/repo slug defending against SSRF."""
        cleaned = repo_slug_or_url.strip()
        if cleaned.startswith("http://") or cleaned.startswith("https://"):
            parsed = urlparse(cleaned)
            if parsed.netloc.lower() not in ("github.com", "api.github.com"):
                raise ValueError(f"Invalid host '{parsed.netloc}'; only github.com is supported.")
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) < 2:
                raise ValueError("Invalid GitHub repository URL path.")
            return f"{parts[0]}/{parts[1]}"
        
        # Assume owner/repo format
        parts = [p for p in cleaned.split("/") if p]
        if len(parts) != 2:
            raise ValueError("Repository must be in 'owner/repo' format or a valid github.com URL.")
        return f"{parts[0]}/{parts[1]}"

    def fetch_commit_diff(self, repo_slug: str, commit_sha: str) -> str:
        """Fetch unified diff for a given commit SHA."""
        clean_repo = self.validate_repository_url(repo_slug)
        url = f"{self.base_url}/repos/{clean_repo}/commits/{commit_sha}"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url, headers={**self._get_headers(), "Accept": "application/vnd.github.v3.diff"})
            if resp.status_code == 200:
                return resp.text
            logger.warning("GitHub commit diff fetch returned HTTP %s", resp.status_code)
            return f"# Failed to fetch commit diff for {commit_sha} (HTTP {resp.status_code})"
        except Exception as exc:
            logger.warning("GitHub API error: %s", exc)
            return f"# Error fetching commit diff: {exc}"