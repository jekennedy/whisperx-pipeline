"""Shared utilities for interacting with the pyannote diarization API."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests

DEFAULT_API_BASE = os.environ.get("PYANNOTE_API_BASE", "https://api.pyannote.ai")
ENV_API_KEY = "PYANNOTE_API_KEY"


def require_api_key() -> str:
    """Fetch the pyannote API key from environment variables."""
    api_key = os.environ.get(ENV_API_KEY)
    if not api_key:
        raise SystemExit(f"{ENV_API_KEY} is not set. Export your pyannote API token first.")
    return api_key


class PyannoteClient:
    """Thin wrapper around the pyannote diarization REST API."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_API_BASE):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def submit_diarization(
        self,
        audio_url: str,
        *,
        model: str = "precision-2",
        webhook: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        include_confidence: bool = True,
        include_turn_confidence: bool = True,
        exclusive: bool = False,
    ) -> str:
        payload: Dict[str, Any] = {
            "url": audio_url,
            "model": model,
            "confidence": include_confidence,
            "turnLevelConfidence": include_turn_confidence,
            "exclusive": exclusive,
        }
        if webhook:
            payload["webhook"] = webhook
        for key, value in (
            ("numSpeakers", num_speakers),
            ("minSpeakers", min_speakers),
            ("maxSpeakers", max_speakers),
        ):
            if value is not None:
                payload[key] = value

        resp = requests.post(
            f"{self.base_url}/v1/diarize",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        self._raise_for_status(resp)
        data = resp.json()
        return data["jobId"]

    def wait_for_job(
        self,
        job_id: str,
        *,
        interval: float = 3.0,
        timeout: float = 900.0,
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout
        while True:
            resp = requests.get(
                f"{self.base_url}/v1/jobs/{job_id}",
                headers=self._headers(),
                timeout=30,
            )
            self._raise_for_status(resp)
            payload = resp.json()
            status = payload.get("status")
            if status in {"succeeded", "failed", "canceled"}:
                if status != "succeeded":
                    message = (
                        payload.get("output", {}).get("error")
                        or payload.get("warning")
                        or payload
                    )
                    raise RuntimeError(f"Job {job_id} ended with {status}: {message}")
                return payload
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for job {job_id}; last status={status}")
            time.sleep(interval)

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text
            raise RuntimeError(f"{resp.request.method} {resp.request.url} failed: {detail}") from exc
