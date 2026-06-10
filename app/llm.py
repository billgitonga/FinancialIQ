# app/llm.py

import os
import json
import time
import logging
import threading
from functools import lru_cache
from typing import Optional, Dict, Any, Generator, Union, Callable
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = os.getenv("OLLAMA_URL", f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "tinyllama")
DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
ENABLE_CACHE = os.getenv("OLLAMA_ENABLE_CACHE", "true").lower() == "true"
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
FORCE_OFFLINE = os.getenv("FINANCEIQ_FORCE_OFFLINE", "false").lower() == "true"

logger = logging.getLogger(__name__)

session = requests.Session()
retry_strategy = Retry(total=MAX_RETRIES, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"])
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)
DEFAULT_HEADERS = {"Content-Type": "application/json"}
if OLLAMA_API_KEY:
    DEFAULT_HEADERS["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
session.headers.update(DEFAULT_HEADERS)


def _get_base_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _validate_prompt(prompt: str, max_chars: int = 50000):
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    if len(prompt) > max_chars:
        logger.warning("Prompt exceeds recommended length (%d chars).", max_chars)


_ollama_health_cache: Optional[bool] = None
_health_check_time: float = 0
_HEALTH_CACHE_TTL = 30  # seconds


def is_ollama_reachable(timeout: int = 2) -> bool:
    if FORCE_OFFLINE:
        logger.info("Force offline mode enabled, skipping Ollama health check.")
        return False
    global _ollama_health_cache, _health_check_time
    now = time.time()
    if _ollama_health_cache is not None and (now - _health_check_time) < _HEALTH_CACHE_TTL:
        return _ollama_health_cache
    base_url = _get_base_url(OLLAMA_URL)
    try:
        response = session.get(base_url, timeout=timeout)
        _ollama_health_cache = response.status_code == 200
        _health_check_time = now
        return _ollama_health_cache
    except Exception:
        _ollama_health_cache = False
        _health_check_time = now
        return False


@lru_cache(maxsize=128)
def _cached_request(prompt: str, system: str, model: str, timeout: int) -> Optional[str]:
    # Use streaming internally but return collected result (avoids hanging on some Ollama versions)
    return _collect_streaming_response(prompt, system, model, timeout)


def _collect_streaming_response(prompt: str, system: str, model: str, timeout: int) -> Optional[str]:
    """Collect streaming response into a single string to avoid hanging."""
    import time
    gen = _ask_ollama_stream(prompt, system, model, timeout)
    if gen is None:
        return None
    result_parts = []
    deadline = time.time() + timeout
    try:
        for chunk in gen:
            if time.time() > deadline:
                logger.warning("Streaming response timed out")
                break
            if chunk:
                result_parts.append(chunk)
        return "".join(result_parts)
    except Exception as e:
        logger.exception("Streaming collection failed: %s", e)
        return None


def _ask_ollama_stream(prompt: str, system: str, model: str, timeout: int) -> Optional[Generator]:
    """Internal streaming generator that handles response cleanup."""
    payload = {"model": model, "prompt": prompt, "system": system, "stream": True}
    headers = session.headers.copy()
    logger.info("Sending Ollama streaming request | model=%s", model)
    start_time = time.perf_counter()
    try:
        response = session.post(OLLAMA_URL, json=payload, headers=headers, timeout=timeout, stream=True)
        elapsed = time.perf_counter() - start_time
        logger.info("Ollama streaming response started | status=%s | %.2fs", response.status_code, elapsed)
        if response.status_code != 200:
            logger.error("Ollama streaming API error | status=%s | body=%s", response.status_code, response.text[:500])
            return None
        def stream_generator():
            try:
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        logger.warning("Failed to decode streaming JSON chunk.")
                        continue
            finally:
                try:
                    response.close()
                except Exception:
                    pass
        return stream_generator()
    except Exception as e:
        logger.exception("Ollama streaming request failed: %s", e)
        return None


def _ask_ollama_internal(prompt: str, system: str, model: str, timeout: int, stream: bool = False, extra_headers: Optional[Dict[str, str]] = None) -> Union[Optional[str], Generator[str, None, None]]:
    payload = {"model": model, "prompt": prompt, "system": system, "stream": stream}
    headers = session.headers.copy()
    if extra_headers:
        headers.update(extra_headers)
    logger.info("Sending Ollama request | model=%s | stream=%s", model, stream)
    start_time = time.perf_counter()
    try:
        response = session.post(OLLAMA_URL, json=payload, headers=headers, timeout=timeout, stream=stream)
        elapsed = time.perf_counter() - start_time
        logger.info("Ollama response received | status=%s | %.2fs", response.status_code, elapsed)
        if response.status_code != 200:
            logger.error("Ollama API error | status=%s | body=%s", response.status_code, response.text[:500])
            return None
        if stream:
            def stream_generator():
                try:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line.decode("utf-8"))
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            logger.warning("Failed to decode streaming JSON chunk.")
                finally:
                    response.close()
            return stream_generator()
        else:
            data = response.json()
            return data.get("response", "")
    except Exception as e:
        logger.exception("Ollama request failed: %s", e)
        return None


def ask_ollama(prompt: str, system: str = "You are a helpful financial assistant.", model: Optional[str] = None, stream: bool = False, timeout: int = DEFAULT_TIMEOUT, use_cache: bool = ENABLE_CACHE, extra_headers: Optional[Dict[str, str]] = None) -> Union[Optional[str], Generator[str, None, None]]:
    _validate_prompt(prompt)
    selected_model = model or MODEL_NAME
    if FORCE_OFFLINE:
        logger.debug("Force offline mode, skipping Ollama request.")
        return None
    if stream:
        return _ask_ollama_internal(prompt=prompt, system=system, model=selected_model, timeout=timeout, stream=True, extra_headers=extra_headers)
    if use_cache:
        return _cached_request(prompt=prompt, system=system, model=selected_model, timeout=timeout)
    # Use streaming to avoid hanging on some Ollama versions, but collect to single string
    return _collect_streaming_response(prompt, system, selected_model, timeout)


def ask_ollama_async(prompt: str, callback: Callable[[Optional[str]], None], system: str = "You are a helpful financial assistant.", model: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT, use_cache: bool = ENABLE_CACHE, extra_headers: Optional[Dict[str, str]] = None) -> None:
    def task():
        result = ask_ollama(prompt=prompt, system=system, model=model, stream=False, timeout=timeout, use_cache=use_cache, extra_headers=extra_headers)
        callback(result)
    thread = threading.Thread(target=task, daemon=True)
    thread.start()