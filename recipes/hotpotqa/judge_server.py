"""Manager for a vLLM OpenAI-compatible server running the LLM judge model.

The judge server is launched as a subprocess on a dedicated GPU and provides
an OpenAI-compatible endpoint for fast, short-output inference.

Also provides :class:`RemoteJudgeClient` for calling an external
OpenAI-compatible API (e.g. DeepSeek V4) without a local GPU.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import aiohttp

from recipes.hotpotqa.judge_prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_VERSION,
    parse_judge_fact_ids,
    parse_judge_score,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_JUDGE_REQUEST_TIMEOUT_S = 10
_JUDGE_MAX_RETRIES = 3
_JUDGE_RETRY_BACKOFF_S = 1.0
_JUDGE_HEALTH_TIMEOUT_S = 120
_JUDGE_HEALTH_POLL_INTERVAL_S = 2.0
_JUDGE_COMPLETION_MAX_TOKENS = 256
_JUDGE_COMPLETION_TEMPERATURE = 0.0
_JUDGE_SCORE_CHOICES = ["0.0", "0.25", "0.5", "0.75", "1.0"]

_T = TypeVar("_T")


# ------------------------------------------------------------------
# Protocol shared by local and remote judge backends
# ------------------------------------------------------------------

@runtime_checkable
class JudgeProtocol(Protocol):
    """Minimum interface that both local and remote judge backends expose."""

    async def start(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def judge_structured(
        self,
        prompt: str,
        *,
        system_prompt: str,
        parser: Callable[[str], Any | None],
        cache_namespace: str,
        max_tokens: int = _JUDGE_COMPLETION_MAX_TOKENS,
        structured_outputs: dict[str, Any] | None = None,
    ) -> JudgeCallResult | None: ...
    async def judge(
        self,
        prompt: str,
        allowed_fact_ids: set[str] | None = None,
    ) -> float | set[str] | None: ...
    def identity_summary(self) -> dict[str, Any]: ...


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class JudgeCallResult(Generic[_T]):
    """One parsed Judge response plus reproducibility and latency metadata."""

    value: _T
    input_hash: str
    cache_hit: bool
    attempts: int
    latency_s: float


# ------------------------------------------------------------------
# Local vLLM judge backend (existing)
# ------------------------------------------------------------------

class JudgeServerManager:
    """Manages a vLLM server for the LLM judge model.

    Parameters
    ----------
    model_path:
        HuggingFace model directory (e.g. ``models/Qwen3-4B``).
    gpu_id:
        CUDA device index on which to launch the server.
    port:
        TCP port for the OpenAI-compatible API server.
    max_model_len:
        Maximum model context length for the judge server.
    gpu_memory_utilization:
        Fraction of GPU memory reserved by vLLM (0.0 – 1.0).
    launch_server:
        Whether this manager owns the vLLM subprocess. Set to ``False`` when
        multiple rollout workers share a server launched by the experiment
        entrypoint.
    """

    def __init__(
        self,
        *,
        model_path: str,
        gpu_id: int,
        port: int = 29500,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.50,
        launch_server: bool = True,
        request_timeout_s: float = _JUDGE_REQUEST_TIMEOUT_S,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.gpu_id = gpu_id
        self.port = port
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.launch_server = launch_server
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        self.request_timeout_s = float(request_timeout_s)

        self._process: subprocess.Popen | None = None
        self._session: aiohttp.ClientSession | None = None
        self._started = False
        self._result_cache: dict[str, Any] = {}

        # Populated after start().
        self.model_checksum: str | None = None
        self.tokenizer_hash: str | None = None
        self.chat_template_hash: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch or attach to the vLLM server and wait until healthy."""
        if not self.launch_server:
            # The A6 entrypoint owns and health-checks the shared server before
            # Ray starts. We still create a persistent session here so that
            # concurrent judge() calls from multiple trajectories share one
            # connector rather than creating/destroying ephemeral sessions
            # (which causes fd conflicts with uvloop).
            self._ensure_session()
            self._started = True
            return

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu_id)}

        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(self.model_path),
            "--port",
            str(self.port),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            str(self.max_model_len),
            "--dtype",
            "bfloat16",
            "--no-enable-log-requests",
        ]

        logger.info(
            "Starting judge server on GPU %d, port %d: %s",
            self.gpu_id,
            self.port,
            " ".join(command),
        )

        self._process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await self._wait_healthy()

        # Compute model identity hashes after the server is up.
        self._compute_model_hashes()

        self._ensure_session()
        self._started = True

        logger.info("Judge server is healthy (version=%s).", JUDGE_VERSION)

    async def shutdown(self) -> None:
        """Gracefully terminate the judge server."""
        if self._session is not None:
            old_connector = self._session.connector
            await self._session.close()
            self._session = None
            if old_connector is not None and not old_connector.closed:
                await old_connector.close()
        self._started = False

        if self._process is not None and self._process.poll() is None:
            logger.info("Shutting down judge server (pid=%d).", self._process.pid)
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
            self._process = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _recreate_session(self) -> None:
        """Close the existing session (if any) and create a fresh one.

        Properly awaits the old connector close so that uvloop deregisters
        the file descriptors before the new connector opens new ones.  This
        prevents the ``RuntimeError: File descriptor N is used by transport``
        crash that occurs when the old connector's keep-alive sockets are
        still registered in the event loop.
        """
        if self._session is not None:
            old_connector = self._session.connector
            await self._session.close()
            self._session = None
            # Drain the old connector's connection pool so that uvloop
            # deregisters all transport objects tied to those sockets.
            if old_connector is not None and not old_connector.closed:
                await old_connector.close()
        self._ensure_session()

    def _ensure_session(self) -> None:
        """Create a persistent aiohttp session if one does not exist yet.

        Uses a ``TCPConnector`` with ``force_close=True`` so that every
        HTTP connection is closed immediately after the response is read.
        This avoids the fd-conflict crashes under uvloop where a reused
        keep-alive socket's file descriptor is still registered as a
        transport when aiohappyeyeballs tries to connect on the same fd.

        The per-request cost of establishing a new TCP connection is
        negligible compared to the vLLM inference latency (~1-5 s), so
        ``force_close=True`` has no measurable throughput impact.
        """
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=0, force_close=True)
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.request_timeout_s),
                connector=connector,
            )

    # ------------------------------------------------------------------
    # Judge inference
    # ------------------------------------------------------------------

    async def judge_structured(
        self,
        prompt: str,
        *,
        system_prompt: str,
        parser: Callable[[str], _T | None],
        cache_namespace: str,
        max_tokens: int = _JUDGE_COMPLETION_MAX_TOKENS,
        structured_outputs: dict[str, Any] | None = None,
    ) -> JudgeCallResult[_T] | None:
        """Call the frozen Judge, retry parse failures, and cache exact inputs.

        The cache key includes the model, prompts, decoding settings, structured
        output constraint, and caller namespace. Only successfully parsed
        responses enter the cache; malformed outputs are retried and ultimately
        fail closed.
        """

        if not self._started:
            logger.error("judge_structured() called before start().")
            return None
        if not cache_namespace.strip():
            raise ValueError("cache_namespace must be non-empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        cache_payload = {
            "namespace": cache_namespace,
            "model": str(self.model_path),
            "system_prompt": system_prompt,
            "user_prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": _JUDGE_COMPLETION_TEMPERATURE,
            "top_p": 1.0,
            "top_k": -1,
            "enable_thinking": False,
            "structured_outputs": structured_outputs,
        }
        input_hash = hashlib.sha256(
            json.dumps(
                cache_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if input_hash in self._result_cache:
            return JudgeCallResult(
                value=copy.deepcopy(self._result_cache[input_hash]),
                input_hash=input_hash,
                cache_hit=True,
                attempts=0,
                latency_s=0.0,
            )

        self._ensure_session()
        session = self._session
        payload: dict[str, Any] = {
            "model": str(self.model_path),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": _JUDGE_COMPLETION_TEMPERATURE,
            "top_p": 1.0,
            "top_k": -1,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if structured_outputs is not None:
            payload["structured_outputs"] = structured_outputs

        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(1, _JUDGE_MAX_RETRIES + 1):
            try:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning(
                            "Judge server returned HTTP %d (attempt %d): %s",
                            resp.status,
                            attempt,
                            body[:500],
                        )
                        last_error = RuntimeError(f"HTTP {resp.status}")
                    else:
                        data = await resp.json()
                        text = data["choices"][0]["message"]["content"]
                        result = parser(text)
                        if result is not None:
                            self._result_cache[input_hash] = copy.deepcopy(result)
                            return JudgeCallResult(
                                value=result,
                                input_hash=input_hash,
                                cache_hit=False,
                                attempts=attempt,
                                latency_s=time.monotonic() - started,
                            )
                        logger.warning(
                            "Judge output did not pass %s parser (attempt %d): %r",
                            cache_namespace,
                            attempt,
                            text[:500],
                        )
                        last_error = ValueError(f"Invalid {cache_namespace} Judge output: {text!r}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning("Judge request failed (attempt %d): %s", attempt, exc)
                last_error = exc
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Judge response envelope failed (attempt %d): %s",
                    attempt,
                    exc,
                )
                last_error = exc
            except RuntimeError as exc:
                logger.warning(
                    "Judge request hit transport error (attempt %d): %s",
                    attempt,
                    exc,
                )
                last_error = exc
                await self._recreate_session()
                session = self._session

            if attempt < _JUDGE_MAX_RETRIES:
                await asyncio.sleep(_JUDGE_RETRY_BACKOFF_S * attempt)

        logger.error(
            "All %d %s Judge attempts failed. Last error: %s",
            _JUDGE_MAX_RETRIES,
            cache_namespace,
            last_error,
        )
        return None

    async def judge(
        self,
        prompt: str,
        allowed_fact_ids: set[str] | None = None,
    ) -> float | set[str] | None:
        """Call the judge model and return the result, or ``None`` on failure.

        Parameters
        ----------
        prompt:
            The user prompt to send to the judge model.
        allowed_fact_ids:
            If provided, the judge operates in fact-ID mode (v4): it expects
            a JSON array of fact IDs, validates them against this set, and
            returns a ``set[str]`` of valid IDs.  If ``None``, the judge
            operates in legacy scalar mode (v3) and returns a ``float``.

        On parse failure, request error, timeout, or transport error, returns
        ``None`` so the calling trajectory can be flagged as ``judge_invalid``
        and excluded from the actor policy update.
        """
        if allowed_fact_ids is not None:
            allowed = frozenset(allowed_fact_ids)
            result = await self.judge_structured(
                prompt,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                parser=lambda text: parse_judge_fact_ids(text, allowed),
                cache_namespace="a6-fact-id:" + ",".join(sorted(allowed)),
            )
        else:
            result = await self.judge_structured(
                prompt,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                parser=parse_judge_score,
                cache_namespace="legacy-a6-scalar",
                structured_outputs={"choice": _JUDGE_SCORE_CHOICES},
            )
        return result.value if result is not None else None

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def _wait_healthy(self) -> None:
        """Poll ``/v1/models`` until the server responds or times out."""
        elapsed = 0.0
        interval = _JUDGE_HEALTH_POLL_INTERVAL_S

        while elapsed < _JUDGE_HEALTH_TIMEOUT_S:
            # Check subprocess hasn't crashed.
            if self._process is not None and self._process.poll() is not None:
                rc = self._process.returncode
                stderr = ""
                if self._process.stderr is not None:
                    try:
                        stderr = self._process.stderr.read(4096).decode(errors="replace")
                    except Exception:
                        pass
                raise RuntimeError(f"Judge server process exited with code {rc}. stderr (first 4k): {stderr}")

            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
                    async with session.get(f"{self.base_url}/v1/models") as resp:
                        if resp.status == 200:
                            return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass

            await asyncio.sleep(interval)
            elapsed += interval

        raise TimeoutError(
            f"Judge server did not become healthy within {_JUDGE_HEALTH_TIMEOUT_S}s on port {self.port}."
        )

    # ------------------------------------------------------------------
    # Model identity
    # ------------------------------------------------------------------

    def _compute_model_hashes(self) -> None:
        """Compute checksums of the judge model artifacts for manifest tracking."""
        model_dir = self.model_path
        if not model_dir.is_dir():
            return

        # Aggregate SHA256 of all safetensors index entries.
        index_file = model_dir / "model.safetensors.index.json"
        if index_file.is_file():
            self.model_checksum = _file_sha256(index_file)
        else:
            # Single-shard model.
            for candidate in sorted(model_dir.glob("model*.safetensors")):
                self.model_checksum = _file_sha256(candidate)
                break

        tokenizer_file = model_dir / "tokenizer_config.json"
        if tokenizer_file.is_file():
            self.tokenizer_hash = _file_sha256(tokenizer_file)

        # Chat template may be embedded in tokenizer_config.json.
        # Extract and hash just the template string if present.
        try:
            with tokenizer_file.open(encoding="utf-8") as fh:
                cfg = json.load(fh)
            template = cfg.get("chat_template")
            if isinstance(template, str):
                self.chat_template_hash = hashlib.sha256(template.encode()).hexdigest()
        except Exception:
            pass

    def identity_summary(self) -> dict[str, Any]:
        """Return a dict of model identity fields for the manifest."""
        return {
            "judge_version": JUDGE_VERSION,
            "judge_model_path": str(self.model_path),
            "judge_model_checksum": self.model_checksum,
            "judge_tokenizer_hash": self.tokenizer_hash,
            "judge_chat_template_hash": self.chat_template_hash,
            "judge_gpu_id": self.gpu_id,
            "judge_port": self.port,
            "judge_max_model_len": self.max_model_len,
            "judge_gpu_memory_utilization": self.gpu_memory_utilization,
            "judge_server_owned": self.launch_server,
            "judge_completion_max_tokens": _JUDGE_COMPLETION_MAX_TOKENS,
            "judge_completion_temperature": _JUDGE_COMPLETION_TEMPERATURE,
            "judge_max_retries": _JUDGE_MAX_RETRIES,
            "judge_request_timeout_s": self.request_timeout_s,
            "judge_cache": "sha256-exact-input-success-only",
            "judge_backend": "local_vllm",
        }


# ------------------------------------------------------------------
# Remote OpenAI-compatible API judge backend
# ------------------------------------------------------------------

class RemoteJudgeClient:
    """Calls an external OpenAI-compatible API for judge inference.

    This backend requires no local GPU.  It uses the ``openai`` async
    client to call a remote endpoint such as DeepSeek V4 Pro.

    Parameters
    ----------
    api_base:
        Base URL of the remote API (e.g. ``https://www.yuanjvai.com``).
    api_key:
        Authentication key for the remote API.
    model_name:
        Model identifier to pass in the request payload
        (e.g. ``deepseek-ai/deepseek-v4-pro``).
    max_tokens:
        Default ``max_tokens`` for judge completions.
    temperature:
        Sampling temperature — should be 0 for deterministic judging.
    request_timeout_s:
        Per-request timeout in seconds.
    max_retries:
        Number of retry attempts on transient failures.
    retry_backoff_s:
        Base backoff interval (multiplied by attempt number).
    """

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model_name: str,
        max_tokens: int = _JUDGE_COMPLETION_MAX_TOKENS,
        temperature: float = _JUDGE_COMPLETION_TEMPERATURE,
        request_timeout_s: float = 30.0,
        max_retries: int = _JUDGE_MAX_RETRIES,
        retry_backoff_s: float = _JUDGE_RETRY_BACKOFF_S,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must be non-empty")
        if not api_base.strip():
            raise ValueError("api_base must be non-empty")
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")

        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

        self._client: Any = None  # openai.AsyncOpenAI
        self._started = False
        self._result_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize the async OpenAI client."""
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=f"{self.api_base}/v1",
            api_key=self.api_key,
            timeout=self.request_timeout_s,
            max_retries=0,  # we handle retries ourselves
        )
        self._started = True
        logger.info(
            "Remote judge client started: api_base=%s, model=%s",
            self.api_base,
            self.model_name,
        )

    async def shutdown(self) -> None:
        """Close the async OpenAI client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._started = False

    # ------------------------------------------------------------------
    # Judge inference
    # ------------------------------------------------------------------

    async def judge_structured(
        self,
        prompt: str,
        *,
        system_prompt: str,
        parser: Callable[[str], _T | None],
        cache_namespace: str,
        max_tokens: int = _JUDGE_COMPLETION_MAX_TOKENS,
        structured_outputs: dict[str, Any] | None = None,
    ) -> JudgeCallResult[_T] | None:
        """Call the remote Judge API, retry parse failures, and cache exact inputs.

        Semantics are identical to :meth:`JudgeServerManager.judge_structured`.
        """

        if not self._started:
            logger.error("judge_structured() called before start().")
            return None
        if not cache_namespace.strip():
            raise ValueError("cache_namespace must be non-empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        cache_payload = {
            "namespace": cache_namespace,
            "model": self.model_name,
            "api_base": self.api_base,
            "system_prompt": system_prompt,
            "user_prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": 1.0,
            "enable_thinking": False,
            "structured_outputs": structured_outputs,
        }
        input_hash = hashlib.sha256(
            json.dumps(
                cache_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if input_hash in self._result_cache:
            return JudgeCallResult(
                value=copy.deepcopy(self._result_cache[input_hash]),
                input_hash=input_hash,
                cache_hit=True,
                attempts=0,
                latency_s=0.0,
            )

        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=1.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                text = response.choices[0].message.content
                result = parser(text)
                if result is not None:
                    self._result_cache[input_hash] = copy.deepcopy(result)
                    return JudgeCallResult(
                        value=result,
                        input_hash=input_hash,
                        cache_hit=False,
                        attempts=attempt,
                        latency_s=time.monotonic() - started,
                    )
                logger.warning(
                    "Remote Judge output did not pass %s parser (attempt %d): %r",
                    cache_namespace,
                    attempt,
                    text[:500] if text else "<empty>",
                )
                last_error = ValueError(
                    f"Invalid {cache_namespace} Judge output: {text!r}"
                )
            except Exception as exc:
                # openai library raises openai.APIError subclasses
                logger.warning(
                    "Remote Judge request failed (attempt %d): %s: %s",
                    attempt,
                    type(exc).__name__,
                    exc,
                )
                last_error = exc

            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_backoff_s * attempt)

        logger.error(
            "All %d %s remote Judge attempts failed. Last error: %s",
            self.max_retries,
            cache_namespace,
            last_error,
        )
        return None

    async def judge(
        self,
        prompt: str,
        allowed_fact_ids: set[str] | None = None,
    ) -> float | set[str] | None:
        """Call the remote judge model and return the result, or ``None`` on failure.

        Identical semantics to :meth:`JudgeServerManager.judge`.
        """
        if allowed_fact_ids is not None:
            allowed = frozenset(allowed_fact_ids)
            result = await self.judge_structured(
                prompt,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                parser=lambda text: parse_judge_fact_ids(text, allowed),
                cache_namespace="a6-fact-id:" + ",".join(sorted(allowed)),
            )
        else:
            result = await self.judge_structured(
                prompt,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                parser=parse_judge_score,
                cache_namespace="legacy-a6-scalar",
                structured_outputs={"choice": _JUDGE_SCORE_CHOICES},
            )
        return result.value if result is not None else None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def identity_summary(self) -> dict[str, Any]:
        """Return a dict of remote judge identity fields for the manifest."""
        return {
            "judge_version": JUDGE_VERSION,
            "judge_backend": "remote_api",
            "judge_api_base": self.api_base,
            "judge_model_name": self.model_name,
            "judge_model_checksum": None,  # not available for remote models
            "judge_tokenizer_hash": None,
            "judge_chat_template_hash": None,
            "judge_gpu_id": None,  # no local GPU
            "judge_port": None,
            "judge_max_model_len": None,  # determined by remote API
            "judge_gpu_memory_utilization": None,
            "judge_server_owned": False,
            "judge_completion_max_tokens": self.max_tokens,
            "judge_completion_temperature": self.temperature,
            "judge_max_retries": self.max_retries,
            "judge_request_timeout_s": self.request_timeout_s,
            "judge_cache": "sha256-exact-input-success-only",
        }


# ------------------------------------------------------------------
# Factory: choose local or remote based on environment
# ------------------------------------------------------------------

def create_judge_from_env(reward_arm: Any) -> JudgeServerManager | RemoteJudgeClient:
    """Create a judge backend based on environment variables (A6 only).

    If ``HOTPOTQA_JUDGE_API_KEY`` is set, a :class:`RemoteJudgeClient` is
    returned pointing to the configured remote API (default:
    ``https://www.yuanjvai.com`` with model ``deepseek-ai/deepseek-v4-pro``).

    Otherwise, a local :class:`JudgeServerManager` is returned using the
    existing vLLM-based configuration.

    Parameters
    ----------
    reward_arm:
        The :class:`RewardArm` enum value (must be A6).
    """
    api_key = os.environ.get("HOTPOTQA_JUDGE_API_KEY", "").strip()
    if api_key:
        return RemoteJudgeClient(
            api_base=os.environ.get(
                "HOTPOTQA_JUDGE_API_BASE",
                "https://www.yuanjvai.com",
            ),
            api_key=api_key,
            model_name=os.environ.get(
                "HOTPOTQA_JUDGE_MODEL",
                "deepseek-ai/deepseek-v4-pro",
            ),
            max_tokens=int(
                os.environ.get(
                    "HOTPOTQA_JUDGE_COMPLETION_MAX_TOKENS",
                    "256",
                )
            ),
            temperature=float(
                os.environ.get(
                    "HOTPOTQA_JUDGE_TEMPERATURE",
                    str(_JUDGE_COMPLETION_TEMPERATURE),
                )
            ),
            request_timeout_s=float(
                os.environ.get(
                    "HOTPOTQA_JUDGE_REQUEST_TIMEOUT_S",
                    "30",
                )
            ),
        )

    # Fallback: local vLLM server
    workspace_dir = os.environ.get(
        "WORKSPACE_DIR",
        os.path.join(os.path.dirname(__file__), "..", ".."),
    )
    default_judge_model = os.path.join(workspace_dir, "models", "Qwen3-4B")
    return JudgeServerManager(
        model_path=os.environ.get(
            "HOTPOTQA_JUDGE_MODEL",
            default_judge_model,
        ),
        gpu_id=int(os.environ.get("HOTPOTQA_JUDGE_GPU", "1")),
        port=int(os.environ.get("HOTPOTQA_JUDGE_PORT", "29500")),
        max_model_len=int(os.environ.get("HOTPOTQA_JUDGE_MAX_MODEL_LEN", "4096")),
        gpu_memory_utilization=float(
            os.environ.get("HOTPOTQA_JUDGE_GPU_MEMORY_UTILIZATION", "0.50")
        ),
        launch_server=os.environ.get("HOTPOTQA_JUDGE_EXTERNAL", "").strip().lower()
        not in {"1", "true", "yes", "on"},
    )


def create_shared_judge_from_env() -> JudgeServerManager | RemoteJudgeClient:
    """Create the cross-domain frozen Judge from ``AGENT_R1_JUDGE_*`` settings."""
    api_key = os.environ.get("AGENT_R1_JUDGE_API_KEY", "").strip()
    if api_key:
        return RemoteJudgeClient(
            api_base=os.environ.get("AGENT_R1_JUDGE_API_BASE", "http://127.0.0.1:29500"),
            api_key=api_key,
            model_name=os.environ.get("AGENT_R1_JUDGE_MODEL", "Qwen3.5-9B"),
            max_tokens=int(os.environ.get("AGENT_R1_JUDGE_COMPLETION_MAX_TOKENS", "256")),
            temperature=0.0,
            request_timeout_s=float(os.environ.get("AGENT_R1_JUDGE_REQUEST_TIMEOUT_S", "30")),
        )

    workspace_dir = os.environ.get(
        "WORKSPACE_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    )
    return JudgeServerManager(
        model_path=os.environ.get(
            "AGENT_R1_JUDGE_MODEL",
            os.path.join(workspace_dir, "models", "Qwen3.5-9B"),
        ),
        gpu_id=int(os.environ.get("AGENT_R1_JUDGE_GPU", "7")),
        port=int(os.environ.get("AGENT_R1_JUDGE_PORT", "29500")),
        max_model_len=int(os.environ.get("AGENT_R1_JUDGE_MAX_MODEL_LEN", "8192")),
        gpu_memory_utilization=float(os.environ.get("AGENT_R1_JUDGE_GPU_MEMORY_UTILIZATION", "0.80")),
        request_timeout_s=float(os.environ.get("AGENT_R1_JUDGE_REQUEST_TIMEOUT_S", "120")),
        launch_server=os.environ.get("AGENT_R1_JUDGE_EXTERNAL", "").strip().lower()
        not in {"1", "true", "yes", "on"},
    )
