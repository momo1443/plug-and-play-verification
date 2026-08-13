import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from recipes.hotpotqa.judge_server import (
    JudgeServerManager,
    RemoteJudgeClient,
    create_judge_from_env,
)
from recipes.hotpotqa.reward_arm import RewardArm


# ------------------------------------------------------------------
# Helpers for local JudgeServerManager tests (existing)
# ------------------------------------------------------------------

class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def json(self):
        return {"choices": [{"message": {"content": "0.75"}}]}


class _FakeSession:
    """Mimics aiohttp.ClientSession with a mock post()."""

    def __init__(self):
        self.url = None
        self.payload = None
        self.closed = False
        self._connector_closed = False

    @property
    def connector(self):
        """Return a connector-like object whose close() is awaitable."""
        conn = MagicMock()
        conn.closed = self._connector_closed

        async def _async_close():
            self._connector_closed = True

        conn.close = _async_close
        return conn

    def post(self, url, *, json):
        self.url = url
        self.payload = json
        return _FakeResponse()

    async def close(self):
        self.closed = True


class JudgeServerManagerTest(unittest.TestCase):
    def test_external_mode_attaches_without_spawning_a_server(self):
        manager = JudgeServerManager(
            model_path="/tmp/nonexistent-judge-model",
            gpu_id=1,
            launch_server=False,
        )
        with (
            patch.object(manager, "_wait_healthy", new=AsyncMock()) as wait_healthy,
            patch.object(manager, "_compute_model_hashes") as compute_hashes,
            patch("recipes.hotpotqa.judge_server.subprocess.Popen") as popen,
        ):
            asyncio.run(manager.start())

        try:
            wait_healthy.assert_not_awaited()
            compute_hashes.assert_not_called()
            popen.assert_not_called()
            # In the updated code, start(launch_server=False) creates a
            # persistent session via _ensure_session() to avoid fd-conflict
            # crashes under uvloop.
            self.assertIsNotNone(manager._session)
            self.assertFalse(manager._session.closed)
            self.assertTrue(manager._started)
        finally:
            # Clean up the real aiohttp session to avoid ResourceWarning.
            asyncio.run(manager.shutdown())

    def test_judge_uses_non_thinking_chat_with_constrained_scores(self):
        """Legacy scalar mode: structured_outputs.choice is set."""
        manager = JudgeServerManager(
            model_path="/tmp/nonexistent-judge-model",
            gpu_id=1,
            launch_server=False,
        )
        session = _FakeSession()
        manager._session = session
        manager._started = True

        score = asyncio.run(manager.judge("test prompt"))

        self.assertEqual(score, 0.75)
        self.assertEqual(session.url, "http://127.0.0.1:29500/v1/chat/completions")
        self.assertEqual(session.payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(
            session.payload["structured_outputs"]["choice"],
            ["0.0", "0.25", "0.5", "0.75", "1.0"],
        )
        self.assertEqual(session.payload["messages"][1]["content"], "test prompt")
        # The persistent session must NOT be closed after a successful call.
        self.assertFalse(session.closed)

    def test_judge_factid_mode_omits_choice_constraint(self):
        """Fact-ID mode (allowed_fact_ids provided): no structured_outputs."""
        manager = JudgeServerManager(
            model_path="/tmp/nonexistent-judge-model",
            gpu_id=1,
            launch_server=False,
        )

        class _FactIdResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def json(self):
                return {"choices": [{"message": {"content": '["fact-1"]'}}]}

        class _FactIdSession:
            def __init__(self):
                self.url = None
                self.payload = None
                self.closed = False
                self._connector_closed = False

            @property
            def connector(self):
                conn = MagicMock()
                conn.closed = self._connector_closed

                async def _async_close():
                    self._connector_closed = True

                conn.close = _async_close
                return conn

            def post(self, url, *, json):
                self.url = url
                self.payload = json
                return _FactIdResponse()

            async def close(self):
                self.closed = True

        session = _FactIdSession()
        manager._session = session
        manager._started = True

        result = asyncio.run(
            manager.judge("test prompt", allowed_fact_ids={"fact-1", "fact-2"})
        )

        self.assertEqual(result, {"fact-1"})
        # No structured_outputs in fact-ID mode
        self.assertNotIn("structured_outputs", session.payload)

    def test_session_recreated_on_transport_error(self):
        """After a RuntimeError (fd conflict), session is recreated."""
        manager = JudgeServerManager(
            model_path="/tmp/nonexistent-judge-model",
            gpu_id=1,
            launch_server=False,
        )
        session = _FakeSession()
        manager._session = session
        manager._started = True

        call_count = 0

        original_post = session.post

        def failing_post(url, *, json):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("File descriptor 42 is used by transport")
            return original_post(url, json=json)

        session.post = failing_post

        # After the RuntimeError, _recreate_session creates a new session.
        # We need to mock _ensure_session to provide a working session for retry.
        new_session = _FakeSession()
        manager._ensure_session = lambda: setattr(manager, "_session", new_session)

        score = asyncio.run(manager.judge("test prompt"))

        # The retry on the new session should succeed
        self.assertEqual(score, 0.75)
        # Old session should have been replaced
        self.assertIs(manager._session, new_session)

    def test_shutdown_closes_session(self):
        """shutdown() should close the persistent session."""
        manager = JudgeServerManager(
            model_path="/tmp/nonexistent-judge-model",
            gpu_id=1,
            launch_server=False,
        )
        session = _FakeSession()
        manager._session = session
        manager._started = True

        asyncio.run(manager.shutdown())

        self.assertTrue(session.closed)
        self.assertIsNone(manager._session)
        self.assertFalse(manager._started)

    def test_identity_summary_includes_backend_type(self):
        manager = JudgeServerManager(
            model_path="/tmp/nonexistent-judge-model",
            gpu_id=1,
            launch_server=False,
        )
        summary = manager.identity_summary()
        self.assertEqual(summary["judge_backend"], "local_vllm")


# ------------------------------------------------------------------
# RemoteJudgeClient tests
# ------------------------------------------------------------------

class RemoteJudgeClientTest(unittest.TestCase):
    def _make_client(self, **overrides) -> RemoteJudgeClient:
        defaults = {
            "api_base": "https://api.example.com",
            "api_key": "test-key-123",
            "model_name": "test-model",
        }
        defaults.update(overrides)
        return RemoteJudgeClient(**defaults)

    def test_constructor_validates_api_key(self):
        with self.assertRaises(ValueError):
            RemoteJudgeClient(api_base="https://x.com", api_key="", model_name="m")

    def test_constructor_validates_api_base(self):
        with self.assertRaises(ValueError):
            RemoteJudgeClient(api_base="", api_key="k", model_name="m")

    def test_constructor_validates_model_name(self):
        with self.assertRaises(ValueError):
            RemoteJudgeClient(api_base="https://x.com", api_key="k", model_name="")

    def test_start_initializes_client(self):
        client = self._make_client()
        mock_instance = MagicMock()
        mock_instance.close = AsyncMock()
        with patch("openai.AsyncOpenAI", return_value=mock_instance) as mock_openai:
            asyncio.run(client.start())

        mock_openai.assert_called_once()
        self.assertTrue(client._started)
        self.assertIsNotNone(client._client)

        # Cleanup
        asyncio.run(client.shutdown())
        self.assertFalse(client._started)

    def test_judge_structured_returns_parsed_result(self):
        client = self._make_client()
        client._started = True

        # Mock the openai client's chat.completions.create
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.75"

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        client._client = mock_client

        from recipes.hotpotqa.judge_prompts import parse_judge_score

        result = asyncio.run(
            client.judge_structured(
                "test prompt",
                system_prompt="You are a judge.",
                parser=parse_judge_score,
                cache_namespace="test",
            )
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.value, 0.75)
        self.assertFalse(result.cache_hit)
        self.assertEqual(result.attempts, 1)

    def test_judge_structured_uses_cache_on_repeat(self):
        client = self._make_client()
        client._started = True

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.75"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        client._client = mock_client

        from recipes.hotpotqa.judge_prompts import parse_judge_score

        result1 = asyncio.run(
            client.judge_structured(
                "test prompt",
                system_prompt="You are a judge.",
                parser=parse_judge_score,
                cache_namespace="test",
            )
        )
        result2 = asyncio.run(
            client.judge_structured(
                "test prompt",
                system_prompt="You are a judge.",
                parser=parse_judge_score,
                cache_namespace="test",
            )
        )

        self.assertFalse(result1.cache_hit)
        self.assertTrue(result2.cache_hit)
        self.assertEqual(result1.value, result2.value)
        # API should only be called once (second hit cache)
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

    def test_judge_structured_returns_none_on_all_retries_fail(self):
        client = self._make_client(max_retries=2)
        client._started = True

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API error")
        )
        client._client = mock_client

        from recipes.hotpotqa.judge_prompts import parse_judge_score

        result = asyncio.run(
            client.judge_structured(
                "test prompt",
                system_prompt="You are a judge.",
                parser=parse_judge_score,
                cache_namespace="test",
            )
        )

        self.assertIsNone(result)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    def test_judge_structured_returns_none_if_not_started(self):
        client = self._make_client()
        # Not started
        from recipes.hotpotqa.judge_prompts import parse_judge_score

        result = asyncio.run(
            client.judge_structured(
                "test prompt",
                system_prompt="You are a judge.",
                parser=parse_judge_score,
                cache_namespace="test",
            )
        )
        self.assertIsNone(result)

    def test_identity_summary_shows_remote_backend(self):
        client = self._make_client()
        summary = client.identity_summary()
        self.assertEqual(summary["judge_backend"], "remote_api")
        self.assertEqual(summary["judge_api_base"], "https://api.example.com")
        self.assertEqual(summary["judge_model_name"], "test-model")
        self.assertIsNone(summary["judge_model_checksum"])
        self.assertIsNone(summary["judge_gpu_id"])

    def test_judge_convenience_method(self):
        client = self._make_client()
        client._started = True

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.5"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        client._client = mock_client

        result = asyncio.run(client.judge("test prompt"))
        self.assertEqual(result, 0.5)


# ------------------------------------------------------------------
# create_judge_from_env tests (A6 only)
# ------------------------------------------------------------------

class CreateJudgeFromEnvTest(unittest.TestCase):
    def test_returns_remote_client_when_api_key_set(self):
        with patch.dict(
            os.environ,
            {
                "HOTPOTQA_JUDGE_API_KEY": "test-key",
                "HOTPOTQA_JUDGE_API_BASE": "https://api.test.com",
                "HOTPOTQA_JUDGE_MODEL": "test-model-v1",
            },
            clear=False,
        ):
            judge = create_judge_from_env(RewardArm.A6)
        self.assertIsInstance(judge, RemoteJudgeClient)
        self.assertEqual(judge.api_base, "https://api.test.com")
        self.assertEqual(judge.model_name, "test-model-v1")

    def test_returns_local_manager_when_no_api_key(self):
        with patch.dict(
            os.environ,
            {"HOTPOTQA_JUDGE_MODEL": "/tmp/test-model"},
            clear=False,
        ):
            # Remove API key if present
            env = os.environ.copy()
            env.pop("HOTPOTQA_JUDGE_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                judge = create_judge_from_env(RewardArm.A6)
        self.assertIsInstance(judge, JudgeServerManager)

    def test_default_remote_config_for_a6(self):
        with patch.dict(
            os.environ,
            {"HOTPOTQA_JUDGE_API_KEY": "test-key"},
            clear=False,
        ):
            env = os.environ.copy()
            for key in [
                "HOTPOTQA_JUDGE_API_BASE",
                "HOTPOTQA_JUDGE_MODEL",
                "HOTPOTQA_JUDGE_COMPLETION_MAX_TOKENS",
            ]:
                env.pop(key, None)
            with patch.dict(os.environ, env, clear=True):
                judge = create_judge_from_env(RewardArm.A6)
        self.assertIsInstance(judge, RemoteJudgeClient)
        self.assertEqual(judge.max_tokens, 256)  # A6 default


if __name__ == "__main__":
    unittest.main()
