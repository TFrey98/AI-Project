import json

import pytest

from story_model.backbones import (
    ChatMessage,
    GenerationSettings,
    LocalOpenAIBackbone,
    ScriptedBackbone,
)


class FakeHTTPResponse:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.data).encode("utf-8")


def test_scripted_backbone_records_exact_messages_and_settings():
    backend = ScriptedBackbone(["The tunnel is open."])
    messages = (
        ChatMessage("system", "Use the supplied scene."),
        ChatMessage("user", "Which route is open?"),
    )
    settings = GenerationSettings(seed=41)

    response = backend.generate(messages, settings)

    assert response.text == "The tunnel is open."
    assert response.backend == "scripted"
    assert response.seed == 41
    assert backend.calls == [(messages, settings)]


def test_scripted_backbone_rejects_exhausted_response_queue():
    backend = ScriptedBackbone([])

    with pytest.raises(RuntimeError, match="no responses remaining"):
        backend.generate(
            [ChatMessage("user", "Hello")],
            GenerationSettings(),
        )


@pytest.mark.parametrize(
    "endpoint",
    (
        "https://example.com/v1/chat/completions",
        "http://192.168.1.25:8080/v1/chat/completions",
        "file:///tmp/model.sock",
    ),
)
def test_local_backend_rejects_non_loopback_endpoints(endpoint):
    with pytest.raises(ValueError, match="localhost or a loopback"):
        LocalOpenAIBackbone("test-model", endpoint=endpoint)


def test_local_backend_normalizes_openai_response(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {"content": "Use the tunnel."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 4,
                },
            }
        )

    monkeypatch.setattr("story_model.backbones.urlopen", fake_urlopen)
    backend = LocalOpenAIBackbone(
        "local-test",
        endpoint="http://localhost:8080/v1/chat/completions",
        timeout_seconds=5.0,
    )
    response = backend.generate(
        [ChatMessage("user", "Which route?")],
        GenerationSettings(
            max_new_tokens=40,
            temperature=0.1,
            top_p=0.8,
            seed=7,
        ),
    )

    assert response.text == "Use the tunnel."
    assert response.prompt_tokens == 20
    assert response.completion_tokens == 4
    assert captured["url"].startswith("http://localhost:8080")
    assert captured["timeout"] == 5.0
    assert captured["payload"]["model"] == "local-test"
    assert captured["payload"]["seed"] == 7
    assert captured["payload"]["stream"] is False
