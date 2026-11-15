"""
Tests for OpenAI /v1 endpoint compatibility (Strata, vLLM, llama.cpp, LocalAI).
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from bookeeper.processing.extractor import KnowledgeExtractor, SectionExtraction
from bookeeper.processing.verifier import IdeaVerifier, VerificationResult
from bookeeper.processing.ollama_pool import OllamaPool


def test_extractor_openai_compat_invoke():
    """Verify that KnowledgeExtractor formats request for OpenAI /v1 endpoint and parses choices."""
    extractor = KnowledgeExtractor(
        pool=OllamaPool.from_urls(["http://192.168.50.121:8080/v1"]),
        model="iq_xs",
    )

    mock_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "concepts": [
                            {
                                "name": "Latent Space Alignment",
                                "category": "Method",
                                "brief_description": "Aligning representations across neural architectures.",
                                "detailed_explanation": "Aligning representations across neural architectures via projection heads.",
                                "weight": 8,
                                "supporting_quote": "We align latent space across architectures.",
                                "related_concepts": ["JEPA", "Representation Learning"],
                            }
                        ]
                    }),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
    }

    mock_res = MagicMock()
    mock_res.returncode = 0
    mock_res.stdout = json.dumps(mock_response)

    with patch("subprocess.run", return_value=mock_res) as mock_run:
        result = extractor._execute_structured_invoke(
            schema_cls=SectionExtraction,
            messages=["Extract concepts from text"],
        )

        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        # Must target /chat/completions endpoint
        assert call_args[-1] == "http://192.168.50.121:8080/v1/chat/completions"

        # Verify payload contains OpenAI response_format and model
        payload = json.loads(call_args[call_args.index("-d") + 1])
        assert payload["model"] == "iq_xs"
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["messages"][0]["content"] == "Extract concepts from text"

        # Verify parsed output
        assert isinstance(result, SectionExtraction)
        assert len(result.concepts) == 1
        assert result.concepts[0].name == "Latent Space Alignment"


def test_verifier_openai_compat_invoke():
    """Verify that IdeaVerifier formats request for OpenAI /v1 endpoint and parses choices."""
    verifier = IdeaVerifier(
        pool=OllamaPool.from_urls(["http://192.168.50.121:8080/v1"]),
        model_name="iq_xs",
    )

    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "is_supported": True,
                        "confidence": 0.95,
                        "explanation": "Strong textual evidence found in chunk.",
                    }),
                }
            }
        ],
        "usage": {"prompt_tokens": 80, "completion_tokens": 25},
    }

    mock_res = MagicMock()
    mock_res.returncode = 0
    mock_res.stdout = json.dumps(mock_response)

    with patch("subprocess.run", return_value=mock_res) as mock_run:
        result = verifier._execute_structured_invoke(
            messages=[MagicMock(type="user", content="Verify idea grounding in chunk text")],
            input_chars=40,
        )

        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert call_args[-1] == "http://192.168.50.121:8080/v1/chat/completions"
        payload = json.loads(call_args[call_args.index("-d") + 1])
        assert payload["model"] == "iq_xs"
        assert payload["response_format"] == {"type": "json_object"}

        assert isinstance(result, VerificationResult)
        assert result.is_supported is True
        assert result.confidence == 0.95
