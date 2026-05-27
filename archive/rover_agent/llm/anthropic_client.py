"""
Anthropic (Claude) LLM client for rover navigation.
"""

from __future__ import annotations

import base64
import logging

import anthropic

from rover_agent.config import CONFIG
from rover_agent.llm.base import NavigationLLMClient
from rover_agent.llm.prompts import build_system_prompt, build_user_prompt
from rover_agent.llm.response_parser import parse_navigation_response
from rover_agent.models import LLMNavigationResponse

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicNavigationClient(NavigationLLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = CONFIG.LLM_MAX_TOKENS,
        image_width: int = CONFIG.IMAGE_WIDTH,
        image_height: int = CONFIG.IMAGE_HEIGHT,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = build_system_prompt(image_width, image_height)
        self._image_width = image_width

    def get_next_waypoint(
        self,
        image_bytes: bytes,
        user_goal: str,
        navigation_history: list[str],
        position_summary: str,
    ) -> LLMNavigationResponse:
        user_text = build_user_prompt(
            user_goal=user_goal,
            navigation_history=navigation_history,
            position_summary=position_summary,
        )
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        logger.debug("Anthropic request → %s (%d bytes)", self._model, len(image_bytes))

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        )

        raw_text = response.content[0].text
        logger.debug("Anthropic raw response: %s", raw_text[:300])
        return parse_navigation_response(raw_text)
