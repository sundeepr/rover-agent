"""
Abstract base class for all LLM navigation clients.

Any new provider (Anthropic, Gemini, OpenAI, …) must implement this interface
so the NavigationLoop can use them interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rover_agent.models import LLMNavigationResponse


class NavigationLLMClient(ABC):
    """Send an image + context to an LLM and get back a navigation response."""

    @abstractmethod
    def get_next_waypoint(
        self,
        image_bytes: bytes,
        user_goal: str,
        navigation_history: list[str],
        position_summary: str,
    ) -> LLMNavigationResponse:
        """
        Analyse the image in the context of the goal and return the next waypoint.

        Raises:
            ValueError: The model returned unparseable output.
            pydantic.ValidationError: The model's JSON doesn't match the schema.
        """
        ...
