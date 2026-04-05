"""
Generate CLIP zero-shot classification prompts from a navigation goal
using a local Qwen3 text model via Ollama.

Called once at agent startup before CLIP encodes the prompts. Falls back
to simple template-based prompts if Ollama is not reachable or returns
an invalid response — so the agent always starts even without Ollama.

Usage:
    from prompt_generator import generate_clip_prompts
    prompts = generate_clip_prompts("Follow the brown path",
                                    ollama_url="http://localhost:11434")
    # {"positive": [...], "negative": [...]}
"""

import json
import logging
import re

log = logging.getLogger("rover.prompt_generator")

_SYSTEM = (
    "You are a computer vision assistant for a ground robot. "
    "Given a navigation goal, generate short natural-language image "
    "description prompts for CLIP zero-shot classification. "
    "Positive prompts describe camera images where the goal IS visible. "
    "Negative prompts describe images where it is NOT visible. "
    "Keep each prompt under 10 words. "
    "Use indoor vocabulary (floor, tape, strip, room). "
    "Respond with valid JSON only."
)

_USER = (
    "Navigation goal: {goal}\n\n"
    "Generate 3–4 positive and 3–4 negative CLIP image description prompts.\n\n"
    "Respond with JSON only:\n"
    '{{"positive": ["...", "..."], "negative": ["...", "..."]}}'
)

# Strips leading navigation verb so "Follow the brown path" → "brown path"
_VERB_RE = re.compile(
    r"^\s*(follow|navigate\s+to|go\s+to|find|reach|head\s+to|move\s+to)\s+(the\s+)?",
    re.IGNORECASE,
)


def _template_prompts(goal: str) -> dict:
    """Simple template fallback — no LLM required."""
    target = _VERB_RE.sub("", goal).strip() or goal
    return {
        "positive": [
            f"{target} visible ahead",
            f"{target} on the floor",
            f"following {target}",
        ],
        "negative": [
            f"no {target} visible",
            f"end of {target}",
            f"floor without {target}",
        ],
    }


def generate_clip_prompts(
    goal: str,
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen3:4b",
) -> dict:
    """
    Ask Qwen3 to generate positive/negative CLIP prompts for the goal.

    Returns {"positive": [...], "negative": [...]}.
    Falls back to template prompts if Ollama is unreachable or returns
    invalid JSON — the agent always gets usable prompts.
    """
    log.info("Generating CLIP prompts for '%s' via %s (%s)…", goal, model, ollama_url)
    try:
        import requests

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                # /no_think disables Qwen3 chain-of-thought for a faster,
                # direct JSON response — reasoning is not needed here.
                {"role": "user", "content": f"/no_think\n{_USER.format(goal=goal)}"},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2, "num_predict": 256},
        }
        r = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=60)
        r.raise_for_status()

        content = r.json()["message"]["content"]
        prompts = json.loads(content)

        if not (isinstance(prompts.get("positive"), list) and
                isinstance(prompts.get("negative"), list) and
                prompts["positive"] and prompts["negative"]):
            raise ValueError("unexpected response structure")

        log.info("Positive prompts : %s", prompts["positive"])
        log.info("Negative prompts : %s", prompts["negative"])
        return prompts

    except Exception as exc:
        log.warning("Ollama unavailable (%s) — falling back to template prompts", exc)
        prompts = _template_prompts(goal)
        log.info("Template positive: %s", prompts["positive"])
        log.info("Template negative: %s", prompts["negative"])
        return prompts
