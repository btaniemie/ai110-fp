"""
MusicAgent: Agentic music recommender powered by the Claude API.

Architecture:
  natural language query
       ↓
  Claude (claude-haiku-4-5) — parses intent, calls search_songs tool
       ↓  (tool_use)
  search_songs → recommend_songs() → songs.csv
       ↓  (tool_result)
  Claude synthesises a personalised response
       ↓
  structured dict: explanation + ranked songs + confidence score
"""

import json
import logging
import os
from typing import Any, Dict, List

import anthropic

from src.recommender import load_songs, recommend_songs

logger = logging.getLogger(__name__)

# Maximum possible score from recommend_songs (genre 2.0 + mood 1.0 + energy 1.5 + valence 1.0)
_MAX_SCORE = 5.5

SYSTEM_PROMPT = """You are a friendly, knowledgeable music recommendation assistant.
When a user describes what they want to listen to, call the search_songs tool to retrieve
matching tracks from the catalog. Then write a warm, personalised explanation of why each
song fits their request — mention specific mood, energy, or style elements.
Always call search_songs before responding. Never recommend songs from memory."""

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "search_songs",
        "description": (
            "Search the music catalog and rank songs based on user preferences. "
            "Returns the top-K matching songs with scores and reasons."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "genre": {
                    "type": "string",
                    "description": (
                        "Preferred genre, e.g. pop, rock, lofi, jazz, hip-hop, "
                        "electronic, classical, ambient, country, folk, r&b, "
                        "metal, reggae, synthwave, indie pop"
                    ),
                },
                "mood": {
                    "type": "string",
                    "description": (
                        "Desired mood, e.g. happy, chill, intense, relaxed, moody, "
                        "energetic, romantic, melancholic, peaceful, focused, confident"
                    ),
                },
                "energy": {
                    "type": "number",
                    "description": "Target energy 0.0 (very calm) to 1.0 (very intense)",
                },
                "valence": {
                    "type": "number",
                    "description": (
                        "Target positivity 0.0 (dark/sad) to 1.0 (uplifting/happy). "
                        "Optional — defaults to 0.6."
                    ),
                },
                "k": {
                    "type": "integer",
                    "description": "Number of songs to return (default 5)",
                },
            },
            "required": ["genre", "mood", "energy"],
        },
    }
]


class MusicAgent:
    """
    Agentic music recommender that combines RAG (retrieval from songs.csv)
    with an LLM agentic loop to handle natural language queries.
    """

    def __init__(self, songs_path: str, model: str = "claude-haiku-4-5-20251001"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it or add it to a .env file before running the agent."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.songs = load_songs(songs_path)
        self.model = model
        logger.info("MusicAgent ready — %d songs loaded, model=%s", len(self.songs), model)

    def handle_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a natural language music request through an agentic loop.

        Returns:
            {
                "explanation": str,          # Claude's personalised response
                "recommendations": list,     # structured song dicts with confidence
                "confidence": float,         # average confidence across top songs
            }
        """
        user_query = (user_query or "").strip()
        if not user_query:
            logger.warning("Empty query — skipping agent call")
            return {
                "explanation": "Please describe what kind of music you are looking for.",
                "recommendations": [],
                "confidence": 0.0,
            }

        messages: List[Dict] = [{"role": "user", "content": user_query}]
        recommendations: List[Dict] = []
        logger.info("Query: %r", user_query)

        # Agentic loop — capped at 5 iterations as a safety guard
        for iteration in range(5):
            logger.debug("Agent iteration %d", iteration + 1)

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                # Cache the static system prompt to reduce latency and cost on repeat calls
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=TOOLS,
                messages=messages,
            )

            logger.debug(
                "stop_reason=%s  input_tokens=%s  output_tokens=%s",
                response.stop_reason,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

            if response.stop_reason == "end_turn":
                text = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                avg_conf = (
                    sum(r["confidence"] for r in recommendations) / len(recommendations)
                    if recommendations
                    else 0.0
                )
                logger.info(
                    "Resolved — %d recommendations, avg confidence=%.2f",
                    len(recommendations),
                    avg_conf,
                )
                return {
                    "explanation": text,
                    "recommendations": recommendations,
                    "confidence": round(avg_conf, 2),
                }

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    logger.info("Tool call: %s(%s)", block.name, json.dumps(block.input))
                    try:
                        result = self._execute_tool(block.name, block.input)
                        recommendations = result.get("songs", [])
                    except Exception as exc:
                        logger.error("Tool %s failed: %s", block.name, exc)
                        result = {"error": str(exc), "songs": []}

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )

                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason (e.g., max_tokens)
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

        logger.error("Agent loop exhausted without resolution")
        return {
            "explanation": "Something went wrong. Please try again.",
            "recommendations": [],
            "confidence": 0.0,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, tool_input: Dict) -> Dict:
        """Dispatch a tool call and return its result dict."""
        if tool_name != "search_songs":
            raise ValueError(f"Unknown tool: {tool_name!r}")

        user_prefs = {
            "genre": tool_input.get("genre", ""),
            "mood": tool_input.get("mood", ""),
            "energy": float(tool_input.get("energy", 0.5)),
            "valence": float(tool_input.get("valence", 0.6)),
        }
        k = int(tool_input.get("k", 5))

        recs = recommend_songs(user_prefs, self.songs, k=k)

        songs_out = []
        for song, score, explanation in recs:
            confidence = round(min(score / _MAX_SCORE, 1.0), 2)
            songs_out.append(
                {
                    "title": song["title"],
                    "artist": song["artist"],
                    "genre": song["genre"],
                    "mood": song["mood"],
                    "energy": song["energy"],
                    "score": round(score, 2),
                    "confidence": confidence,
                    "why": explanation,
                }
            )

        logger.info("search_songs returned %d results for prefs=%s", len(songs_out), user_prefs)
        return {"songs": songs_out, "catalog_size": len(self.songs)}
