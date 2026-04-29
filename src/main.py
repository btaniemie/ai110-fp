"""
Music Recommender Simulation — CLI runner.

Usage
-----
Batch simulation (original behaviour, no API key required):
    python -m src.main

Interactive agent — single query:
    python -m src.main --agent "I want something chill to study to"

Interactive agent — REPL (type 'quit' to exit):
    python -m src.main --agent
"""

import argparse
import logging
import os
import sys

# Load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.recommender import load_songs, recommend_songs

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Hardcoded profiles for the original batch simulation ─────────────────────
PROFILES = {
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.85,
        "valence": 0.82,
    },
    "Chill Lofi": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.38,
        "valence": 0.58,
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.92,
        "valence": 0.35,
    },
    "Adversarial (conflicting prefs)": {
        # High energy but jazz genre — tests whether scoring handles the tension
        "genre": "jazz",
        "mood": "happy",
        "energy": 0.90,
        "valence": 0.75,
    },
}

_DATA_PATH = "data/songs.csv"


# ── Batch simulation (original) ───────────────────────────────────────────────

def print_recommendations(profile_name: str, recs, top_k: int = 5) -> None:
    print(f"\n{'='*60}")
    print(f"  Profile: {profile_name}")
    print(f"{'='*60}")
    for rank, (song, score, explanation) in enumerate(recs[:top_k], start=1):
        print(f"\n  #{rank}  {song['title']} by {song['artist']}")
        print(f"       Score : {score:.2f}")
        print(f"       Why   : {explanation}")
    print()


def run_batch() -> None:
    logger.info("Starting batch simulation")
    songs = load_songs(_DATA_PATH)
    logger.info("Loaded %d songs from %s", len(songs), _DATA_PATH)

    for profile_name, user_prefs in PROFILES.items():
        recs = recommend_songs(user_prefs, songs, k=5)
        print_recommendations(profile_name, recs)

    logger.info("Batch simulation complete")


# ── Agent mode ────────────────────────────────────────────────────────────────

def run_agent(query: str | None) -> None:
    from src.agent import MusicAgent  # import here so missing API key only errors in agent mode

    try:
        agent = MusicAgent(songs_path=_DATA_PATH)
    except EnvironmentError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if query:
        _run_single_query(agent, query)
    else:
        _run_repl(agent)


def _run_single_query(agent, query: str) -> None:
    print(f"\nQuery: {query}")
    result = agent.handle_query(query)
    _print_agent_result(result)


def _run_repl(agent) -> None:
    print("\nMusic Recommender Agent  (type 'quit' to exit)")
    print("─" * 50)
    while True:
        try:
            query = input("\nWhat do you want to listen to? ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        result = agent.handle_query(query)
        _print_agent_result(result)


def _print_agent_result(result: dict) -> None:
    recs = result.get("recommendations", [])
    confidence = result.get("confidence", 0.0)
    explanation = result.get("explanation", "")

    print(f"\n{'─'*60}")
    if recs:
        print(f"Top {len(recs)} recommendations  (avg confidence: {confidence:.0%})\n")
        for i, song in enumerate(recs, 1):
            print(f"  #{i}  {song['title']} by {song['artist']}")
            print(f"       Genre/Mood : {song['genre']} / {song['mood']}")
            print(f"       Score      : {song['score']:.2f}  (confidence: {song['confidence']:.0%})")
            print(f"       Why        : {song['why']}")
        print()

    if explanation:
        print(explanation)
    print(f"{'─'*60}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Music Recommender — batch simulation or AI agent mode"
    )
    parser.add_argument(
        "--agent",
        "-a",
        nargs="?",
        const="",          # --agent with no value → REPL
        metavar="QUERY",
        help="Run the AI agent. Optionally pass a query string; omit for interactive REPL.",
    )
    args = parser.parse_args()

    if args.agent is None:
        # No --agent flag — original batch mode
        run_batch()
    else:
        run_agent(args.agent or None)


if __name__ == "__main__":
    main()
