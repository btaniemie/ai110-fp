"""
Reliability evaluation for the MusicAgent.

Runs deterministic test cases against the recommender layer (no API calls needed)
and against the agent's tool-execution layer. Reports pass/fail counts and
average confidence scores.

Run with:  pytest tests/test_reliability.py -v
"""

import json
import pytest

from src.recommender import load_songs, recommend_songs, score_song
from src.agent import MusicAgent


# ── Fixtures ──────────────────────────────────────────────────────────────────

SONGS_PATH = "data/songs.csv"


@pytest.fixture(scope="module")
def songs():
    return load_songs(SONGS_PATH)


@pytest.fixture(scope="module")
def agent(songs):
    """
    Agent instance with songs pre-loaded.  We bypass __init__ to avoid
    requiring an API key in unit tests — we only test _execute_tool here.
    """
    a = MusicAgent.__new__(MusicAgent)
    a.songs = songs
    a.model = "claude-haiku-4-5-20251001"
    return a


# ── Reliability test cases ────────────────────────────────────────────────────
# Each case defines a user profile, the expected top genre, and minimum confidence.

RELIABILITY_CASES = [
    {
        "name": "High-energy pop user gets a pop song first",
        "prefs": {"genre": "pop", "mood": "happy", "energy": 0.85, "valence": 0.82},
        "expected_top_genre": "pop",
        "min_confidence": 0.85,
    },
    {
        "name": "Chill lofi user gets a lofi song first",
        "prefs": {"genre": "lofi", "mood": "chill", "energy": 0.38, "valence": 0.58},
        "expected_top_genre": "lofi",
        "min_confidence": 0.80,
    },
    {
        "name": "Intense rock user gets a rock song first",
        "prefs": {"genre": "rock", "mood": "intense", "energy": 0.92, "valence": 0.35},
        "expected_top_genre": "rock",
        "min_confidence": 0.80,
    },
    {
        "name": "Peaceful classical user gets a classical song first",
        "prefs": {"genre": "classical", "mood": "peaceful", "energy": 0.18, "valence": 0.68},
        "expected_top_genre": "classical",
        "min_confidence": 0.70,
    },
    {
        "name": "Hip-hop confident user gets a hip-hop song first",
        "prefs": {"genre": "hip-hop", "mood": "confident", "energy": 0.78, "valence": 0.70},
        "expected_top_genre": "hip-hop",
        "min_confidence": 0.70,
    },
    {
        "name": "Adversarial jazz+high-energy gets a jazz song first (genre dominance)",
        "prefs": {"genre": "jazz", "mood": "happy", "energy": 0.90, "valence": 0.75},
        "expected_top_genre": "jazz",
        "min_confidence": 0.50,
    },
]


@pytest.mark.parametrize("case", RELIABILITY_CASES, ids=[c["name"] for c in RELIABILITY_CASES])
def test_top_recommendation_matches_genre(songs, case):
    """Top recommendation should match the user's preferred genre."""
    recs = recommend_songs(case["prefs"], songs, k=5)
    assert recs, f"No recommendations returned for {case['name']}"

    top_song, top_score, _ = recs[0]
    assert top_song["genre"] == case["expected_top_genre"], (
        f"Expected genre '{case['expected_top_genre']}', "
        f"got '{top_song['genre']}' ({top_song['title']}, score={top_score:.2f})"
    )


@pytest.mark.parametrize("case", RELIABILITY_CASES, ids=[c["name"] for c in RELIABILITY_CASES])
def test_top_confidence_meets_threshold(songs, case):
    """Top recommendation's confidence score should meet the minimum threshold."""
    _MAX_SCORE = 5.5
    recs = recommend_songs(case["prefs"], songs, k=5)
    assert recs

    _, top_score, _ = recs[0]
    confidence = min(top_score / _MAX_SCORE, 1.0)
    assert confidence >= case["min_confidence"], (
        f"Confidence {confidence:.2f} below threshold {case['min_confidence']} "
        f"for '{case['name']}'"
    )


# ── Tool-layer tests (no API key needed) ──────────────────────────────────────

def test_execute_tool_returns_songs(agent):
    result = agent._execute_tool(
        "search_songs",
        {"genre": "pop", "mood": "happy", "energy": 0.85, "valence": 0.82, "k": 3},
    )
    assert "songs" in result
    assert len(result["songs"]) == 3


def test_execute_tool_confidence_range(agent):
    result = agent._execute_tool(
        "search_songs",
        {"genre": "lofi", "mood": "chill", "energy": 0.4, "k": 5},
    )
    for song in result["songs"]:
        assert 0.0 <= song["confidence"] <= 1.0, (
            f"Confidence {song['confidence']} out of range for {song['title']}"
        )


def test_execute_tool_unknown_raises(agent):
    with pytest.raises(ValueError, match="Unknown tool"):
        agent._execute_tool("nonexistent_tool", {})


def test_execute_tool_partial_input_uses_defaults(agent):
    """Energy and valence should default gracefully when omitted."""
    result = agent._execute_tool(
        "search_songs",
        {"genre": "ambient", "mood": "chill", "energy": 0.3},
    )
    assert result["songs"], "Expected at least one result with partial input"


# ── Score-level unit tests ────────────────────────────────────────────────────

def test_perfect_genre_and_mood_match_scores_highest(songs):
    """A song that matches genre + mood + energy exactly should have max score."""
    # Sunrise City: pop, happy, energy=0.82
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.82, "valence": 0.84}
    target = next(s for s in songs if s["title"] == "Sunrise City")
    score, reasons = score_song(prefs, target)
    # genre(2.0) + mood(1.0) + near-perfect energy + near-perfect valence ≈ 5.5
    assert score >= 5.0, f"Expected high score, got {score:.2f}"


def test_score_reasons_contain_all_components(songs):
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.7}
    song = songs[0]
    _, reasons = score_song(prefs, song)
    assert len(reasons) >= 2, "Expected at least energy and valence reason"


def test_recommend_returns_k_results(songs):
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    recs = recommend_songs(prefs, songs, k=3)
    assert len(recs) == 3


def test_recommend_sorted_descending(songs):
    prefs = {"genre": "lofi", "mood": "chill", "energy": 0.4}
    recs = recommend_songs(prefs, songs, k=5)
    scores = [score for _, score, _ in recs]
    assert scores == sorted(scores, reverse=True), "Results not sorted descending"


# ── Reliability summary (run as a standalone script) ─────────────────────────

def _run_reliability_report():
    """Print a human-readable reliability report — call via python -m pytest or directly."""
    songs_data = load_songs(SONGS_PATH)
    passed = 0
    total = len(RELIABILITY_CASES)
    confidences = []

    print("\n" + "=" * 60)
    print("  Reliability Evaluation Report")
    print("=" * 60)

    for case in RELIABILITY_CASES:
        recs = recommend_songs(case["prefs"], songs_data, k=5)
        if not recs:
            print(f"  FAIL  {case['name']} — no results")
            continue
        top_song, top_score, _ = recs[0]
        _MAX_SCORE = 5.5
        conf = min(top_score / _MAX_SCORE, 1.0)
        confidences.append(conf)
        ok = top_song["genre"] == case["expected_top_genre"] and conf >= case["min_confidence"]
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {case['name']}")
        print(f"        top={top_song['title']!r}  genre={top_song['genre']}  conf={conf:.2f}")

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    print("=" * 60)
    print(f"  Result: {passed}/{total} tests passed")
    print(f"  Average confidence: {avg_conf:.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    _run_reliability_report()
