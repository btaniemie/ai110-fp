"""Unit tests for the core recommender module (no API key required)."""

import pytest
from src.recommender import Song, UserProfile, Recommender, score_song, recommend_songs


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def make_songs_dict_list():
    return [
        {
            "id": 1, "title": "Pop Hit", "artist": "A", "genre": "pop",
            "mood": "happy", "energy": 0.8, "tempo_bpm": 120,
            "valence": 0.9, "danceability": 0.8, "acousticness": 0.2,
        },
        {
            "id": 2, "title": "Lofi Study", "artist": "B", "genre": "lofi",
            "mood": "chill", "energy": 0.4, "tempo_bpm": 80,
            "valence": 0.6, "danceability": 0.5, "acousticness": 0.9,
        },
        {
            "id": 3, "title": "Rock Storm", "artist": "C", "genre": "rock",
            "mood": "intense", "energy": 0.95, "tempo_bpm": 150,
            "valence": 0.4, "danceability": 0.6, "acousticness": 0.1,
        },
    ]


# ── Recommender class tests ───────────────────────────────────────────────────

def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_recommend_respects_k():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    assert len(rec.recommend(user, k=1)) == 1
    assert len(rec.recommend(user, k=2)) == 2


def test_recommend_k_larger_than_catalog_returns_all():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=100)
    assert len(results) == len(rec.songs)


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(user, rec.songs[0])
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_explain_recommendation_contains_genre_when_matched():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(user, rec.songs[0])
    assert "genre match" in explanation


def test_explain_recommendation_no_genre_mention_when_not_matched():
    user = UserProfile(
        favorite_genre="rock",
        favorite_mood="intense",
        target_energy=0.9,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    # songs[0] is pop — no genre match
    explanation = rec.explain_recommendation(user, rec.songs[0])
    assert "genre match" not in explanation


# ── score_song / recommend_songs tests ───────────────────────────────────────

def test_score_song_genre_match_adds_two_points():
    prefs = {"genre": "pop", "mood": "calm", "energy": 0.5, "valence": 0.5}
    song_match = {"genre": "pop", "mood": "happy", "energy": 0.5, "valence": 0.5}
    song_no_match = {"genre": "rock", "mood": "happy", "energy": 0.5, "valence": 0.5}
    score_match, _ = score_song(prefs, song_match)
    score_no_match, _ = score_song(prefs, song_no_match)
    assert score_match - score_no_match == pytest.approx(2.0)


def test_score_song_mood_match_adds_one_point():
    prefs = {"genre": "x", "mood": "happy", "energy": 0.5, "valence": 0.5}
    song_match = {"genre": "y", "mood": "happy", "energy": 0.5, "valence": 0.5}
    song_no_match = {"genre": "y", "mood": "chill", "energy": 0.5, "valence": 0.5}
    s_match, _ = score_song(prefs, song_match)
    s_no_match, _ = score_song(prefs, song_no_match)
    assert s_match - s_no_match == pytest.approx(1.0)


def test_score_song_perfect_energy_match_adds_1_5():
    prefs = {"genre": "x", "mood": "x", "energy": 0.7, "valence": 0.5}
    song = {"genre": "y", "mood": "y", "energy": 0.7, "valence": 0.5}
    score, reasons = score_song(prefs, song)
    # energy contribution should be exactly 1.5
    energy_reason = next(r for r in reasons if "energy" in r)
    assert "+1.50" in energy_reason


def test_score_song_returns_reasons_list():
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.7}
    song = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.7}
    score, reasons = score_song(prefs, song)
    assert isinstance(reasons, list)
    assert len(reasons) >= 2


def test_recommend_songs_sorted_descending():
    songs = make_songs_dict_list()
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.9}
    recs = recommend_songs(prefs, songs, k=3)
    scores = [score for _, score, _ in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommend_songs_top_is_genre_match():
    songs = make_songs_dict_list()
    prefs = {"genre": "lofi", "mood": "chill", "energy": 0.4, "valence": 0.6}
    recs = recommend_songs(prefs, songs, k=3)
    top_song, _, _ = recs[0]
    assert top_song["genre"] == "lofi"


def test_recommend_songs_k_1_returns_one():
    songs = make_songs_dict_list()
    prefs = {"genre": "rock", "mood": "intense", "energy": 0.95}
    recs = recommend_songs(prefs, songs, k=1)
    assert len(recs) == 1


def test_recommend_songs_explanation_non_empty():
    songs = make_songs_dict_list()
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    recs = recommend_songs(prefs, songs, k=1)
    _, _, explanation = recs[0]
    assert isinstance(explanation, str)
    assert explanation.strip() != ""
