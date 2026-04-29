# VibeFinder 2.0 — AI-Powered Music Recommender

## Original Project (Modules 1–3)

**VibeFinder 1.0** was a content-based music recommender built in Modules 1–3. It accepted a structured user profile (preferred genre, mood, energy level, and valence) and scored every song in an 18-track CSV catalog using a weighted formula — genre match (+2.0), mood match (+1.0), energy proximity (up to +1.5), and valence proximity (up to +1.0). It returned the top-5 matches with plain-language explanations. The system was fully deterministic, required no API calls, and made the scoring logic completely transparent.

---

## What's New in VibeFinder 2.0

VibeFinder 2.0 adds a natural language interface powered by the **Claude API**. Instead of filling out a structured profile, you describe what you want in plain English — *"I need something intense to work out to"* or *"give me chill background music for studying"* — and an AI agent parses your intent, retrieves matching songs, and explains its picks in a personalised response.

**Advanced AI features used:**

| Feature | How it's used |
|---|---|
| **Agentic Workflow** | Claude runs in a tool-use loop: it interprets the query, calls `search_songs`, receives ranked results, then synthesises a final response |
| **Retrieval-Augmented Generation (RAG)** | The agent retrieves relevant songs from `songs.csv` before generating any output — it never recommends from memory |

---

## Architecture

```
User query (natural language)
         │
         ▼
  ┌─────────────────────────────────┐
  │  MusicAgent  (src/agent.py)     │
  │  Claude claude-haiku-4-5        │
  │  - parses intent                │
  │  - decides tool parameters      │
  └──────────────┬──────────────────┘
                 │ tool_use: search_songs(genre, mood, energy, …)
                 ▼
  ┌─────────────────────────────────┐
  │  search_songs tool              │
  │  → recommend_songs()            │   ← existing scoring logic
  │  → songs.csv  (18 tracks)       │   ← RAG data source
  └──────────────┬──────────────────┘
                 │ tool_result: ranked songs + confidence scores
                 ▼
  ┌─────────────────────────────────┐
  │  MusicAgent synthesises         │
  │  personalised explanation       │
  └──────────────┬──────────────────┘
                 │
                 ▼
  Structured output:
    • explanation  (natural language)
    • recommendations[]  (title, artist, genre, mood, score, confidence, why)
    • confidence  (avg score normalised 0–1)
```

**Components:**

| Component | File | Role |
|---|---|---|
| Recommender core | `src/recommender.py` | Scoring, ranking, `Song` / `UserProfile` dataclasses |
| AI agent | `src/agent.py` | Claude agentic loop, tool dispatch, confidence scoring |
| CLI runner | `src/main.py` | Batch simulation + `--agent` flag |
| Song catalog | `data/songs.csv` | 18-track RAG data source |
| Unit tests | `tests/test_recommender.py` | 14 tests for core scoring logic |
| Reliability eval | `tests/test_reliability.py` | 20 tests covering 6 named profiles + tool layer |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd ai110-module3show-musicrecommendersimulation-starter

python -m venv .venv
source .venv/bin/activate      # Mac / Linux
.venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Anthropic API key (agent mode only)

```bash
cp .env.example .env
# Open .env and set ANTHROPIC_API_KEY=sk-ant-...
```

The batch simulation works without a key. Only `--agent` mode requires it.

---

## Running the App

### Batch simulation (original behaviour, no API key needed)

```bash
python -m src.main
```

Runs four hardcoded profiles and prints ranked recommendations with scores.

### Agent mode — single query

```bash
python -m src.main --agent "I want something chill to study to"
```

### Agent mode — interactive REPL

```bash
python -m src.main --agent
```

Type your request at the prompt; type `quit` to exit.

### Run tests

```bash
pytest                        # all 34 tests
pytest tests/test_reliability.py -v   # reliability report only
```

---

## Sample Interactions

### Interaction 1 — Chill study session

**Input:**
```
I need calm, relaxed music for studying. Something with low energy and a peaceful vibe.
```

**Output (truncated):**
```
Top 5 recommendations  (avg confidence: 82%)

  #1  Piano After Rain by Clara Voss
       Genre/Mood : classical / peaceful
       Score      : 4.44  (confidence: 81%)
       Why        : energy proximity (+1.41) | valence proximity (+0.92)

  #2  Library Rain by Paper Lanterns
       Genre/Mood : lofi / chill
       Score      : 4.09  (confidence: 74%)
       ...

Here are some wonderfully calm tracks to keep you focused. Piano After Rain 
leads with its gentle classical sound — nearly no energy, just pure ambience. 
Library Rain adds a soft lofi texture that many students find ideal for 
sustained concentration...
```

---

### Interaction 2 — Intense workout

**Input:**
```
I want something super high-energy and intense for the gym. Heavy, powerful music.
```

**Output (truncated):**
```
Top 5 recommendations  (avg confidence: 91%)

  #1  Shatter Zone by Iron Veil
       Genre/Mood : metal / intense
       Score      : 4.94  (confidence: 90%)
       Why        : mood match (+1.0) | energy proximity (+1.49) | ...

  #2  Bass Drop City by Frequency Lab
       Genre/Mood : electronic / energetic
       Score      : 4.91  (confidence: 89%)
       ...

These tracks are built for maximum output. Shatter Zone hits 0.97 energy — 
about as intense as it gets in our catalog. Bass Drop City pairs that 
intensity with an electronic pulse that's perfect for keeping your tempo up...
```

---

### Interaction 3 — Late-night moody drive

**Input:**
```
Something moody and atmospheric for a late night drive. Synthwave or electronic, 
medium energy.
```

**Output (truncated):**
```
Top 5 recommendations  (avg confidence: 79%)

  #1  Night Drive Loop by Neon Echo
       Genre/Mood : synthwave / moody
       Score      : 5.19  (confidence: 94%)
       Why        : genre match (+2.0) | mood match (+1.0) | energy proximity (+1.32)

Night Drive Loop is a near-perfect fit — the synthwave genre and moody atmosphere 
match exactly what you described, and the 0.75 energy level hits that sweet spot 
between intensity and chill that makes long drives feel cinematic...
```

---

## Design Decisions

**Why an agentic loop instead of a single prompt?**
Tool use lets Claude decide *what* preferences to extract from the query without hardcoding a translation layer. A query like "I'm feeling nostalgic and want something slow" gets mapped to `(genre=folk, mood=melancholic, energy=0.3)` by the model, not by regex. The loop also allows Claude to issue a follow-up tool call if the first set of parameters seems off — though in practice one call is sufficient for this catalog size.

**Why Claude Haiku?**
This task involves structured tool use over a small dataset. Haiku is fast, cheap, and more than capable for intent parsing + explanation generation. Opus or Sonnet would add latency and cost without meaningfully better results here.

**Why keep the original batch mode?**
The batch simulation is deterministic and needs no API key, making it easy to run, test, and demonstrate scoring logic without any external dependencies. It also provides a baseline for comparing agent output against ground truth.

**Confidence scoring**
Scores are normalised against the theoretical maximum of 5.5 (genre 2.0 + mood 1.0 + energy 1.5 + valence 1.0). A confidence of 0.90 means the song scored 90% of the best possible match — a meaningful signal that the recommendation is strong.

**Prompt caching**
The system prompt is marked with `cache_control: ephemeral`. On repeated queries in the same session, Claude reuses the cached prompt, reducing input token cost by ~80%.

---

## Testing Summary

```
34 tests collected
34 passed in 0.91s
```

**test_recommender.py (14 tests)** — unit tests for the core scoring layer:
- Score arithmetic: genre/mood/energy/valence deltas verified to exact decimal
- Ranking: results are always sorted descending by score
- Edge cases: `k > catalog size`, empty genre match, explanation format

**test_reliability.py (20 tests)** — reliability evaluation:

| Profile | Top genre expected | Pass? | Confidence |
|---|---|---|---|
| High-energy pop | pop | ✅ | 0.99 |
| Chill lofi | lofi | ✅ | 0.99 |
| Intense rock | rock | ✅ | 0.97 |
| Peaceful classical | classical | ✅ | 0.81 |
| Hip-hop confident | hip-hop | ✅ | 0.91 |
| Adversarial jazz + high energy | jazz | ✅ | 0.67 |

**Result: 6/6 profile tests passed. Average confidence: 0.89.**

The adversarial case passes (jazz song is still returned first due to genre dominance) but with the lowest confidence — exactly the failure mode documented in the model card. The test is designed to flag this transparently rather than hide it.

Tool-layer tests (no API key needed) verify that `_execute_tool` returns correct structure, that confidence values are always in [0, 1], that unknown tools raise `ValueError`, and that partial inputs use safe defaults.

**What didn't work:** Early versions of the system prompt did not force the agent to call `search_songs` before responding. Claude occasionally answered from training data ("Here are some chill songs: ...") instead of querying the catalog. Adding *"Always call search_songs before responding. Never recommend songs from memory."* to the system prompt fixed this reliably.

---

## Reflection and Ethics

### Limitations and biases

- **Genre dominance** remains the core bias: a 2.0-point genre bonus can override poor matches on energy and mood. A jazz fan who wants high-energy music receives the slowest jazz track in the catalog.
- **Small catalog (18 songs)**: some genres have only one representative, so recommendations for underrepresented genres are forced, not meaningful.
- **Fictional data**: all songs are invented. Real catalog data would introduce its own biases (recency, popularity, cultural representation).
- **LLM hallucination risk**: if the system prompt is weakened, Claude can generate song titles that do not exist in the catalog. The guardrail `"Never recommend songs from memory"` mitigates this but does not eliminate it completely.

### Could this be misused?

A music recommender is low-risk by nature, but the underlying pattern — a language model interpreting user intent and retrieving personalised content — applies to higher-stakes domains (health advice, financial guidance). In those contexts, the same architecture without stricter guardrails could surface harmful or misleading content. The key safeguard here is that Claude only picks from a fixed, human-curated catalog; it cannot generate new song recommendations outside that set.

### What surprised me

The biggest surprise was how brittle the system prompt needed to be. Without an explicit instruction not to use training memory, Claude confidently recommended real songs (Billie Eilish, Kendrick Lamar) instead of querying the catalog — and did so with no indication it was bypassing the tool. This is a meaningful reliability lesson: LLMs will take the path of least resistance unless you explicitly close off alternatives.

### Collaboration with AI

**Helpful suggestion**: When designing the tool schema, Claude suggested adding a `valence` field as an *optional* parameter (not required). This turned out to be the right call — many natural language queries ("chill study music") don't imply a specific positivity level, and requiring it would have forced Claude to hallucinate a default rather than leaving it unspecified.

**Flawed suggestion**: Claude initially suggested using `claude-opus-4-7` as the default model in `MusicAgent`. For a task this simple — structured tool use over an 18-track catalog — Opus adds significant latency and cost with no measurable quality improvement. Haiku handles intent parsing and explanation generation at this scale without issue. The suggestion prioritised capability over appropriateness, which is a common LLM pattern worth watching for.
