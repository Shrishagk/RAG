# Self-Evaluation Sheet

## Part 1: Adaptive Persona Engine

- Status: Implemented.
- Output: `data/persona_drift.json`.
- Method: Groups `User 1` messages by CSV row/day, computes tone features, detects day-to-day tone and sentiment changes, and stores trigger type as topic, event, person, or tone.
- Limitation: The detector is heuristic and explainable, not a deep emotion model.

## Part 2: Offline Intent Classifier

- Status: Implemented.
- Output: `data/intent_model.json`.
- Model: Multinomial Naive Bayes with a capped vocabulary.
- Labels: `reminder`, `emotional-support`, `action-item`, `small-talk`, `unknown`.
- Runtime: Fully offline Python standard library inference; designed for CPU sub-200ms per message.
- Limitation: Training labels are weak labels generated from deterministic patterns, so edge cases may need manually labeled examples for production quality.

## Part 3: Conflict Resolution in RAG

- Status: Implemented.
- Demo query: `Did I mention anything about my sister?`
- Method: Retrieves relevant checkpoints and chunks, ranks chunks by TF-IDF relevance, recency, emotional weight, and exact term boost, then flags contradictory positive/negative or negated claims around the same query term.
- Limitation: Contradiction detection is phrase-level and conservative.

## Part 4: System Design Doc

- Status: Implemented.
- File: `SYSTEM_DESIGN.md`.
- Covers: on-device storage, what syncs, what stays local, architecture diagram, and conflict policy.

## End-to-End Demo Checklist

- Build index: `python app.py --build`
- Start app: `python app.py --serve --host 0.0.0.0 --port 8000`
- Test persona drift: ask `Show persona drift timeline`
- Test intent: ask `classify: remind me to call my sister tomorrow`
- Test conflict RAG: ask `Did I mention anything about my sister?`
