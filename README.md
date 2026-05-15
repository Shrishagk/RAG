# Conversation RAG + Persona Chatbot

This project builds a lightweight, local RAG system over chronological conversation data and extracts a structured persona for `User 1`.

It uses only Python standard library modules. No OpenAI API or hosted LLM is required.

## Features

- Parses the CSV in chronological row order.
- Splits the stream message by message.
- Creates topic checkpoints whenever the lexical topic changes.
- Creates independent 100-message checkpoints.
- Builds message chunks for retrieval.
- Builds a structured persona JSON from actual `User 1` evidence.
- Builds an adaptive persona drift timeline by day.
- Trains a lightweight offline intent classifier for five intent classes.
- Resolves hard family/person queries with recency, emotional weight, and contradiction flags.
- Serves a simple chatbot UI that answers using topic summaries, chunks, and persona data.

## Quick Start

```bash
cd conversation-rag-persona
python app.py --csv "C:\Users\shris\Downloads\conversations.csv" --build
python app.py --csv "C:\Users\shris\Downloads\conversations.csv" --serve
```

Open:

```text
http://localhost:8000
```

If you already built the index once, start only the server:

```bash
python app.py --serve
```

## Outputs

The build step writes:

- `data/messages.jsonl` - chronological parsed messages
- `data/topic_checkpoints.json` - topic segments with summaries
- `data/hundred_checkpoints.json` - summaries for every 100 messages
- `data/chunks.json` - retrievable chronological message chunks
- `data/persona.json` - structured persona with evidence
- `data/persona_drift.json` - day-wise mood/tone drift timeline
- `data/intent_model.json` - offline Naive Bayes intent classifier
- `data/index.json` - complete app index

## How Topic Changes Are Detected

The processor reads the data in chronological order and maintains a rolling keyword profile for the current topic segment. For each new message it:

1. Tokenizes the message into normalized content words.
2. Compares the message keywords to the current topic profile using cosine similarity.
3. Requires a minimum segment length before allowing a split, so short conversational turns do not create noisy topics.
4. Adds a checkpoint when similarity stays below the configured threshold and the segment has enough content.
5. Forces a checkpoint when the segment grows too long, keeping summaries bounded.

Each topic checkpoint stores:

- start and end message numbers
- start and end CSV row numbers
- detected top keywords
- extractive summary of that topic segment
- representative messages used as evidence

This avoids treating the whole dataset as one topic.

## How Retrieval Works

The system retrieves from two independent sources:

1. Topic summaries, which provide higher-level context.
2. Message chunks, which provide concrete nearby evidence.

Both are ranked with a local TF-IDF cosine scorer. The query is tokenized with the same pipeline used during indexing. At answer time, the chatbot combines:

- top topic summaries
- top message chunks
- persona fields when the question is about habits, traits, facts, or communication style

This makes the answer grounded in chronological checkpoints and actual messages.

## How Persona Is Built

Persona extraction only uses `User 1` messages. It looks for direct signals such as:

- habit statements: "I usually...", "I always...", "I like...", "I enjoy..."
- personal facts: "I am...", "I work as...", "I study...", "I live...", "my wife..."
- personality signals: enthusiasm, humor, emotional phrasing, curiosity, gratitude
- communication style: message length, punctuation, emoji use, question rate, tone markers

Each structured item includes evidence examples and counts. The extractor avoids unsupported guesses by leaving categories sparse when signals are not present.

## L2: Adaptive Persona Drift

The drift engine reads the same chronological stream and groups `User 1` messages by CSV row/day. For each day it computes:

- sentiment score from positive and negative emotional terms
- question rate and exclamation rate
- casual/formal markers
- emotional word density
- top keywords and possible trigger type

A drift is recorded when the tone label changes, sentiment moves meaningfully, or the main keywords change. Triggers are labeled as `person`, `event`, `topic`, or `tone` using direct text signals such as family mentions, work/school/event terms, and newly dominant keywords.

Output file:

```text
data/persona_drift.json
```

Example chatbot question:

```text
Show persona drift timeline
```

## L2: Offline Intent Classifier

The intent classifier is a compact Multinomial Naive Bayes model stored as JSON. It is trained locally from deterministic weak labels in the conversation data and does not call OpenAI, Gemini, or any external API.

Supported classes:

- `reminder`
- `emotional-support`
- `action-item`
- `small-talk`
- `unknown`

Inference is a small token-count calculation and is designed to run on CPU in under 200ms per message.

Example:

```bash
python app.py --ask "classify: remind me to call my sister tomorrow"
```

## L2: Conflict Resolution in RAG

For questions like:

```text
Did I mention anything about my sister?
```

the resolver retrieves relevant topic summaries and message chunks, then ranks chunks with:

```text
TF-IDF relevance + recency boost + emotional weight + exact term boost
```

It also scans the selected evidence for contradiction signals, such as positive/supportive claims and negative or negated claims around the same person/family term. If conflicts are found, the answer preserves uncertainty instead of collapsing everything into one unsupported fact.

## L2 Written Artifacts

- `SYSTEM_DESIGN.md` - one-page sync architecture with diagram and conflict policy
- `SELF_EVALUATION.md` - implementation checklist, limitations, and demo prompts

## CLI Examples

Ask from the terminal:

```bash
python app.py --ask "What kind of person is this user?"
python app.py --ask "What are their habits?"
python app.py --ask "How do they talk?"
python app.py --ask "Show persona drift timeline"
python app.py --ask "Did I mention anything about my sister?"
python app.py --ask "classify: remind me to call my sister tomorrow"
```

Tune topic splitting:

```bash
python app.py --csv conversations.csv --build --topic-threshold 0.08 --min-topic-messages 12
```

## Cloud Hosting

This is a single-process Python web app, so it can be hosted on Render, Railway, Fly.io, or a small VM.

Example start command:

```bash
python app.py --serve --host 0.0.0.0 --port $PORT
```

For deployment, upload the repo with the generated `data/index.json`, or run the build command during deploy with the CSV available in the project.

## Submission Checklist

Add these links before submitting:

- GitHub repo: `PASTE_GITHUB_REPO_LINK_HERE`
- Live chatbot URL: `PASTE_CLOUD_URL_HERE`
- Video demo: `PASTE_LOOM_LINK_HERE`

Recommended deployment flow:

1. Push this folder to GitHub.
2. Make sure `data/index.json` is included in the repo so the server can start without rebuilding the CSV in cloud.
3. Create a new Web Service on Render or Railway.
4. Connect the GitHub repo.
5. Use this start command:

```bash
python app.py --serve --host 0.0.0.0 --port $PORT
```

6. Open the generated public URL and test:

```text
What kind of person is this user?
What are their habits?
How do they talk?
```
