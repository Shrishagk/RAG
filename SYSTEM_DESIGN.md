# L2 System Design: Private Persona RAG Sync

## Goal

Keep raw conversation data private on-device while syncing only the minimum artifacts needed for cross-device continuity: checkpoint summaries, persona facts with evidence references, drift metadata, and model/version metadata.

## Architecture

```text
User device
  Raw CSV/messages
  Local encrypted store
  Local RAG index + intent model
  Persona/drift extractor
        |
        | syncs summary artifacts only
        v
Cloud sync API
  Auth + device registry
  Versioned artifact store
  Conflict resolver
        |
        v
Second device
  Downloads summaries/persona
  Rebuilds local retrieval cache
```

## What Stays Local

- Raw messages and CSV rows.
- Full message chunks containing sensitive text.
- Offline intent classifier inference inputs.
- Device encryption keys.

## What Syncs

- Topic checkpoint summaries, message ranges, and keywords.
- 100-message checkpoint summaries.
- Persona JSON fields with short evidence snippets or evidence ids.
- Persona drift timeline: day, tone labels, trigger type, and trigger keywords.
- Index version, extraction version, and last processed message id.

## On-Device Storage

Use a local encrypted SQLite database or file store. Tables are split by artifact type: messages, chunks, topic checkpoints, persona facts, drift days, and sync metadata. Every row carries `source_message_start`, `source_message_end`, `created_at`, `updated_at`, and `extractor_version`.

## Sync Flow

1. Device processes new messages chronologically.
2. Device writes raw data locally.
3. Device uploads only compact artifacts and version metadata.
4. Cloud stores artifacts by user id, device id, artifact id, and logical clock.
5. Other devices download changed artifacts and rebuild their local TF-IDF cache.

## Conflict Resolution

For factual persona conflicts, keep both claims if they have different evidence windows. Rank the displayed answer by recency, emotional weight, and evidence count. If two claims conflict directly, mark the field as `conflicted` and show both evidence snippets instead of overwriting one.

For sync conflicts, use last-write-wins only for generated cache metadata. For persona facts and drift entries, merge by artifact id and message range. If two devices process overlapping ranges with different extractor versions, prefer the newer extractor version but keep the older artifact as audit history.

## Tradeoffs

This design reduces privacy risk because raw conversations do not leave the device. The tradeoff is that a new device cannot answer highly specific quote-level questions until it has local raw data or downloaded approved evidence snippets. Summary sync is enough for persona continuity and high-level RAG, while sensitive retrieval remains local.
