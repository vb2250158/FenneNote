# Voice identity and playback integration

FenneNote and OumuQ should remain useful as independent applications.

- FenneNote is the local listening, transcription, audio identity, and anti-feedback center.
- OumuQ is the character voice generation and playback center.
- RabiRoute receives structured events and should not own low-level audio capture, playback, or speaker verification.

The integration rule is: each app works alone, and optional HTTP/JSON links add richer behavior when both are running.

## Current baseline

FenneNote currently supports:

- physical microphone transcription
- optional microphone segment saving under `cache/audio/`
- RabiRoute `voice_transcript` webhook output
- optional TTS guard reading from `cache/tts_guard.json`
- transcript dropping when text is highly similar to recent TTS text
- FenneNote-initiated playback requests through OumuQ
- a local speaker registry with sample metadata
- lightweight local speaker embedding / voiceprint matching
- optional DashScope non-real-time diarization for speaker subtitles
- speaker metadata attached to transcript previews and RabiRoute events

OumuQ already supports:

- character registry reading from `voice-references/reference-index.json`
- `POST /speak` compatible worker routing
- generated audio output files in worker output directories
- character IDs such as `tamamo_no_mae`

The remaining parts are:

- calibrated thresholds from real user and TTS samples
- stronger production speaker embedding models if the lightweight local embedding is not accurate enough
- hard anti-feedback decisions for TTS speakers after enough local validation

## Phase 1: playback and guard bridge

FenneNote adds an optional voice output panel that can call an OumuQ-compatible endpoint:

```text
POST /api/speak or POST /speak
```

Example request:

```json
{
  "text": "晚上好，今天辛苦了。",
  "play": true,
  "character_id": "tamamo_no_mae"
}
```

Before sending the playback request, FenneNote writes its local TTS guard file:

```json
{
  "active": true,
  "ignore_until": 1780800008.0,
  "text": "晚上好，今天辛苦了。",
  "tts_text": "晚上好，今天辛苦了。",
  "recent_texts": [
    {
      "time": 1780800000.0,
      "text": "晚上好，今天辛苦了。",
      "speaker_id": "tts:tamamo_no_mae",
      "source": "fennenote_oumuq"
    }
  ],
  "source": "fennenote_oumuq",
  "character_id": "tamamo_no_mae"
}
```

This makes the existing FenneNote guard immediately useful even before voiceprint recognition is implemented.

When playback requests come from RabiRoute, FenneNote treats the incoming JSON packet as the source of truth. Fields such as `model`, `character_id`, `language`, emotion controls, and `worker_url` are forwarded to OumuQ unchanged. FenneNote may inspect those fields to maintain a local active playback target and do low-latency preparation, but it should not rewrite the packet unless the user explicitly configures an override.

Low-latency model switching belongs in FenneNote/OumuQ, not in Codex or RabiRoute. FenneNote compares the incoming target with the current active target, updates local state, probes the requested worker when `worker_url` is present, and later can call an OumuQ worker-management/preload API when such an API exists.

OumuQ should accept additional playback fields and pass them through to the worker payload after removing only OumuQ-local routing fields such as `worker_url` and batch-only `lines`. This keeps future Codex fields, such as new emotion controls or packet IDs, usable without changing RabiRoute.

## Phase 2: speaker registry

FenneNote owns a local speaker registry, separate from OumuQ's voice reference registry.

Use the user-facing term "speaker profile" or "voiceprint profile" instead of "voiceprint file". A profile contains sample audio, metadata, and eventually an embedding file.

```json
{
  "version": 1,
  "speakers": [
    {
      "id": "user_main",
      "display_name": "User",
      "kind": "human",
      "samples": [
        "cache/speakers/user_main/user-001.wav"
      ],
      "embedding_file": "cache/speakers/user_main/embedding.npy"
    },
    {
      "id": "tts_tamamo_no_mae",
      "display_name": "Tamamo-no-Mae TTS",
      "kind": "tts",
      "character_id": "tamamo_no_mae",
      "samples": [
        "cache/speakers/tts_tamamo_no_mae/generated-001.wav"
      ],
      "embedding_file": "cache/speakers/tts_tamamo_no_mae/embedding.npy"
    }
  ]
}
```

OumuQ `character_id` can be mapped to a FenneNote `speaker_id`, but the systems do not have to share the same ID space.

## Progressive enrollment

Voiceprint profiles should be generated progressively:

```text
new audio segment
-> compare with existing speaker embeddings
-> matched above threshold: attach existing speaker_id
-> not matched: create a new pending speaker profile
-> user names the profile in the FenneNote GUI
-> future audio can match the named profile
```

Pending profile example:

```json
{
  "id": "unknown_20260606_183012",
  "display_name": "未命名声音",
  "kind": "unknown",
  "status": "pending_name",
  "samples": [
    "cache/audio/20260606-183012-123_dur2.4s_peak0.041.wav"
  ],
  "embedding_file": "cache/speakers/unknown_20260606_183012/embedding.npy",
  "created_by": "auto_enroll"
}
```

When OumuQ finishes generating playback audio, FenneNote can also register that output as a TTS speaker sample:

```text
OumuQ response output WAV
-> speaker_id = tts_<character_id>
-> add output file to samples
-> mark as TTS profile
```

This makes the workflow mostly automatic. The user should only need to rename profiles such as:

- `unknown_20260606_183012` -> `User`
- `tts_tamamo_no_mae` -> `YeYu TTS`
- `unknown_20260606_184500` -> `Friend A`

Suggested GUI controls:

- speaker list: ID, display name, kind, sample count, threshold, last updated
- add sample from microphone
- import sample from file
- import generated audio from an OumuQ output directory
- map OumuQ character ID to speaker ID
- rebuild embeddings
- test-identify a selected WAV or current microphone phrase

## Phase 3: voiceprint matching

FenneNote should expose an internal function shaped like:

```text
identify_speaker(audio) -> speaker_result
```

Suggested result:

```json
{
  "speaker_id": "user_main",
  "speaker_name": "User",
  "speaker_kind": "human",
  "confidence": 0.82,
  "decision": "accept"
}
```

Recommended model candidates:

- SpeechBrain ECAPA-TDNN
- pyannote speaker embedding
- WeSpeaker
- NVIDIA NeMo speaker embedding

Start with conservative thresholds and logging-only mode. After collecting enough local samples, enable hard decisions:

- `human/user` -> transcribe and route normally
- `tts` -> drop as feedback or record silently
- `other_human` -> transcribe with metadata, but let downstream intent rules decide whether to answer
- `unknown` -> keep existing guard and text-similarity fallback

Voiceprint matching and speaker diarization are different layers:

- Voiceprint matching answers: "Does this audio segment sound like speaker A?"
- Speaker diarization answers: "How many speakers are in this recording, and when did each one speak?"

If one microphone phrase contains two voices, a single whole-phrase embedding is not enough. FenneNote should first split the phrase into speaker turns, then run voiceprint matching on each turn:

```text
raw phrase
-> VAD / speech regions
-> speaker diarization turns
-> per-turn speaker embedding
-> speaker_id per turn
-> transcript with speaker metadata
```

The first implementation can start with whole-phrase matching for anti-feedback. Multi-speaker phrases should be treated as a later diarization phase.

## RabiRoute event metadata

When FenneNote routes a transcript, it should include speaker metadata:

```json
{
  "type": "voice_transcript",
  "text": "继续干活。",
  "speaker_id": "user_main",
  "speaker_name": "User",
  "speaker_kind": "human",
  "speaker_confidence": 0.82,
  "speaker_decision": "accept"
}
```

RabiRoute can use these fields in templates, but it should not perform voiceprint recognition itself.

## Implementation order

1. Add FenneNote playback settings and OumuQ test playback.
2. Write `tts_guard.json` from FenneNote before playback.
3. Add speaker registry data structures and GUI management.
4. Add logging-only speaker embedding matching.
5. Attach speaker metadata to RabiRoute events.
6. Enable hard anti-feedback decisions only after local threshold calibration.
