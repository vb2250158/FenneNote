# QQ mixed virtual microphone routing

This page documents the safe routing pattern for using FenneNote during a QQ voice call when QQ friends should hear both:

- the user's physical microphone
- YeYu / Codex TTS playback

FenneNote does not install or configure virtual audio drivers. Install and configure tools such as VoiceMeeter or VB-CABLE manually, then use FenneNote's input device selector to keep transcription isolated from the mixed QQ microphone.

## Core rule

```text
FenneNote input = physical microphone only
QQ microphone = mixed virtual microphone
```

Do not select `VoiceMeeter Output`, `CABLE Output`, or another mixed virtual microphone as the FenneNote microphone. If FenneNote listens to the mixed device, YeYu / Codex TTS can be transcribed again and routed back into Codex.

## Recommended topology

```text
Physical microphone
  -> FenneNote input
  -> VoiceMeeter Hardware Input 1

YeYu / Codex TTS output
  -> VoiceMeeter virtual input
  -> local speakers/headphones for monitoring

VoiceMeeter B1 output
  -> QQ microphone
```

In this mode:

- FenneNote hears the real user microphone.
- QQ hears the VoiceMeeter B1 mixed output.
- The user hears TTS locally through speakers or headphones.

## VoiceMeeter preset

1. Set `Hardware Input 1` to the physical microphone.
2. Route the physical microphone to `B1`.
3. Route YeYu / Codex TTS playback into a VoiceMeeter virtual input.
4. Route the TTS input to `B1` so QQ can hear it.
5. Route the TTS input to `A1` if the user needs local monitoring.
6. In QQ, select `VoiceMeeter Output` or the B1 virtual output as the microphone.
7. In FenneNote, select the physical microphone directly.

## VB-CABLE plus VoiceMeeter

Use this when the TTS worker cannot directly choose a VoiceMeeter playback device:

```text
YeYu / Codex TTS output -> CABLE Input
CABLE Output -> VoiceMeeter input
Physical microphone -> VoiceMeeter Hardware Input
VoiceMeeter B1 -> QQ microphone
Physical microphone -> FenneNote input
```

VB-CABLE by itself is usually not enough for the QQ mixed mode because QQ needs one mixed microphone containing both the real microphone and TTS playback.

## FenneNote presets

FenneNote exposes an `audio_route_preset` setting:

```json
{
  "audio_route_preset": "solo_voice_input"
}
```

Supported values:

- `solo_voice_input`: normal mode. FenneNote listens to the physical microphone or system default microphone.
- `qq_mixed_output_mode`: documentation preset for QQ calls. FenneNote still listens to the physical microphone. QQ should use the mixed virtual microphone.

The preset does not change Windows, QQ, VoiceMeeter, or VB-CABLE settings. It only records the intended routing mode and shows the correct setup reminder in the GUI.

## Optional TTS guard

Device isolation is the main anti-loop protection. FenneNote also has an optional local guard that can be enabled as a backup:

```json
{
  "tts_guard_enabled": true,
  "tts_guard_file": "cache/tts_guard.json",
  "tts_guard_resume_margin_seconds": 0.8,
  "tts_guard_recent_text_window_seconds": 20.0,
  "tts_guard_similarity_threshold": 0.86
}
```

The TTS side can write `cache/tts_guard.json` before or during playback:

```json
{
  "ignore_until": 1780500000.8,
  "text": "This is the TTS text that was just spoken.",
  "recent_texts": [
    {
      "time": 1780500000,
      "text": "This is the TTS text that was just spoken."
    }
  ]
}
```

When enabled, FenneNote skips microphone phrases while `ignore_until` is in the future. It also drops transcripts that are highly similar to recent TTS text. Keep this disabled until a TTS worker is ready to write the guard file.

## Verification

1. In FenneNote, select the physical microphone. Confirm the level meter moves when the user speaks.
2. In QQ, select the VoiceMeeter or VB-CABLE mixed virtual microphone.
3. Speak normally. FenneNote should transcribe the user, and QQ friends should hear the user.
4. Play YeYu / Codex TTS. QQ friends should hear TTS.
5. Watch FenneNote during TTS playback. It should not transcribe the TTS unless the physical microphone picks up room audio from speakers.
6. If TTS still appears in FenneNote, use headphones, lower speaker volume, rotate the microphone rejection side toward speakers, or enable the optional TTS guard after the TTS side can write `cache/tts_guard.json`.

## Troubleshooting

- FenneNote transcribes TTS immediately: FenneNote is probably listening to the mixed virtual microphone, or speakers are being picked up by the physical microphone.
- QQ friends hear only the user: TTS is not routed into the mixer or not routed to `B1`.
- QQ friends hear only TTS: the physical microphone is not routed to the virtual mixer output.
- FenneNote is silent: the selected physical microphone index may have changed after Windows audio device changes. Reopen FenneNote and select the microphone again.
