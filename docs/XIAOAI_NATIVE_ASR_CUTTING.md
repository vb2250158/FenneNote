# XiaoAI Native ASR Cutting

This note documents the current XiaoAI input design for FenneNote.

## Goal

Use XiaoAI's native ASR for text recognition, while applying FenneNote's local trigger parameters to decide when a phrase should end.

The target behavior is:

1. XiaoAI microphone audio is monitored by the Open-XiaoAI runtime.
2. FenneNote provides the same thresholds used by its microphone recorder:
   - `recordThreshold`
   - `transcribeThreshold`
   - `transcribePauseSeconds`
   - `xiaoaiLevelGain`
3. Open-XiaoAI wakes XiaoAI native ASR when the gained level crosses `recordThreshold`.
4. Once the phrase has crossed `transcribeThreshold`, Open-XiaoAI measures silence.
5. When silence lasts longer than `transcribePauseSeconds`, Open-XiaoAI releases its local recognition window so the next utterance can wake immediately.
6. The final transcript still comes from XiaoAI `SpeechRecognizer.RecognizeResult`.

## Implemented Path

FenneNote exposes the current settings at:

```text
GET http://127.0.0.1:8793/api/fennenote/xiaoai/config
```

The XiaoAI bridge forwards that config through:

```text
GET http://127.0.0.1:8799/v1/xiaoai/config
```

The Open-XiaoAI MiGPT runtime reads this config and applies a small local pause-cut state machine:

```text
voice >= transcribeThreshold -> mark phrase ready
voice < transcribeThreshold for transcribePauseSeconds -> release local recognition window
```

The runtime intentionally does not call `finishRecognition()` by default. Real-world logs showed that the pnshelper event pair can cancel an in-flight XiaoAI final result and eat the next sentence during continuous speech.

`finishRecognition()` remains available as an experimental hard-cut helper. It uses the same Open-XiaoAI pnshelper event pair that upstream uses for cancel wake / microphone transitions:

```sh
ubus -t1 -S call pnshelper event_notify '{"src":3, "event":7}' 2>&1
ubus -t1 -S call pnshelper event_notify '{"src":3, "event":8}' 2>&1
```

The default path keeps the recognition provider as XiaoAI native ASR. It does not run FenneNote ASR or DashScope ASR for this external text mode.

The current behavior is a soft cut:

```text
FenneNote decides when the local phrase window is ready to end.
XiaoAI still owns the real capture stop and final ASR result.
Open-XiaoAI can wake the next phrase as soon as the level is back in range; it no longer waits for a falling/rising edge.
```

## Device Findings

The target speaker exposes these relevant ubus capabilities:

```text
mibrain.ai_service
mibrain.aivs_event_post
pnshelper.event_notify
```

The speaker also has:

```text
/usr/share/xiaomi/vad_config.json
```

with:

```json
{
  "--min-voice-length": 10,
  "--min-sil-length": 50
}
```

`/usr/share` is firmware-owned on this device, so changing that config persistently requires a firmware/bind-mount style approach. That is a separate hardening path, not the current default runtime path.

Open-XiaoAI monitors XiaoAI native instructions from:

```text
/tmp/mico_aivs_lab/instruction.log
```

and receives final text from `SpeechRecognizer.RecognizeResult`.

## Known Boundary

`SpeechRecognizer.StopCapture` is visible in the monitored instruction schema, but it is an instruction emitted by the XiaoAI/AIVS side to tell the client to stop audio capture. The current local integration does not have a proven, stable client-to-cloud `StopCapture` command.

Therefore the current runtime solution is best described as:

```text
FenneNote controls native ASR wake and requested phrase-end timing.
XiaoAI still owns the final ASR segmentation and transcript result.
```

If the soft cut is still not fast enough in real use, the next step is to test a firmware-level VAD override for `--min-sil-length` or identify a supported `mibrain.aivs_event_post` payload that cleanly maps to ending the active recognition round without cancelling the pending final text.

## Verification

Static checks:

```powershell
corepack pnpm exec tsc --noEmit --pretty false
```

State machine check:

```text
Given a phrase above transcribeThreshold,
when the level stays below transcribeThreshold for transcribePauseSeconds,
the controller emits exactly one cut request.
```

Runtime checks:

```text
GET http://127.0.0.1:8799/v1/xiaoai/config
```

should include:

```json
{
  "transcribeThreshold": 0.015,
  "transcribePauseSeconds": 2
}
```

During speech tests, Open-XiaoAI logs should show:

```text
达到 FenneNote 安静切句条件，释放本地识别窗口
收到小爱最终转写
```

For continuous speech, after `收到小爱最终转写`, the next phrase should be allowed to wake immediately when the gained XiaoAI level is above `recordThreshold`. There is no extra local cooldown or rising-edge requirement.
