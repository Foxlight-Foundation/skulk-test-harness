# Speech fixtures

These WAV files are deterministic request inputs for speech qualification. They
are not Skulk voice presets or claims that a mounted TTS model supports the
fixture's language.

- `dashboard-stt-release.wav` is the English microphone fixture used by the
  dashboard and realtime transcription journeys.
- `french-translation-release.wav` is the fixed French upload used only to
  exercise `/v1/audio/translations`. It was captured from the successful
  retained output of the Foxlight speech battery, normalized to mono 16-bit
  PCM at 16 kHz, and is paired with `french-translation-release.txt`. The WAV
  has SHA-256
  `ac4dfe702825efd2c823c37407b364fd1d59260ae3b1558d54b477faa25be6f9`.
