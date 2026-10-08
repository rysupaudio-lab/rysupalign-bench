# Original research runs (2026-10-06 .. 2026-10-08)

Per-case results as written by `rab_score.py` at the time (`mean`, `median`, `raw_mean`, `n` per case).
Name prefix = case set: `R_` all 48 R cases, `RS_` R + all 120 S cases, `RS90_` R + the 90 S cases of
variants t/s/h (run before variant o existed), `S_` all 120 S cases, `Rsub_` the 16 R_ER cases,
`Ssub_` the 21 fury t/s/h cases (28 = including the 7 fury octave cases), `D_` the 12 held-out cases.
Feature names follow `runners/fdtw.py` (`<model>@<layer>`, `+mel`): hubert/wavlm/mert/mms = base
models, hubertL/wavlmL/mertL = large, xlsr = XLS-R 300M, mms1b = MMS-1B, synthgt = HubertFA SynthGT
posteriors. `v6` = learned v6 (+ log-mel, gated); `engine_*` = RysUpAlign C++ engine.
