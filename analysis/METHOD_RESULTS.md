# Alignment METHOD study — 2026-10-07

Research notes from the study that produced method M1 (lightly edited for publication). Numbers are from the original runs with the PyTorch checkpoint; the README tables are reruns of the public code with the released ONNX model and agree within about 0.3 ms.

Features are fixed throughout: learned v6 (`align_v6_final`) + fdtw log-mel ×2, with the interval gate (mel only for unison).
Only the DTW method changes. Tools: `runners/method_lib.py`, `runners/method_variants.py`, `analysis/method_run.py`, `analysis/method_diag.py`, `analysis/method_truth.py`, and `runners/m1_dtw.py` (a `rab_score.py` runner).
Suite D was not touched.

## 1. Diagnosis of suite R (baseline = v6+mel×2 gate, fdtw recursion: 43.5 / 20.2 / 49.6%)

- **Note changes dominate the error.**
  - Note changes: 3,764 of 5,365 events, mean 51.4 ms, median 25.2.
  - Onsets: 1,601 events, mean 24.8 ms, median 13.0.
  - Note changes account for about 83% of the total absolute error.
- **The tail dominates the mean.**
  - Events with error over 100 ms are 12% of events and carry 48% of the total error.
  - Events with error over 50 ms are 27% of events and carry 73% of the total error.
- **The aligner undershoots large offsets.**
  - The slope of predicted offset against truth offset is 0.25.
  - Where truth says the offset is 100–260 ms (15% of events), the aligner predicts only about 20–30 ms. Those events carry 46% of the error.
- **Time since onset.** Error is lowest right at onsets (mean 28). It is highest 50–300 ms after an onset (mean 76), where the scoops and glides into the first note change sit.

### Truth problems (evidence; the bench scoring is NOT changed)

1. **Invalid annotation: `CSD_ND_tenor_3.f0` is a copy of `CSD_ND_tenor_2.f0`.**
   - The two files are 100% identical on co-voiced frames.
   - The tenor_3 audio is a different take (sample correlation −0.005).
   - Against yin pitch from each audio file: the annotation matches tenor_2 at 0.03 st median (92% within 0.25 st), but tenor_3 at only 49%. Every other one of the 47 CSD f0 files matches its own audio at 90–98%.
   - So the truth for **R_ND_tenor_13 and R_ND_tenor_23 is invalid** (418 events, 7.8% of R).
   - tenor_23 shows "raw 0.6 ms", and every aligner looks worse than raw on it.
   - I recommend rebuilding R without these two cases, or with tenor_3 removed. "Rc" below = R without them.
2. **Note-change truth is ambiguous at the 20 ms level.**
   - The bench's note-change time is the frame of maximum 9-frame median step.
   - A second reasonable definition is the midpoint crossing (same f0, same estimator on both singers).
   - The two disagree by median 17 ms and mean 44 ms, and by more than 50 ms on 27% of note events. Onsets are consistent.
3. **Probable mismatches among large-offset events.**
   - Matching is greedy, same note, within ±250 ms.
   - For truth offsets of 150–260 ms, the independent estimate agrees within 20 ms only 26% of the time; its median offset is about 100 ms.
   - Truth offsets also jump event to event: the median deviation from neighbouring events within 1 s is 41 ms.
4. **Approximate noise floor.**
   - On the 58% of events where the two truth definitions agree within 20 ms (all onsets plus the clean note changes), the baseline already scores 24.0 / 12.6 ms / 66%.
   - On the other 42% it is worse than raw (70 against 68 ms).
   - Realistic floor with this truth: median about 8–12 ms and mean about 25–30 ms. No aligner can reach mean < ~25 on R as built.

## 2. Levers (one at a time from the baseline; R = all 48, Rev/Rodd = even/odd manifest halves, Rc = clean 46)

| lever | R mean/med/≤20 | Rev | Rodd | Rc | S mean / p90 / octave |
|---|---|---|---|---|---|
| raw (no alignment) | 52.0 / 34.8 / 33.7% | | | 54.7 / 34.8 / 30.4% | 39.5 / – / – |
| **baseline** (fdtw recursion, diag 2) | 43.4 / 20.2 / 49.6% | 40.7 / 17.8 / 53.4% | 45.8 / 23.2 / 46.3% | 41.7 / 18.4 / 52.6% | 3.08 / 5.3 / 3.97 |
| engine recursion (textbook min(D+w·c)), diag 2 = C++ today | 44.1 / 20.4 / 49.4% | 40.6 / 17.7 / 53.5% | 47.1 / 23.5 / 45.8% | 42.0 / 18.3 / 52.5% | 4.14 / 5.8 / 7.19 |
| textbook, diag **1.5** | 43.3 / 20.0 / 50.0% | 40.3 / 17.4 / 54.3% | 46.0 / 22.9 / 46.3% | 41.3 / 18.0 / 53.0% | **2.88 / 5.2 / 3.43** |
| textbook, diag 1 | 43.7 / 21.2 / 48.3% | | | | 2.78 / 5.0 / 3.07 |
| diag 3 | 44.2–46.1, worse | | | | 3.9–6.1 |
| + pitch distance (0.25, cap 2 st) | 41.5 / 17.7 / 53.7% | 38.4 / 15.8 / 57.8% | 44.1 / 20.0 / 50.0% | 39.4 / 15.9 / 56.7% | 3.03 / 5.5 / 3.64 |
| + pitch-slope distance (0.5, cap 0.3 st/frame, unison only) | 40.5 / 14.8 / 57.8% | 37.2 / 13.2 / 62.0% | 43.3 / 16.7 / 54.2% | 38.4 / 13.6 / 61.2% | 3.15 / 5.6 / 3.83 |
| **M1 = diag 1.5 + pitch 0.25 + slope 0.5 (both unison only)** | **39.8 / 14.8 / 58.6%** | **36.3 / 13.2 / 62.8%** | **42.8 / 16.7 / 55.0%** | **37.6 / 13.3 / 61.9%** | 3.15 / 5.6 / 3.89 |
| slope cost on non-unison too | R same | | | | 6.1 / 8.5 / 11.2 (yin octave errors) |
| path median smoothing 5–41 frames | 43.5–44.9, worse | | | | ≈ same |
| band 0.3 / 0.5 / 1.5 s | 43.0 / 43.0 / 43.5 (±0) | | | | same |
| mel weight 1 / 1.5 / 3 / 4 (2 is best) | 44.6 / 43.9 / 43.9 / 45.3 | | | | same |
| corridor re-DTW ±40–100 ms around the M1 path (pitch-slope heavier) | 40.7–41.5, worse | | | | 3.1–3.3 |
| silence = fixed cost | ±0 on R | | | | S +0.5 worse |
| mel / learned CMN, loudness-slope cost, median-filtered pitch, voicing-mismatch 1–2 | within ±0.4 of M1 | | | | ±0.2 |

- M1 checked with the official scorer, `rab_score.py method_M1 runners/m1_dtw.py`:
  - R 39.8 / 14.8 / 58.6%, p90 114.2.
  - S 3.2 / p90 5.6, with variants t 2.9, s 2.8, h 3.0, o 3.9; 0 S cases worse than raw.
  - The same runner with METHOD=base_py gives R 43.5 / 20.2 / 49.6% (the documented baseline).
- Split check: the lever choices were round values from small sweeps. They are best or tied-best on the even half, and they improve the odd half by the same amount (odd: median 23.2 → 16.7, ≤20 ms 46.3% → 55.0%).
- **What M1 changes:**
  - Note changes: mean 51.4 → 46.0, median 25.2 → 15.8.
  - Onsets: unchanged (24.8 → 25.1).
  - On the clean-truth subset: 24.0 / 12.6 / 66% → 19.7 / 9.9 / 75.7%. This subset is defined with f0, so there is some bias in favour of the pitch terms.
- **Cost:**
  - S unison t/s lose about 0.3 ms (2.6 → 2.9).
  - The S octave set is now 3.9 (baseline-py 3.97). The C++ engine's current recursion gives 7.19 there.

## 3. Port to the plug-in engine (summary)

M1 was ported to RysUpAlign's C++ engine (the engine source is not published). Three details mattered:

1. **Pitch track.** The engine's original real-time pitch tracker differed from `librosa.yin` by about 0.15–0.2 st median, and the slope term (Q) is very sensitive to that noise (R 46.8 ms with the old tracker). An exact port of `librosa.yin` (8 kHz, 64 ms frames centred on the frame time, trough threshold 0.1, 1600 Hz ceiling) fixed it.
2. **Phrase safety.** The engine keeps a per-phrase identity fallback. Scoring that fallback on the learned embeddings alone undid M1's pitch-driven corrections (R 42.0); scoring it on the same local cost the DTW minimised gives the same result as no fallback on RAB (R 39.9) while keeping the protection.
3. **Unison gate.** The interval gate must read the 1600 Hz-ceiling tracks; with an 800 Hz ceiling two octave cases were called unison.

| engine | R mean / median / ≤20 ms (p90) | S mean (p90) | S t / s / h / o |
|---|---|---|---|
| diag 1.5 only | 43.7 / 20.2 / 49.5% | 3.2 (5.4) | 2.61 / 2.57 / 3.04 / 4.59 |
| M1 port, final (as shipped in RysUpAlign 2.1) | **39.9 / 14.9 / 58.2% (113.9)** | **3.2 (5.6)**, 0 worse than raw | **2.88 / 2.85 / 3.04 / 3.92** |
| Python M1 (reference) | 39.8 / 14.8 / 58.6% (114.2) | 3.2 (5.6) | 2.9 / 2.8 / 3.0 / 3.9 |

- Three S octave cases (DavidTyo_21_o, Actions_14/15_o) are still read as unison by both trackers; they are equally bad in Python M1.
- Suite D was run once after this, see the README.
