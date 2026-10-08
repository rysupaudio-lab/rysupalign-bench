# RAB: RysUp Align Bench

RAB is a benchmark for **audio-to-audio vocal alignment**: given a guide vocal (usually the lead) and
a double or backing vocal of the same line, move the double's timing onto the guide. No lyrics, text
or MIDI are given. This repository contains:

- the benchmark: build scripts for three suites with measured or exact timing truth, the scorer, and
  the runner contract for scoring any aligner;
- reference aligners: plain feature DTW on log-mel and on public self-supervised models
  (HuBERT, WavLM, MERT, MMS, XLS-R), and our learned features with the DTW method M1;
- the training and export code of the **RysUpAlign learned alignment model v6** (weights on
  [Hugging Face](https://huggingface.co/rysupaudio/rysupalign-learned-v6) and in the
  [v1.0 release](https://github.com/rysupaudio-lab/rysupalign-bench/releases/tag/v1.0));
- all results, per case, and the script that turns them into the tables below.

The learned model and method M1 are used in the RysUpAlign plug-in by Rys Up Audio
(https://rysupaudio.com).

## The task and why a new benchmark

Double tracks, gang vocals and backing stacks are aligned to a guide by ear or with alignment tools.
The tool sees two recordings of the same words sung by the same or a different singer, possibly at a
different pitch (harmony or octave), with different timbre, timing slop of tens to hundreds of
milliseconds, and bleed or noise. The quantity that matters is time: where in the double is the
material that belongs at each moment of the guide.

Public alignment benchmarks measure something else. MIREX lyrics-to-audio alignment and JamendoLyrics
evaluate **text-to-audio** alignment (word onsets against lyrics). Score-following and music-sync
datasets align recordings to symbolic scores or align whole performances of instrumental music.
We found no public set that measures audio-to-audio alignment of sung doubles with timing truth at
the 10 ms level, so we built one from public multitrack choir recordings, pop multitracks and our own
gang takes.

## Suites

| suite | cases | truth points | what it is | truth |
|---|---|---|---|---|
| **R** (real) | 48 | 5,365 events | Pairs of different singers of the same section singing in unison, from the Choral Singing Dataset (Cuesta et al. 2018). 3 pieces × 4 sections × 4 pairs. | Each singer's human-corrected f0 (5.8 ms hop). Events = voicing onsets (≥60 ms of silence before, ≥60 ms voiced) and note changes (≥0.8 st step), matched singer-to-singer in order: same kind, same note (±1 st), within ±250 ms. Raw singer offset: median 34.8 ms. |
| **S** (synthetic, exact) | 120 | 126,525 frames | 30 real takes: 23 backing-vocal stems from 5 Cambridge-MT sessions and 7 gang takes from the song "fury" (included in `data/fury`). Each take gives 4 cases: **t** typical slop (45 ms spread), **s** sloppy (90 ms), **h** harmony (guide shifted +4 st), **o** octave (guide +12 st, the double an octave below it). | The guide is the take itself, re-voiced (tilt EQ, noise at -45 dB; h/o also pitch-shifted with formants kept). The double is the same take cut at silences and syllable dips, with every segment slid by its own offset (phrase offset plus per-syllable jitter, 8 ms crossfades, no time-stretching). The truth is sample-exact on every voiced 10 ms frame. |
| **D** (held out) | 12 | 470 events | Pairs of different bass singers in section takes of Dagstuhl ChoirSet (Rosenzweig et al. 2020), headset or dynamic mics with some neighbour bleed. | CREPE f0 of each singer's larynx microphone (no bleed), same event rules as R. The CREPE annotation is automatic and on a 10 ms grid. |

Suite D was built after all development and **evaluated once**, for the final systems only; it was not
used for training, tuning or model selection. Please keep it that way if you report on it.

### Metric

For every truth point (guide time g, true double time d) the aligner's warp gives a predicted double
time d̂(g); the error is |d̂(g) − d| in ms. Reported per suite over all points pooled (so long cases
weigh more): mean, median, p90, the share of points within 20 ms, and the number of cases where the
aligner is more than 2 ms worse (mean) than doing nothing. "raw" is the identity warp (no alignment).

## Results

All rows use the same dense band DTW (±1 s band, 10 ms grid) and the same read-out unless noted;
only the features (and, for M1, the DTW cost) change. "worse than raw" counts cases whose mean error
grew by more than 2 ms.

- Rows marked **rerun** were produced with the code in this repository on cases rebuilt from scratch
  with the build scripts (every case's truth matched the published checksum), and pooled statistics are
  exact.
- The other rows are the original research runs (2026-10-06 to 2026-10-08). Their per-case results are
  in `results/original`; the pooled mean is computed exactly from them, the median / ≤20 ms / p90 are
  the scorer's printed output at the time ("–" where it was not kept). The weights of MERT, WavLM, MMS,
  XLS-R and the large models were deleted after the study, so these rows were not rerun.
- `python results/summarize.py` prints every table below from the files in `results/`.

**Systems.** *log-mel DTW*: 80-band log-mel, frame-centred. *<model> L<n>*: hidden layer n of the
model, mean-removed per take, at its native ~50 fps resampled to the 10 ms grid. *+ log-mel*: both
blocks unit-normalised and concatenated. *gated*: the log-mel block is used only when the double is
in unison with the guide (median yin f0 interval ≤ 0.5 st after a first pass without it), because
log-mel compares harmonic patterns and fails on harmony and octave doubles. *learned v6*: our model
(below); "+ log-mel" there means log-mel at weight 2 with the gate. *M1*: the DTW method in
`analysis/METHOD_RESULTS.md` (textbook recursion with diagonal weight 1.5, plus pitch-distance and
pitch-slope terms in the local cost for unison pairs). *RysUpAlign 2.1 engine*: the C++ implementation
in the plug-in (int8 model, plug-in defaults, phrase-level fallback); its code is not published, it was
scored through the same runner contract.

### Suite R: real unison pairs

Rc = R without R_ND_tenor_13 and R_ND_tenor_23, whose truth is invalid (see "Known problems").

| system | R mean | R median | R ≤20 ms | R p90 | R worse than raw | Rc mean | Rc median | Rc ≤20 ms |
|---|---|---|---|---|---|---|---|---|
| raw (no alignment) | 52.0 | 34.8 | 33.7% | 127.7 | – | 54.7 | 34.8 | 30.4% |
| log-mel DTW | 54.9 | 21.6 | 48.0% | 138.6 | 15/48 | 53.3 | 19.2 | 50.9% |
| HuBERT-base L6 | 52.0 | 26.8 | 41.8% | 129.0 | 12/48 | 50.5 | 24.2 | 44.3% |
| HuBERT-base L6 + log-mel | 47.4 | 24.1 | 45.0% | 120.3 | 6/48 | 45.6 | 21.6 | 47.7% |
| HuBERT-base L6 + log-mel (gated) | 47.4 | 24.1 | 45.0% | 120.3 | 6/48 | 45.6 | 21.6 | 47.7% |
| WavLM-base+ L6 | 54.2 | – | – | – | 18/48 | 52.5 | – | – |
| MERT-v1-95M L4 | 46.1 | 20.6 | 49.4% | 119.0 | 6/48 | 44.0 | – | – |
| MERT-v1-95M L4 + log-mel | 44.8 | 20.0 | 49.9% | 116.0 | 5/48 | 42.8 | – | – |
| learned v6 alone | 46.0 | 23.5 | 45.6% | 116.8 | 4/48 | 44.2 | 21.1 | 48.5% |
| learned v6 + log-mel (gated) | 43.8 | 20.3 | 49.5% | 115.0 | 3/48 | 42.0 | 18.6 | 52.4% |
| learned v6 + log-mel + M1 DTW | 39.8 | 14.8 | 58.4% | 113.9 | 2/48 | 37.6 | 13.4 | 61.7% |
| learned v6 + log-mel + M1 DTW, int8 model | 39.7 | 14.8 | 58.5% | 113.9 | 2/48 | 37.5 | 13.5 | 61.8% |
| RysUpAlign 2.1 engine (C++, int8) | 39.9 | 14.9 | 58.2% | 113.9 | 2/48 | 37.7 | – | – |
| RysUpAlign 2.0.4 engine (previous release) | 53.0 | 34.8 | 32.0% | – | 6/48 | 53.0 | – | – |

Rows: *rerun* = log-mel, HuBERT-base (all), learned v6 (all). Original runs = WavLM-base+, MERT,
RysUpAlign engines. On R every pair is unison, so the gated and ungated HuBERT rows are identical.

### Suite S: exact-truth cut-and-slide cases

| system | S mean | S median | S p90 | worse than raw | t | s | h (+4 st) | o (+12 st) |
|---|---|---|---|---|---|---|---|---|
| raw (no alignment) | 39.5 | 30.6 | 81.4 | – | 30.8 | 63.1 | 30.4 | 33.8 |
| log-mel DTW | 62.7 | 3.1 | 187.3 | 41/120 | 2.6 | 2.6 | 203.1 | 41.9 |
| HuBERT-base L6 | 7.8 | 3.1 | 8.2 | 7/120 | 3.5 | 3.7 | 12.3 | 11.8 |
| HuBERT-base L6 + log-mel | 6.1 | 2.7 | 5.8 | 4/120 | 2.7 | 2.6 | 13.4 | 5.8 |
| HuBERT-base L6 + log-mel (gated) | 7.4 | 3.0 | 7.1 | 7/120 | 2.7 | 2.6 | 12.3 | 11.9 |
| learned v6 alone | 3.4 | 2.6 | 5.4 | 1/120 | 2.8 | 2.8 | 3.3 | 4.8 |
| learned v6 + log-mel (gated) | 3.9 | 2.6 | 5.1 | 2/120 | 2.6 | 2.6 | 3.3 | 7.3 |
| learned v6 + log-mel + M1 DTW | 3.2 | 2.6 | 5.5 | 0/120 | 2.9 | 2.8 | 3.0 | 3.9 |
| learned v6 + log-mel + M1 DTW, int8 model | 3.1 | 2.6 | 5.5 | 0/120 | 2.9 | 2.8 | 3.0 | 3.7 |
| RysUpAlign 2.1 engine (C++, int8) | 3.2 | – | 5.6 | 0/120 | 2.9 | 2.8 | 3.0 | 3.9 |

### Feature sweep on subsets

Before the full runs, every public model and several layers were compared on two subsets: R_ER (the 16
R cases of the first piece) and the 21 fury cases of S (variants t, s, h). Means are event-weighted and
computed from the per-case results. The best layer of each base model on R_ER was then used in the
full tables, so R_ER is not an independent test for those rows. Whisper-large-v3 encoder features were
also tried; the runner produced identity warps, so they are not reported.

| features (dense DTW, no mel unless stated) | R_ER mean (16 cases) | worse than raw | S fury t/s/h mean (21 cases) | worse than raw |
|---|---|---|---|---|
| raw (no alignment) | 51.7 | – | 45.6 | – |
| log-mel DTW | 45.4 | 4/16 | 47.6 | 7/21 |
| HuBERT-base L6 | 45.3 | 1/16 | 3.2 | 0/21 |
| HuBERT-base L6 + log-mel | 42.1 | 0/16 | 2.9 | 0/21 |
| HuBERT-base L9 | 47.0 | 3/16 | 3.5 | 0/21 |
| HuBERT-base L12 | 48.1 | 4/16 | 4.0 | 0/21 |
| WavLM-base+ L6 | 46.1 | 3/16 | 3.2 | 0/21 |
| WavLM-base+ L9 | 45.0 | 2/16 | 3.7 | 0/21 |
| WavLM-base+ L12 | 45.5 | 3/16 | 4.2 | 0/21 |
| MERT-v1-95M L4 | 39.9 | 0/16 | 3.1 | 0/21 |
| MERT-v1-95M L4 + log-mel | 39.0 | 0/16 | – | – |
| MERT-v1-95M L7 | 40.0 | 0/16 | 2.9 | 0/21 |
| MERT-v1-95M L10 | 42.3 | 1/16 | 2.9 | 0/21 |
| MMS-300m aligner L12 | 45.0 | 4/16 | 3.7 | 0/21 |
| MMS-300m aligner L18 | 46.6 | 4/16 | 5.0 | 0/21 |
| MMS-300m aligner L24 | 48.6 | 4/16 | 8.7 | 0/21 |
| MERT-v1-330M L6 | 40.9 | 1/16 | 3.1 | 0/21 |
| MERT-v1-330M L10 | 40.1 | 1/16 | 3.0 | 0/21 |
| MERT-v1-330M L14 | 40.9 | 1/16 | 3.0 | 0/21 |
| WavLM-large L8 | 44.4 | 2/16 | 3.3 | 0/21 |
| WavLM-large L12 | 44.6 | 2/16 | 4.6 | 0/21 |
| WavLM-large L16 | 44.7 | 2/16 | 6.1 | 0/21 |
| HuBERT-large L8 | 43.8 | 2/16 | 3.1 | 0/21 |
| HuBERT-large L12 | 44.0 | 2/16 | 3.8 | 0/21 |
| HuBERT-large L16 | 44.5 | 1/16 | 5.1 | 0/21 |
| XLS-R 300M L8 | 42.3 | 1/16 | 3.9 | 0/21 |
| XLS-R 300M L12 | 42.3 | 2/16 | 5.1 | 0/21 |
| XLS-R 300M L16 | 42.2 | 2/16 | 4.7 | 0/21 |
| MMS-1B L16 | 43.0 | 2/16 | 6.9 | 0/21 |
| MMS-1B L24 | 43.4 | 2/16 | 7.1 | 0/21 |
| HubertFA SynthGT posteriors | 76.5 | 4/16 | 32.6 | 5/21 |
| HubertFA SynthGT posteriors + log-mel | 58.4 | 6/16 | 11.4 | 2/21 |
| learned v6 alone | 41.8 | 1/16 | 2.8 | 0/21 |
| learned v6 + log-mel (gated) | 38.7 | 0/16 | 2.7 | 0/21 |
| learned v6 + log-mel + M1 DTW | 33.1 | 0/16 | 2.7 | 0/21 |

### Suite D: held out, evaluated once

| system | D mean | D median | D ≤20 ms | D p90 | cases better than raw |
|---|---|---|---|---|---|
| raw (no alignment) | 65.1 | 50.0 | 23.0% | – | – |
| HuBERT-base L6 + log-mel | 42.6 | 20.0 | 52.8% | 121.0 | 12/12 |
| MERT-v1-95M L4 + log-mel | 47.8 | 20.0 | 52.8% | 131.0 | 11/12 |
| MERT-v1-330M L10 + log-mel | 45.2 | 20.0 | 51.7% | 130.0 | 12/12 |
| learned v6 + log-mel (gated) | 42.0 | 20.0 | 52.3% | 120.0 | 12/12 |
| RysUpAlign 2.1 engine (learned v6 + M1, C++) | 43.0 | 20.0 | – | – | 11/12 |

D medians are all exactly 20.0 ms because the CREPE truth sits on a 10 ms grid. With 12 cases the
differences between the best rows are within noise: a case-level bootstrap of the event-weighted mean
difference gives engine − HuBERT-base L6 + log-mel = +0.4 ms (95% interval −4.5 to +5.8) and learned v6
+ log-mel − HuBERT-base L6 + log-mel = −0.5 ms (−2.8 to +1.8).

### What the results show

- **Suite R (real pairs).** Learned v6 + log-mel + M1 is the best system we measured: 39.8 ms mean /
  14.8 ms median / 58.4% within 20 ms (37.6 / 13.4 / 61.7% on Rc), against 44.8 / 20.0 / 49.9% for the
  best public features we tried (MERT-v1-95M L4 + log-mel) and 52.0 / 34.8 / 33.7% for no alignment.
  Most of the gain over the plain learned features comes from the DTW method (M1: 43.8 → 39.8 mean,
  20.3 → 14.8 median). The learned features without M1 are only slightly better than MERT-v1-95M:
  alone they tie it on mean and are worse on median (46.0 / 23.5 vs 46.1 / 20.6); with log-mel they
  have a 1 ms lower mean and the same median and ≤20 ms (43.8 / 20.3 / 49.5% vs 44.8 / 20.0 / 49.9%).
  The C++ engine reproduces the Python result (39.9 / 14.9 / 58.2%).
- **Suite S (exact truth).** Learned v6 + M1 has the lowest mean (3.2 ms), no case worse than raw, and
  stays at 3.0–3.9 ms on harmony and octave doubles, where log-mel DTW fails (203 / 42 ms) and
  HuBERT-base features lose 3–4x accuracy. Unison S cases are easy for every reasonable feature
  (2.6–2.9 ms); the differences are in h and o.
- **Suite D (held out).** The method's gain on R did **not** carry over to D. The learned model and the
  engine (42.0 and 43.0 ms) are statistically tied with HuBERT-base L6 + log-mel (42.6 ms) and slightly
  ahead of MERT-v1-330M L10 + log-mel (45.2 ms) and MERT-v1-95M L4 + log-mel (47.8 ms). Every system
  roughly halves the raw error (65.1 ms). A new held-out set would be needed for any further claim.
- **Commercial tools were not evaluated.** We did not have access to commercial alignment plug-ins
  (for example VocAlign) under conditions that allow scoring, so this benchmark makes no claim about
  them. Their output can be scored with RAB by converting a result into a warp (see "Score your own
  aligner").

### Known problems and caveats

- **R truth bug.** `CSD_ND_tenor_3.f0` in the Choral Singing Dataset is an exact copy of
  `CSD_ND_tenor_2.f0` on all co-voiced frames, while the tenor_3 audio is a different take (sample
  correlation −0.005; the annotation matches tenor_2's audio at 92% of frames within 0.25 st and
  tenor_3's at 49%, against 90–98% for the other 47 files). The truth of **R_ND_tenor_13 and
  R_ND_tenor_23** (418 events, 7.8% of R) is therefore invalid. We kept them in R so that results stay
  comparable and report Rc (without them) alongside. `analysis/method_truth.py` reproduces the check.
- **Note-change truth is ambiguous at the 20 ms level.** Two reasonable definitions of a note-change
  time (maximum step vs midpoint crossing) disagree by 17 ms median on R, and large-offset matches
  (> 150 ms) are often mismatched notes. We estimate that no aligner can get below about 25 ms mean on R
  as built. Details: `analysis/METHOD_RESULTS.md`.
- **Selection on R.** About 15 checkpoints / runner variants and the M1 levers were compared on R (M1's
  levers were chosen on even-indexed cases and confirmed on odd-indexed ones). R numbers for our systems
  are therefore optimistic; D is the clean test.
- **S is synthetic in its timing.** The audio is real, but the timing errors are generated
  (cut-and-slide), and the guide is the same performance as the double. S measures precision and
  robustness to pitch / timbre changes, not the ability to align two different performances.
- **D is small** (12 cases, one voice type, automatic truth on a 10 ms grid).

## The learned model (v6)

- **Encoder:** HuBERT-base (`facebook/hubert-base-ls960`, Apache-2.0), frozen, first 9 layers.
- **Head (1.66 M parameters):** LayerNorm and a learned weighted sum of layers 3–9, Linear 768→256,
  ×2 upsampling to 100 fps, plus a log-mel branch (64 bands at 25 ms and 80 high-resolution bands
  over 80–2000 Hz at 64 ms) on the same 10 ms grid, 3 residual dilated Conv1d blocks, Linear → 128,
  L2-normalised.
- **Output:** 128-d unit embeddings at 100 fps, frame k centred at 10k + 12.5 ms.
- **Training:** symmetric frame-level InfoNCE (positive = the true fractional frame of the other take,
  negatives = all other frames within ±1 s) on synthetic doubles with exact timing (WSOLA slop, pitch,
  formant and colour changes) made from our own studio vocals and VocalSet, plus pseudo-labelled real
  doubles. No benchmark audio was used for training. Recipe and data table: `learned/README.md`.
- **Files:** `rysupalign_learned_v6.onnx` (fp32, 305 MB) and `rysupalign_learned_v6.int8.onnx`
  (dynamic int8, 112 MB; same accuracy on RAB, see the tables). Input `wav` [1, S]: mono 16 kHz,
  normalised to zero mean and unit variance; output `emb` [1, T, 128].
- **Download:** https://huggingface.co/rysupaudio/rysupalign-learned-v6 (primary) or the
  [v1.0 GitHub release](https://github.com/rysupaudio-lab/rysupalign-bench/releases/tag/v1.0).

| file | sha256 |
|---|---|
| rysupalign_learned_v6.onnx | `db3bf5b88834bba926c147d8785e829845c7827248efc74e2966f0bec7ec8b32` |
| rysupalign_learned_v6.int8.onnx | `e48d692579af3d8ed710ff99e47a1c37d0c1e0205440d5a69d1521a69f505246` |

**Licence of the weights:** PolyForm Noncommercial License 1.0.0 (`MODEL_LICENSE.md`); the HuBERT
encoder weights inside the file remain under Apache-2.0 (`NOTICE`). Commercial licensing is available
from Rys Up Audio: https://rysupaudio.com/pages/contact-us

## Reproduce

Requirements: Python 3.10+ (3.13 was used), the packages in `requirements.txt`, and the Rubber Band
command-line tool (`rubberband`, version 4.0.0 was used) for the harmony / octave guides of suite S.
`torch` and `transformers` are needed only for the self-supervised baselines and for training.

```bash
git clone https://github.com/rysupaudio-lab/rysupalign-bench && cd rysupalign-bench
python -m pip install -r requirements.txt

# 1. Datasets (not redistributed here)
#    R: Choral Singing Dataset, https://doi.org/10.5281/zenodo.2649950 (ChoralSingingDataset.zip, 1.1 GB)
#    D: Dagstuhl ChoirSet 1.2.3, https://doi.org/10.5281/zenodo.4618287 (5.1 GB)
#    S: the 5 Cambridge-MT sessions listed in rab/s_sources.json ("download" field), from
#       https://www.cambridge-mt.com/ms3/mtk/ ; the fury takes are in data/fury
python rab/build_rab.py R --csd datasets/ChoralSingingDataset
python rab/build_rab.py S --mt datasets/cambridge-mt      # folder holding the unzipped sessions
python rab/build_dcs.py datasets/DagstuhlChoirSet_V1.2.3   # held out: please evaluate once
# Each builder checks every case's truth against the sha256 in rab/manifest.json.
# Cases go to ./cases (or $RAB_CASES); caches go to ./work (or $RAB_WORK).

# 2. Model (for the learned runners)
mkdir -p models && curl -L -o models/rysupalign_learned_v6.onnx \
  https://huggingface.co/rysupaudio/rysupalign-learned-v6/resolve/main/rysupalign_learned_v6.onnx

# 3. Score
python rab/rab_score.py raw            runners/identity.py   --suite R
python rab/rab_score.py mel_dtw        runners/fdtw.py       --env FEAT=mel --save-errors
python rab/rab_score.py hubert6_mel    runners/fdtw.py       --env FEAT=hubert@6+mel --jobs 3
python rab/rab_score.py learned_v6_mel runners/learned_onnx.py --save-errors
python rab/rab_score.py learned_v6_mel_m1 runners/m1_dtw.py  --save-errors
python rab/rab_score.py learned_v6_mel_m1 runners/m1_dtw.py  --suite R --exclude R_ND_tenor_13,R_ND_tenor_23
python results/summarize.py
```

`python rab/rab_score.py --help` lists the options (`--suite R|S|D`, `--filter`, `--exclude`,
`--jobs`, `--env K=V,...`, `--model`, `--save-errors`). Without `--suite` the scorer runs every built R
and S case; D runs only with `--suite D`. Results go to `results/NAME.json` (per case) and, with
`--save-errors`, `results/NAME.errors.npz` (per point). Expect small differences (a few tenths of a ms)
from library versions; the S truth itself is bit-exact.

### Score your own aligner (runner contract)

`rab_score.py` calls your runner once per case:

```
RUNNER MODEL GUIDE.wav DOUBLE.wav
```

(a `.py` runner is started with the current Python; `MODEL` is `--model` / `$RAB_MODEL`, default `-`,
and may be ignored). The runner must print, as the **last line of stdout**, a JSON object

```json
{"warp": [[0.00, 0.00], [0.51, 0.47], [0.52, 0.49], ...]}
```

with monotonic `[guideTime, doubleTime]` pairs in seconds: the double's material that belongs at
`guideTime` is at `doubleTime` in the double file, linearly interpolated between knots. An empty list
means identity. Other keys are optional. `runners/identity.py` is the smallest example and
`runners/fdtw.py` a complete one. To score a tool that only returns audio, recover its warp first (for
example by aligning the tool's output to the unprocessed double, which differ only in timing).

## Repository layout

| path | contents |
|---|---|
| `rab/` | `build_rab.py` (suites R and S), `build_dcs.py` (suite D), `rab_score.py` (scorer), `manifest.json` (cases, truth checksums), `s_sources.json` / `s_cases.json` (suite-S sources and per-case random-generator states) |
| `runners/` | `identity.py`, `fdtw.py` (feature DTW), `ssl_feats.py` (self-supervised features), `embed_onnx.py`, `learned_onnx.py`, `m1_dtw.py`, `method_lib.py`, `method_variants.py` |
| `analysis/` | the method study (`METHOD_RESULTS.md`, `method_run.py`, `method_diag.py`, `method_truth.py`) |
| `learned/` | model, training, data generation and ONNX export code (`learned/README.md`) |
| `results/` | reruns with this code, `original/` research runs, `pooled_stats.json`, `summarize.py` |
| `data/fury/` | 7 gang-vocal takes, CC BY 4.0, © Rys Up Audio |

## Licences

- Code: MIT (`LICENSE`).
- Model weights: PolyForm Noncommercial 1.0.0 (`MODEL_LICENSE.md`), HuBERT encoder Apache-2.0 (`NOTICE`,
  `LICENSES/Apache-2.0.txt`). Commercial licensing: https://rysupaudio.com/pages/contact-us
- `data/fury`: CC BY 4.0, © Rys Up Audio (`data/fury/README.md`).
- Datasets used by the build scripts keep their own licences: Choral Singing Dataset and Dagstuhl
  ChoirSet CC BY 4.0; Cambridge-MT multitracks under the Cambridge-MT terms (not redistributed here);
  VocalSet (training only) CC BY 4.0.

## Citations

If you use RAB, please also cite the datasets it is built from.

- Choral Singing Dataset: H. Cuesta, E. Gómez, A. Martorell, F. Loáiciga. "Analysis of Intonation in
  Unison Choir Singing." Proc. 15th International Conference on Music Perception and Cognition (ICMPC),
  2018. Data: https://doi.org/10.5281/zenodo.2649950
- Dagstuhl ChoirSet: S. Rosenzweig, H. Cuesta, C. Weiß, F. Scherbaum, E. Gómez, M. Müller. "Dagstuhl
  ChoirSet: A Multitrack Dataset for MIR Research on Choral Singing." Transactions of the International
  Society for Music Information Retrieval 3(1), 2020. https://doi.org/10.5334/tismir.48
- VocalSet: J. Wilkins, P. Seetharaman, A. Wahl, B. Pardo. "VocalSet: A Singing Voice Dataset." Proc.
  ISMIR 2018. Data: https://doi.org/10.5281/zenodo.1442513
- HuBERT: W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, A. Mohamed. "HuBERT:
  Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units." IEEE/ACM
  Transactions on Audio, Speech, and Language Processing 29, 2021. arXiv:2106.07447
- MERT: Y. Li, R. Yuan, G. Zhang, Y. Ma, X. Chen, H. Yin, et al. "MERT: Acoustic Music Understanding
  Model with Large-Scale Self-supervised Training." ICLR 2024. arXiv:2306.00107
- CREPE: J. W. Kim, J. Salamon, P. Li, J. P. Bello. "CREPE: A Convolutional Representation for Pitch
  Estimation." Proc. IEEE ICASSP 2018. arXiv:1802.06182
- Cambridge-MT "Mixing Secrets" multitrack library, M. Senior, https://www.cambridge-mt.com/ms3/mtk/

Cite this repository with `CITATION.cff`. Questions and licensing: https://rysupaudio.com/pages/contact-us
