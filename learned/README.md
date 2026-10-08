# Learned alignment embeddings (training code)

This folder holds the code that trained the RysUpAlign learned alignment model v6 and exported it to
ONNX. The training data is not published (see "Data" below), so the recipe can be re-run on your own
material but the released weights cannot be reproduced bit for bit from this repository alone.

## Model (`model.py`)

- HuBERT-base (`facebook/hubert-base-ls960`, Apache-2.0) is frozen and truncated to 9 transformer layers.
- Hidden states of layers 3–9 each get a LayerNorm, then a learned softmax-weighted sum.
- The sum goes through Linear(768→256) and a ×2 transposed convolution (50 → 100 fps).
- It is added to a log-mel branch on the same 10 ms grid: 64 mel bands with a 25 ms window (60–8000 Hz)
  plus 80 high-resolution bands with a 64 ms window over 80–2000 Hz, so sung harmonics and pitch are resolved.
- Then 3 residual dilated Conv1d blocks (kernel 5, dilation 1/2/4), Linear → 128, L2-normalised.
- Output: 100 fps, frame k centred at 10k + 12.5 ms. The trainable head has **1.66 M parameters**.

## Training (`train.py`)

- Loss: frame-level InfoNCE, symmetric (A→B and B→A). The positive is the true fractional B frame, with
  the label split linearly between the two neighbouring frames; every other frame within ±1 s is a
  negative. Pseudo-labelled pairs use a Gaussian soft target (σ = 20 ms).
- Unison pairs are sampled ×3. AdamW, cosine schedule, batch 16 pairs of 6.4 s.
- Recipe: v3 = 4000 steps from scratch on tr1+ps1 (0.7/0.3), lr 2e-3; then v6 = 3000 steps from v3 on
  tr1/trvs/ps1/psvs (0.25/0.3/0.15/0.3), lr 1e-3:

```
python learned/train.py v3 --pairs tr1,ps1 --mix 0.7,0.3 --unison 3 --mel 64 --melhi 80 --steps 4000
python learned/train.py v6 --init v3s4k --pairs tr1,trvs,ps1,psvs --mix 0.25,0.3,0.15,0.3 --unison 3 --mel 64 --melhi 80 --steps 3000 --lr 1e-3
```

Working files go to `$RAB_WORK` (default `./work`): `training/src*` (source chunks), `training/pairs`,
`training/pseudo*`, `training/ckpt`.

## Data

| set | source | how | licence |
|---|---|---|---|
| tr1 (12k pairs) | ~28 min of Rys Up Audio's own unreleased studio vocal recordings (`prep_sources.py`, `$OWN_VOCALS_DIR`) | `gen_pairs.py`: exact WSOLA timing slop (grain positions recorded), Rubber Band R3 pitch/formant shift (time-neutral, checked), EQ / drive / reverb / noise | owned, not distributed |
| ps1 (4k) | 236 real double takes in the same recordings, found by `mine_doubles.py` | pseudo labels = HuBERT-base L6 + log-mel DTW (`gen_pseudo.py`) | owned, **pseudo** |
| trvs (8k) | VocalSet 1.2 (18 training singers; female9 / male11 held out; no long tones; excerpts ×3) (`prep_vocalset.py`, `$VOCALSET_DIR`) | `gen_pairs.py --src src_vs --v 2` | CC BY 4.0 (Wilkins et al. 2018) |
| psvs (3k) | 130 VocalSet different-singer excerpt pairs | `mine_vocalset_cross.py`: two aligners must agree within 20 ms, otherwise the frame is masked | CC BY 4.0, **pseudo** |

Never used for training: the Choral Singing Dataset, Dagstuhl ChoirSet, the Cambridge-MT stems, the fury
takes and all other RAB audio.

## Evaluation and export

- `learned_dtw.py` is the original RAB runner on a PyTorch checkpoint
  (`CKPT=<ckpt> FEAT=learned+mel MELW=2 GATE=1`). The public runner `runners/learned_onnx.py` does the
  same on the released ONNX graph.
- `export_learned.py CKPT.pt OUT.onnx` writes the single ONNX graph (HuBERT layers 1–9 + head + log-mel
  front end with the STFT as fixed conv1d kernels). The int8 file was made with
  `onnxruntime.quantization.quantize_dynamic` (weight type int8).

## Findings

- The synthetic task alone is easy: raw HuBERT L6 already reaches about 7 ms median on it, and gains there
  transfer only partly to real choir pairs.
- The largest real-data gains came from the 100 fps spectral branch, weighting the external log-mel block
  at 2, and unison emphasis.
- Pitch-behaviour augmentation (vibrato / scoops / drift) and VocalSet mostly improved the mean and the
  p90 tail, not the median.
- No help: a trainable built-in mel skip block, longer training on the augmented set. Ensembles gave
  +0.4–0.8 points of ≤20 ms at 2–3× the cost.
- About 15 checkpoint / runner variants were compared on suite R, so there is some selection on that
  suite (see the main README). Suite D was held out from all of it.
