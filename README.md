# Button-press sleep timing from wrist accelerometry

Code accompanying **Predicting Sleep Intent Boundaries from Wrist Actigraphy** .

The model localises a participant's bedtime (or wake time) within a 24-hour wrist-accelerometer recording. Labels come from the device's event button: participants press it when going to bed and when getting up, and a movement-based selection step picks the most plausible pair of presses for each night. A patch-based temporal network then learns to predict the press time directly from the raw tri-axial signal.

```
(N, T, 3) accelerometer day, 15:00 → 15:00
   └─ PatchEmbedding     non-overlapping patches → strided Conv1d stack → (N, P, d_model)
   └─ temporal trunk     non-causal TCN  or  pre-norm transformer
   └─ head               distribution: one logit per patch  (soft-label cross-entropy)
                         regression:   one scalar per night (L1/L2, ablation)
```

## Repository layout

| File | Contents |
| --- | --- |
| `model.py` | `SleepModel` (PyTorch Lightning), `PatchEmbedding`, `TransformerTrunk`, positional encoding |
| `transformer_encoder.py` | Minimal pre-norm multi-head self-attention encoder |
| `click_candidates.py` | Button-press extraction from GENEActiv `.bin` files and bed/wake pair selection |
| `preprocessing.py` | Raw recording → per-day tensors → participant-level cross-validation folds |
| `notebooks/synthetic_demo.ipynb` | End-to-end demo on simulated data (no study data required) |

## Installation

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
pip install -r requirements.txt
```

`wristpy` is only needed to read raw device files; the model, label selection and demo notebook run without it.

## Quick start (synthetic data)

```bash
jupyter notebook notebooks/synthetic_demo.ipynb
```

The notebook simulates ~700 nights from 120 participants at 1-minute resolution, demonstrates the press-pair selection on a night with a spurious press, builds participant-level folds, and trains the TCN and transformer trunks with both heads. It runs in a few minutes on a laptop CPU. The simulator is a smoke test for the code; its numbers are not indicative of performance on real recordings.

## Method

### Labels: selecting bed and wake presses

Presses are parsed from the page flags of the GENEActiv `.bin` file, debounced (presses < 2 min apart count as one), and assigned to a 15:00-to-15:00 day. Presses between 18:00 and 06:00 are bedtime candidates, those between 03:00 and 14:00 are wake candidates (the overlap is intentional).

For each day, every (bed, wake) candidate pair enclosing 2–14 h is scored by the movement contrast between the enclosed interval and the 60-min periods on either side, excluding 5 min around each press. Activity is the per-minute mean absolute first difference of the signal, and each window is summarised by its median, so a brief night-time awakening does not disqualify a night. Both flanks must be at least 1.5× more active than the interval; among passing pairs, the one maximising log(contrast_bed) + log(contrast_wake) is kept. If none passes, the night's label is missing. Scoring pairs rather than single presses prevents a stray 03:00 press, followed by any movement, from outscoring the real wake press.

All thresholds are module-level constants at the top of `click_candidates.py`, and `get_day_clicks(..., details=True)` returns per-pair diagnostics for auditing rejections.

### Model

Each day is split into non-overlapping patches of `patch_len` samples; a three-layer strided convolution with GroupNorm and GELU maps each patch to a `d_model` vector. A non-causal TCN (`pytorch-tcn`) or a pre-norm transformer encoder then mixes information across patches.

The **distribution head** produces one logit per patch. The target is a Gaussian of width `label_sigma_min` minutes centred on the labelled press, and the loss is soft-label cross-entropy. At inference, the prediction is the expected position under the softmax (`calc_pred`); `calc_pred_local` offers a centre-of-mass within ±`k` patches of the mode instead.

The **regression head** mean-pools over patches and predicts the event position as a fraction of the recording. It is the ablation arm: a single scalar cannot represent more than one candidate event, so on ambiguous nights it falls between them. Both heads are evaluated with the same localisation error in minutes.

### Evaluation

Nights are split into five folds **by participant**, so no individual contributes to both training and evaluation. For fold *i*, fold *i* is the test set, fold *i+1* the validation set, and the rest training. Per-channel normalisation statistics are computed on the training rows only.

## Using the study pipeline

Paths are set with environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SLEEPCLICK_BIN_ROOT` | `./data/raw` | Directory tree containing the raw `.bin` recordings |
| `SLEEPCLICK_DATA_DIR` | `./data` | Working directory for cleaned days and cached tensors |
| `SLEEPCLICK_CLICKS_DIR` | `./data/clicks` | Output directory for per-recording press candidates |

Recording file names are expected to start with a numeric participant id (`{pid}_*.bin`).

```bash
export SLEEPCLICK_BIN_ROOT=/path/to/recordings
export SLEEPCLICK_DATA_DIR=/path/to/workdir

# 1) cut recordings into complete, fully-worn 15:00–15:00 days,
# 2) extract press labels, 3) cache tensors at 10 Hz, 1 Hz, 1 min and 5 min
python preprocessing.py --split-raw --freq 1min --target evening
```

Then train on a fold:

```python
import pytorch_lightning as pl
from preprocessing import create_folds
from model import SleepModel

folds = create_folds(None, freq_name="1min", target="evening", batch_size=32, num_workers=8)
train_loader, val_loader, test_loader = folds.get_fold(0)

model = SleepModel(
    in_channels=3,
    seq_len=1440, sample_rate=1 / 60,      # 1-min resolution
    patch_len=10,                          # 10-min patches → 144 positions
    temporal_model="tcn", num_channels=[128, 128, 128], d_model=128,
    head_type="distribution", label_sigma_min=5,
)
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
```

`seq_len` and `sample_rate` must match the chosen resolution: 864000 / 10 for 10 Hz, 86400 / 1 for 1 Hz, 1440 / (1/60) for 1 min, 288 / (1/300) for 5 min. `SleepModel.predict_hours(x)` returns predictions as clock hours.

Logged metrics are `{train,val,test}/loss`, `/loc_err_mean` and `/loc_err_median` (minutes).

## Citation

```bibtex
@article{TODO,
  title   = {[PAPER TITLE]},
  author  = {[AUTHORS]},
  journal = {[VENUE]},
  year    = {[YEAR]}
}
```

## License

[LICENSE]
