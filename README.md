# Small-network phoneme recognition: structured vs generic readouts at equal size

Miltiadis Argianis Karakitsos

This repository holds the plan, the code and the result records of a research project on small neural networks for phoneme recognition under a 2 MB model-size limit. The central question is whether structured readouts (a phoneme tree, or per-speaker-private heads) beat a generic network of the same exported byte size when training data is small, or whether at that size the generic model is just as good. The work also records the individual training tricks that were tested along the way, each against a byte-matched or equal-cost control.

The data is LibriSpeech (16 kHz read speech; the small-data experiments use a 5.4 h train-clean subset, calibration uses train-clean-100 and a 5,000-image CIFAR-100 split for the network-size questions). Background noise for training comes from ESC-50. No audio is included in this repository: the scripts read their inputs from a data directory given by the environment variable `DATA_DIR` (default `data/`), and large caches and checkpoints go to `SCRATCH_DIR` (default `scratch/`).

Every comparison is made at equal exported size in bytes, and the evaluation protocol is fixed before a run starts. A difference counts only if it clears a pre-declared margin with an interval that excludes zero; anything else is reported as inconclusive rather than as a gain.

To run the main pieces: `code/base6.py` trains the conventional comparator; `code/base6s.py` trains the structured/private-readout arms; `code/base6sy.py` trains the byte-matched top-only readout arms; `code/small_net/cifar_q1.py` runs the image calibration arms. The scripts under `code/step0/` are the step-0 tooling (manifests, batch plans, label and closure checks, scoring). Results and their known-answer checks are in `results/` as JSON, and the full plan and the appended results log are in `docs/PLAN.md` and `docs/RESULTS.md`. The scripts need Python 3 with NumPy and PyTorch.

Status: research in progress. The work documented here is the plan, the calibration, and the exploratory runs recorded so far. Results are EXPLORATORY unless `docs/RESULTS.md` records a pre-declared test as passed; that log records one such pass for the speaker-warp training comparison (arm 2a) and reports the 5.4 h structured-versus-generic readout comparison as a tie or slightly better that missed its 0.5-point bar. The repository is versioned at 0.1 and will change as the work continues.

## License

MIT, 2026, Miltiadis Argianis Karakitsos. See `LICENSE`.
