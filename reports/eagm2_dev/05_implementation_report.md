# EAGM 2.0 development implementation

This is an exploratory implementation, not an accepted research contribution.
Author approval covers nine development roots only. The 45 confirmation roots
remain locked pending a second author decision. Historical paper and data are
not edited. The development branch is `EAGM-2.0-full-validation`, based on
`81ecd5f3ad1727c86fb8bc67f140e5def6e500c6`.

## Implemented interfaces and evidence

* `eagm2/prior.py`: causal nominal FIFO command for delays 0, 1, 2 with explicitly
  available previous filter state. Unknown gain, extra lag, rate limit and
  smoothing remain outside the nominal prior. The actual Controller B Cartesian
  test executes 972 steps across 108 configurations; it compares the nominal
  queue and an independently computed execution stack separately.
* `eagm2/models.py`: original/delay/free command heads, optional residual,
  recurrent execution estimate and window memory control. Observation updates
  consume observed state/record/previous action; imagination propagates learned
  memory. Future observations and privileged controller internals are not model
  inputs. The privileged API is separate and has not been trained as an oracle.
* `eagm2/gradients.py`: matched actor-parameter finite differences at three
  radii; boundary-aware stencils, stability mask, direction/norm/joint losses.
  Full-coordinate gradient metrics are primary; mask-only metrics are separate.
  Undefined cosine/sign metrics remain null. Labels are fixed during policy
  optimization; adaptive refresh has not been implemented or validated.
* `eagm2/training.py`: paired data/minibatch streams and name/shape-keyed common
  parameter initialization. The eight D/R/G combinations support factorial
  comparisons. Capacity-matched MLP/window-GRU, free command head, and gradient
  loss ablations are separate methods. Final checkpoints, not held-out winners,
  are evaluated.
* `eagm2/legacy_adapter.py`: unchanged historical EAGMBase forward/gradient
  adapter. `legacy_base/dir` in the common new protocol differ from the separate
  strict historical protocol reproduction. The new `ff_base` is not renamed
  historical EAGMBase: its command normalization/residual scaling differ.
* PAML/VaGraM are explicitly adaptations, not full published reproductions.
  Contextual, train-only FQE supplies their critic. Fixed theta-zero actor,
  continuing discounted target, incomplete observable state, and behavior
  coverage are limitations. Critic training costs and held-out TD residuals are
  saved; TD residual is not ground-truth value error.
* `eagm2/contact.py`: actual MuJoCo tool/fixture contact, exact differentiable
  kinematic reward, snapshot restoration and success/timeout criteria. Scripted
  feasibility is not evidence of learned-policy success. Original robot XML is
  not overwritten.
* `eagm2/vision.py` and `visual_training.py`: actual RGB enters a single-frame
  CNN policy. Train-only behavior cloning freezes the encoder/base and a 32-D
  external adapter is subsequently optimized. Learned latent reward consumes
  no simulator state; physical evaluation uses actual physical reward. This is
  neither full-network visual RL nor the planned two-frame architecture.
* `eagm2/budget_controls.py` and `extra_data.py`: actual simulator control-step
  caps for Direct FD/SPSA and additional-data Base. The latter retains the same
  model update count, so more data does not imply equal per-sample training.
  Direct methods may complete fewer policy updates under their shared cap.

## Reproducibility and evaluation

New attempts are immutable directory names. Every attempt captures source,
configuration, environment, train/validation/development-test episode IDs,
checkpoints and actual simulator accounting. Normalization and visual training
use train episodes only. All nine roots are exploratory. Thirty test starts per
full root are repeated measurements, not thirty independent seeds.

Contact/vision training starts rotate over saved phases while crossing delay
and phase; held-out evaluation starts remain fixed at frame 32. Histories are
32 observable frames, not a guaranteed sufficient belief or full reset history.
H8 tracking/H32 contact imagination optimizes a local objective; the latter
must not be described as differentiating the full ten-second success event.

Strict deterministic training is enabled. An initial visual smoke failed
because CUDA adaptive-pooling backward was nondeterministic. An equivalent
2x2 bin-mean implementation replaces that operator, with forward/backward
equivalence tests; the failed attempt remains in the ledger.

## Limits pending development results

No claim of consistent calibration gain, multi-step gradient advantage,
cross-controller generalization or CVPR acceptance readiness follows from
engineering tests. OOD execution/camera, RGB-only information ablations,
refresh-frequency selection and trained privileged-oracle experiments are not
completed by the current common-model matrix. Those gaps must remain visible
at the next author review. A module may be rejected after development; no
positive outcome is required or presumed.
