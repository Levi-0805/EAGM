# Development verification gate

The complete discovered suite passed **78 tests in 32.501 seconds** before
`development_v1` was launched. Exact verbose output is
`tests_full_discovery_final.log`; the batch gate records its SHA-256.

Coverage includes true Controller B timing/FIFO and unknown execution stack,
delay switching, joint bounds, alpha inversion limits, causal memory and
privileged API isolation, model-gradient finite differences, finite-difference
stencils/masks/calibration, PAML/VaGraM gradient flow, contextual critic data
isolation, contact FK/reward/snapshots, actual RGB dependence and input isolation,
train-only visual normalization, checkpoint state restoration, capacity checks,
disjoint data streams, simulator budget guards and nine-root authorization.

The controller matrix performs 972 real control steps across 108 settings.
Maximum nominal mismatch and independent executed-stack mismatch were both zero.
These are distinct checks: the nominal command is not asserted equal to the
unknown-controller executed command.

Three critic integration functions originally ran only in a separate harness.
The strict batch gate detected they were not in unittest discovery, so wrappers
were added and the entire suite rerun. Earlier 75-test and smaller logs remain
as historical records; they are not the final gate.

Successful smoke gates: tracking `smoke_matrix_v3` (all 21 methods), contact
`smoke_v1`, vision `smoke_v2`. Contact and vision smoke cover representative
methods rather than every final combination. The failed initial tracking and
visual attempts are retained, including their costs and tracebacks.

The CUDA adaptive-pooling failure was fixed using mathematically equivalent
deterministic bin means. Tests compare both forward and input gradients against
adaptive pooling over even, odd and small spatial sizes; the repaired actual
GPU vision smoke succeeded with strict deterministic algorithms enabled.

Passing tests establish implementation properties, not scientific effectiveness.
Learned physical-model errors, gradient fidelity, success rates and cross-seed
benefits require the actual exploratory batch. Confirmation remains unauthorized.
