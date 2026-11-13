# Stage 8.3 preregistration

Registered before collecting new test trajectories. Source hashes and complete constants: `configs/stage8_3_preregistered.json`.

Frozen Dir checkpoint, no retraining. K2 episodes0,2; K4 adds3,5, all frame0 from original training split. These two additions are new to the Stage8.2 two-start comparison, not newly collected training episodes. Validation episodes22,30 from original validation split, frame0; they are independent trajectories, not claimed never inspected historically. Fresh test: collection seed83017, two200-step trajectories, frame0. Distinct episode keys include collection seed. Original six test trajectories remain historical exploratory only. Two fresh trajectories are not independent model seeds and do not provide confirmatory inference.

B vs D isolates learned vs privileged transition derivatives with the same true reward derivatives and common forward values. At theta0, average K gradients. Raw direction g/||g||; fresh Adam with m=v=0, standard bias corrections and eps1e-8. Both updates rescaled to L2=0.03*sqrt(28), the fixed nominal Stage8.2 lr times sqrt(parameter dimension), not a test-selected magnitude. One update only; no projection should be active. H8 gradients, H16 true EE return; retain H8 return to separate horizon effects.

Budget upper bound: four H8 full-state oracle paths 4*8*(2*168+1)=10784; train FD4*28*2*8=1792; validation/fresh FD4*28*2*8*2=3584; collection400; nine policies (theta0+8factor cells)*14starts*16=2016. Total18576, hard cap20000 includes failures and tests that step physics. Cached query savings logged. No need to compute full Jacobians at validation/test points: direct policy FD suffices for true reference and saves budget.

Hypothesis screens fixed before outcomes: H1 requires K4 improvement in aggregate alignment and true mean return on BOTH validation and fresh test for BOTH oracle optimizers, and positive paired effects on both fresh starts. H2 analogously requires raw>Adam for BOTH K values and both groups, positive on both fresh starts. No arbitrary effect-size tuning; report magnitudes and inconclusive small/discordant effects. Neither screen establishes statistical significance or authorizes five seeds. No tuning on validation in this one-shot diagnostic.

Stop if either reference-scale instability or replay/implementation mismatch invalidates interpretation. Stability mask uses spread<=max(1e-9,.1*abs(g_.01)); full cosine primary. Training single-scale masks are unavailable. FailRate delta<0; no task-success threshold invented.

Paper work is separate: eight-page draft framework under paper_eagm/, no changes to CVPR2027. Oracle outcomes never become deployable-method claims.
