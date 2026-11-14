# Contact and RGB development modules — implementation record

Status: implemented and engineering-tested, **not a learned-policy result**. Historical files are untouched. `ContactEnv` reuses Controller B's actual `step`/unknown execution mapping and the archived pilot robot, with a separate newly generated MJCF and strict model-hash snapshots. It explicitly bypasses the legacy constructor that rewrites `assets/pilot_effective_model.linux.xml`.

## APIs

- `ContactEnv(config=None, delay=0, alpha=.5)`; `reset(seed,delay,jitter=.002)`, `step(action)->x28`, `reference(horizon,start_frame=0)->[H,14]`, `reward(action,prev)`, `metrics()`, `render_rgb(size=64,camera_shift=None)`, `snapshot/restore`, `close`.
- `TorchContactReward(env)(x28,a14,prev14)` computes the **same continuous physical reward** using exact hinge-chain transforms compiled from MuJoCo's model. It is a state-task reward implementation, not a deployable input to the RGB policy. No fitted FK network is used.
- `VisionPolicy(obs_dim=32,record_dim=29,adapter_dim=32)`; `encode(rgb)`, `action_from_latent(z,c_obs,theta)`, `forward(rgb,c_obs,theta)`, `freeze_base`. `record_dim=0` gives the RGB-only input condition.
- `fit_base` performs train-data BC fitting of encoder/base and then freezes them. Its caller must enforce episode splits and count demonstration costs. The external 32-vector theta remains the policy optimization variable. The adapter's randomly initialized basis is frozen with the base.

## Task scope and geometry

A rigid 8 mm diameter, 60 mm long cylindrical peg is attached to the left end effector. A fixed square aperture is 10 mm wide, with a bottom at 35 mm insertion depth. Initial nominal tip is 15 mm before the mouth. Right arm remains nominal. Robot-body/self contacts are disabled; **only the tool--fixture collision pair is active**. This is restricted single-arm tool insertion, not grasping or bimanual assembly.

Success requires depth >=20 mm and <=36.5 mm, lateral error <=2 mm, axis error <=5 degrees, penetration <=1 mm, maintained for .5 s by actual control time and achieved by 10 s. Dense reward is independent of the success predicate: `-100*lateral² -20*(depth-.03)² -.04*(1-axis_dot) -.0005*mean(qdot²) -.05*mean((a-prev)²)`.

The nominal reference is IK-generated once from public fixture/robot geometry, interpolating align then insert, then hold. The engineering feedback controller uses `.55*(ref-qtarget)+.1*(ref-q)` clipped to ±.03. This controller is a feasibility witness, not a trained EAGM policy or evidence that the 28-parameter actor improves success.

## Tests actually run

`tasks_contact_tests_final.log`: four passed tests, FK and reward parity, autograd vs MuJoCo FK finite differences, full integration/controller/task snapshot replay, and physical insertion/contact.

`tasks_vision_tests_final.log`: four passed tests, real RGB dependency, nonzero adapter gradient, frozen base parameters, absence of privileged-state arguments, RGB-only input, and identical latent/real encoder action paths.

`tasks_feasibility.json`: four physical episodes, each 294 control steps at actual dt=.034s (9.996s); d0/d1/d2 fixed B scripted successes at 3.570/3.570/3.604s. Contact fractions ~.57 and penetrations <1e-5m. **Identity-controller d0 fails** (lateral error ~12.1 mm), retained rather than removed. Controller variation is therefore a real task limitation and success is not claimed to generalize.

EGL rendering produced real 64x64 RGB images. The first camera/lighting version was poorly framed/dark; retained as `tasks_contact_render.png`. A closer entry-side view with headlight is `tasks_contact_render_v2.png`, and final inserted pose is `tasks_contact_inserted.png`. Rendering refreshes derived positions in separate scratch MuJoCo data and does not change live integration state; it is not an animation proxy for success.

## Limits to retain in the development report

The initial implementation uses **one RGB frame**, not the optional two-frame/sequence encoder in the draft. Any experiment using it must disclose this input simplification. No world-model future true images may be supplied to imagined policy actions. No claim of end-to-end CNN policy optimization is warranted after encoder freezing. Contact-rich long-horizon learning and robustness remain to be established by the separately run development seeds; these engineering tests do not establish either.

Snapshot hash includes fixture/lighting asset, even when state dimensions match. Model width derives from current MuJoCo state size, never hardcoded to 197. Original source meshes remain an external prerequisite at `/home/aaa/HIL-OpenArm/third_party/openarm_mujoco/v2/assets`; raw generated assets are under `eagm2/assets`.
