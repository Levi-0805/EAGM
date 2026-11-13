# Stage 8.1 research decision

Exploratory seed20 frozen-model causal diagnostic. No new world-model training, no independent confirmation. Sources: runs/stage8_1_geometry/; script analysis/stage8_1_reports.py.

```json
{
  "case": "B",
  "common_theta_gradient_improved": true,
  "mean_return_improved": true,
  "positive_starts": 4,
  "five_seed_launch": false,
  "independent_confirmation": false
}
```

Case B here means partial mitigation without restored reliability: common-theta mean cosine remains negative and true return remains below theta0. Small numerical improvements do not meet the requested significant recovery criterion. No significance claim is made. Geometry derivative replacement is an intervention on the complete reward derivative with dynamics frozen; it is not a proof that Jacobian error alone caused all Stage7 failure. Position residual and J both contribute. The selected Dir model was originally trained with surrogate geometry and true FD labels, so geometry replacement may expose training/model derivative compensation. No retraining or cross-seed validation was performed.

Do not launch five seeds from one exploratory seed. If both gradient and return improve, next is one fresh, preregistered engineering replication with verified geometry and unchanged budgets; otherwise investigate dynamics Jacobians/horizon/update direction before architecture changes. C1/C2 remain exploratory, C3 requires independent calibration episodes. No new losses, vision, hardware or paper-core claim changes. Exact geometry here is an oracle diagnostic, not yet a deployable learned-method contribution.


Total added physical control steps including actual-state coverage: **19488/20000**. Main and supplemental raw budgets are preserved separately; combined accounting is total_budget.json. Static metrics CSV, component/goal/configuration summaries and SHA-256 inventory are in the run directory. No prior file changed relative to Stage8.1 preservation manifest.

Mechanistic refinement: on held-out trajectory states, native reward-gradient cosine is 0.441; substituting true J alone with the FK residual yields 0.409, while retaining FK J with the true positional residual yields 0.957. Thus forward positional bias dominates this particular state-level directional error more than J orientation. This is a controlled state-level decomposition, not proof that the same factor dominates closed-loop policy failure. Frozen true-geometry policy gradients remain negative on average; investigate learned dynamics sensitivity and training-time derivative compensation next. No new five-seed proposal is justified by these results.
