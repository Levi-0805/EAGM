# Stage 8 venue assessment

This is a scope assessment, not an acceptance forecast or deadline recommendation.

CoRL: strongest conceptual fit for execution-aware robot learning if there is a clear algorithmic mechanism, credible independent tasks and hardware or transferable simulation evidence.
ICRA: plausible fit for an execution-interface method with reproducible robotics validation and measured practical cost. A meaningful system demonstration would strengthen it.
RSS: would require a particularly crisp insight/mechanism and convincing robotics evidence; current narrow simulator results alone do not justify confidence.
CVPR: current state-only control experiments offer limited computer-vision contribution. Adding images without isolating perception/execution/model errors would not resolve this. Prioritize a scientifically necessary visual task only after low-dimensional objective validity and method gains pass.

No external public-benchmark reproduction was attempted this cycle. Local CPU runs are feasible; the historical host has an 8GB RTX4060 but this process reports CUDA unavailable. This does not establish that any baseline is impossible. A fair external comparison needs a shared action interface, reset privilege, observation specification, simulator/compute budget and official baseline configuration; no scores should be invented. Before proposing a full baseline run, benchmark memory/time on one task and separate observation/action advantages.
