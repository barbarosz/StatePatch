# StatePatch package status

## Completed and locally validated

- 64-item paired StatePatchBench generated (32 counterfactual pairs across color, position, count, shape)
- CPU-tested transformer activation recorder
- top-level generator call indexing
- full activation donor capture
- full/delta/low-rank subspace patching core
- state-type-specific linear probe sweep
- probe heatmap plotting
- Self-Forcing/Causal-Forcing-compatible Wan adapter
- I2V visual-state capture pipeline
- counterfactual donor/target patch runner
- experiment config
- paper draft
- two-hour runbook
- exact recommended Causal Forcing setup
- Python syntax compilation passed
- unit tests passed (2/2)

## Not executed in this environment

GPU video inference and model checkpoint experiments cannot be executed here because this runtime does not contain the Wan/Self-Forcing/Causal-Forcing model weights or an appropriate GPU setup. Consequently, no empirical StatePatch result is claimed or fabricated.

## First real experimental milestone

Run 16 I2V examples with the Causal Forcing frame-wise checkpoint, produce `probe_heatmap_color.png`, select a real high-signal `(call, block)`, and perform the first `color_000` donor-to-target patch with wrong-block/wrong-call controls.
