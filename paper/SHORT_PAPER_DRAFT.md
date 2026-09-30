# StatePatch: Causal Discovery and Repair of World-State Representations in Autoregressive Video Diffusion Models

## Abstract

Autoregressive video generators can produce visually convincing rollouts while losing persistent facts about the generated world. We study whether persistent state is represented inside the hidden activations of an autoregressive video diffusion model and whether those representations are causally used during future generation. StatePatch uses controlled counterfactual pairs that differ in one state variable, layer/time probing to localize decodable state, and matched activation patching to test whether transferring an internal representation changes the later generated state. We further define intervention specificity, which measures target-state change relative to collateral visual change. **Results are intentionally left blank until experiments are run; no unexecuted result should be inserted here.**

## 1. Introduction

Modern video generators increasingly operate autoregressively, maintaining a history through transformer activations and key-value caches. Long rollouts expose a gap between visual realism and persistent world-state consistency: an object can change identity, position, count, or state after occlusion or camera motion. Existing approaches often add memory modules, retrieval mechanisms, additional supervision, or explicit state representations. We instead ask a mechanistic question: what persistent state is already represented inside a pretrained generator, and does the generator causally rely on it?

Our core experiment uses paired worlds that differ in exactly one persistent state variable. We first measure where the state can be decoded from hidden activations, then intervene at matched internal locations by replacing or interpolating the target activation with its counterfactual donor. A causal state representation should satisfy two requirements: intervention changes the later target state, and unrelated visual content changes substantially less.

### Contributions

1. A controlled paired benchmark for persistent state variables in video continuation.
2. A layer-by-time probing framework for autoregressive video diffusion transformers.
3. A matched counterfactual activation-patching framework for testing causal use of world-state representations.
4. An intervention-specificity analysis separating targeted state changes from general video corruption.

## 2. Method

Let `H_(l,t)` denote hidden tokens at transformer block `l` during generator call `t`. For each state variable `s`, we create matched trajectories A and B that differ only in `s`. We estimate decodability with a linear probe on token-pooled hidden states. Decodability is treated only as localization evidence and not as causality.

For causal validation, a donor activation from trajectory A is patched into the corresponding block/call in trajectory B. The simplest intervention is

`H' = H_B + alpha (H_A - H_B)`.

A low-rank version projects the counterfactual difference into a learned state subspace before applying the intervention. We sweep intervention strength and use wrong-block, wrong-call, wrong-pair, and random-subspace controls.

## 3. Benchmark

StatePatchBench contains paired initial visual states covering color, position, count, and shape. The continuation instruction does not state the hidden value; therefore, in a valid image/video-conditioned setup, later knowledge of the state must originate from visual history rather than the text prompt.

## 4. Experiments

### 4.1 Probe localization

**TODO after execution:** report balanced accuracy by transformer block and generator call. Include a heatmap.

### 4.2 Counterfactual patching

**TODO after execution:** report intervention success for target block/call and matched controls.

### 4.3 Specificity

**TODO after execution:** quantify target-state change relative to collateral perceptual/video change.

### 4.4 Generalization

**TODO after execution:** repeat across state families and unseen pair instances.

## 5. Limitations

A major threat is text leakage. Text-to-video prompt pairs in which state is named are not evidence of memory because cross-attention can reintroduce the state at every generation step. The primary causal experiment therefore requires visual initialization/continuation with a neutral prompt. Another limitation is architectural specificity: a result in Self-Forcing does not establish that all video world models represent state in the same way.

## 6. Results checklist

Before claiming a causal state mechanism, require all of the following:

- state is measured after it is absent from direct visual input;
- prompt does not name the state value;
- same seed/noise is used for paired trajectories where possible;
- target intervention beats wrong-block and wrong-call controls;
- intervention alters the intended state more than unrelated content;
- effect replicates across multiple pairs/seeds;
- no result is inferred from a single cherry-picked video.
