# Activation Patching, Ablation, and Direct Logit Attribution
## A Conceptual Guide with Worked Example

*Written as part of a mechanistic interpretability investigation into set-completion circuits in transformer language models. Full experiment: [Activation_Patching_.ipynb](https://github.com/metarun/Mechanistic-Interpretability-Exploration/blob/main/Activation_Patching_.ipynb)*

---

## What Problem Are We Solving?

A transformer has dozens of layers, hundreds of attention heads, and billions of parameters. When it produces the right answer, you cannot simply look at the weights and understand why. The computation is too entangled.

Mechanistic interpretability asks: **which specific internal components are causally responsible for a model's output?** Not correlated — causally responsible. If you remove or replace that component, the answer changes.

The three tools below — activation patching, ablation, and direct logit attribution — each answer a slightly different version of that question.

---

## The Core Setup: A Minimal Task

To study internal computation, you need a task where:
- The correct answer is a single token
- One small input change flips the answer
- The underlying rule is interpretable

**Clean prompt:**
`A basket has three fruits: Apple, Mango, Orange. I counted one Apple, one Mango, and one`
→ correct answer: `Orange`

**Corrupted prompt:**
`A basket has three fruits: Apple, Mango, Orange. I counted one Apple, one Orange, and one`
→ correct answer: `Mango`

Same fruits declared. One word swapped in the counted list. The correct completion flips because the missing fruit changes.

This gives us a clean controlled pair: we know exactly what changed in the input, and we know exactly what should change in the output. Any difference in internal activations between these two prompts can be linked to the computation of "which fruit is missing."

---

## The Measurement: Recovery Score

We run the model on both prompts and measure the logit difference between the two answer tokens at the final position:

```
logit_diff = logit(correct_answer) - logit(wrong_answer)
```

This gives us two baselines:
- `baseline_clean` : large positive number — model strongly prefers the right answer
- `baseline_corrupted` : large negative number — model strongly prefers the wrong answer

When we intervene, we measure how much of the clean preference we restore:

```
recovery = (patched_logit_diff - baseline_corrupted) / (baseline_clean - baseline_corrupted)
```

- `0%` — intervention did nothing, model still answers as if corrupted
- `100%` — intervention fully restored clean behavior
- `negative` — intervention made things worse
- `>100%` — intervention overshot the clean baseline

---

## Tool 1: Activation Patching

**The idea:** Run the clean prompt and save all internal activations into a cache. Then run the corrupted prompt, but at one specific layer and position, swap the cached clean activation back in. Measure how much recovery this produces.

This is a surgical causal intervention. You are asking: *if I gave the model the clean representation at this specific point, would it answer correctly?*

### Level 1 — Residual Stream Patching

The residual stream is the single vector that accumulates information as it passes through the model. Every layer reads from it and writes to it. By layer 12, it contains everything layers 0–11 have contributed.

When you patch the residual stream at layer L, you replace the entire accumulated state at that point:

```
corrupted run reaches Layer L
↓
swap: resid_stream[layer=L, position=-1] ← clean_cache[layer=L, position=-1]
↓
layers L+1 through N now receive a clean, coherent signal
```

**Why recovery tends to be high:** You gave downstream layers everything they need in one complete swap. They react normally to a fully clean residual stream.

**What this tells you:** Which layer is the critical decision point — the layer where clean information is sufficient to restore the correct answer.

---

### Level 2 — Component Patching (Attention vs MLP)

Inside each layer, the residual stream receives two separate additions:

```
resid = resid_before_layer
resid = resid + attention_output     ← attention writes here
resid = resid + mlp_output           ← MLP writes here
resid_after_layer = resid
```

These are genuinely separate writes. You can patch each one independently.

**Attention patching:** Replace only the attention output at layer L with the clean cached version. The MLP output at that layer is still from the corrupted run. Downstream layers receive a mixed signal.

**MLP patching:** Replace only the MLP output. Attention output is still corrupted.

**What this tells you:** Whether the critical computation at that layer happens in the attention block or the MLP block.

---

### Level 3 — Head Patching

Within the attention block, each head produces its own output. After multiplying by W_O, these outputs are combined. You can patch individual heads by replacing just their contribution to the attention output.

**What this tells you:** Which specific heads carry the signal — whether it is one dominant head or a distributed computation across many.

---

### Why Component Recovery Scores Do Not Add Up

This is subtle and important.

Suppose at layer 12 you find:
- Attention patching: 80% recovery
- MLP patching: 17% recovery

You might expect residual stream patching to give 97%. Sometimes it does, but this is not guaranteed and when it holds it is partly coincidental.

The reason: each component patch is measured in a different world.

When you measure attention patching recovery, the MLP is still corrupted in that forward pass. When you measure MLP patching recovery, the attention is still corrupted. These are two separate experiments. Downstream layers react differently in each.

**A concrete case where they clearly do not add up:**

Suppose attention at layer 12 writes a signal, and MLP at layer 12 reads that signal and amplifies it tenfold:

- Patch attention alone → MLP gets clean attention signal → amplifies it → high recovery
- Patch MLP alone → MLP gets corrupted attention signal → amplifies the wrong thing → low recovery or negative
- Both patches together → would be different from either alone

The components interact. Adding independent measurements assumes independence that does not exist.

**Residual stream patching sidesteps this entirely** — it swaps the complete state, so downstream components always react to a coherent signal.

---

## Tool 2: Ablation

**The idea:** Instead of injecting clean activations into a corrupted run, you destroy a component's output entirely — set it to zero — and measure how much the model's performance drops.

Patching asks: *is this component sufficient to carry the signal?*
Ablation asks: *is this component necessary when everything else is working normally?*

These are different questions and they can give opposite answers for the same component.

```
ablation_damage = baseline_clean - ablated_logit_diff
```

High damage means the component was load-bearing. Low damage means the model compensated through other paths.

### The Asymmetry That Makes Ablation Interesting

Run the same ablation on the clean prompt and on the corrupted prompt separately.

**Clean run ablation:** If the model has redundant paths to the correct answer, removing one component causes little damage. Other components compensate.

**Corrupted run ablation:** If that component is the *only* surviving path carrying the correct signal in the corrupted run — because every other path is already propagating the corrupted answer — then removing it causes total collapse.

This asymmetry reveals something that patching cannot: a component can be **redundant under normal conditions but causally necessary under corruption.** Patching identifies it as sufficient. Clean-run ablation says it is not necessary. Corrupted-run ablation reveals it is the last surviving correct path.

---

## Tool 3: Direct Logit Attribution (DLA)

**The idea:** Project each component's output directly into token space using the unembedding matrix W_U, without running another forward pass.

Patching and ablation both run a modified forward pass and observe the final output. This means downstream layers react to your intervention and their reactions are mixed into the result.

DLA reads cached activations and multiplies by W_U directly:

```
component_vote = component_output[position=-1] @ W_U
```

This gives a score for every token in the vocabulary — how much that component pushed each token up or down — with no downstream interaction at all.

**Why the math works:** The residual stream is a sum of contributions:

```
resid_final = embed + layer_0_delta + layer_1_delta + ... + layer_N_delta
```

W_U is a linear operation. Linear applied to a sum equals sum of linears:

```
logits = resid_final @ W_U
       = (embed @ W_U) + (layer_0_delta @ W_U) + ... + (layer_N_delta @ W_U)
```

Each term is that component's direct contribution to the final logit. **These add up exactly, by construction.** No interactions, no approximations.

**What this tells you:** What each layer or head is actually writing in token space. Not what it causes — what it writes. You can say "Head 4 directly promoted Orange by +2.1 logits and suppressed Mango by -0.8 logits." That is a mechanistic claim, not just a causal attribution.

---

## The Four Tools Together

| Tool | What you do | What it tells you | Scores add up? |
|---|---|---|---|
| Residual stream patching | Inject clean state at layer L | Which layer is the decision point | N/A |
| Component patching | Inject clean attention or MLP output | Attention vs MLP split | Not guaranteed |
| Head patching | Inject clean output of one head | Which heads carry the signal | Not guaranteed |
| Ablation | Zero out a component, measure damage | Is it necessary? Redundant or load-bearing? | N/A |
| Direct logit attribution | Project output through W_U, no forward pass | What is each component writing in token space | Yes — exactly |

**The natural order of investigation:**

1. Residual stream patching → find the critical layer
2. Component patching → is it attention or MLP?
3. Head patching → which heads?
4. Ablation (clean + corrupted) → are they redundant or necessary?
5. DLA → what are they actually writing?

---

## What We Found Running This on Qwen1.5-1.8B

Full code and results: [Activation_Patching_.ipynb](https://github.com/metarun/Mechanistic-Interpretability-Exploration/blob/main/Activation_Patching_.ipynb)

**Task:** Set-completion with four fruits across Qwen1.5-1.8B (24 layers, 16 heads).

**Residual stream patching** identified Layer 12 as the decisive layer — recovery jumps from near zero at Layer 11 to 97.1% at Layer 12.

**Component patching** at Layer 12 split as: attention 79.9%, MLP 17.2%.

**Head patching** at Layer 12 found that no single head dominates. Head 4 and Head 9 recover 17.2% each individually, but 74.8% together — strongly superadditive. The gap between their sum (34.4%) and their joint contribution (74.8%) shows that these heads are compositionally dependent. Attention pattern inspection gave an interpretation:

- Head 4 attends to fruit positions across the sequence — a list-tracking head
- Head 9 attends to structural tokens (the subject noun and count digits) while giving near-zero weight to fruit tokens — a structural context head

**Ablation** revealed the most interesting result. Zeroing all 16 heads at Layer 12 on the clean run causes only 7.4% damage — the model compensates through other paths. Zeroing the same heads on the corrupted run causes 99.1% damage — near-total collapse. Layer 12 attention is redundant on the clean run but is the only surviving correct path on the corrupted run. Every other computational path is already propagating the corrupted answer.

**Direct logit attribution** confirmed the functional dissociation: Head 4's output projects through W_U to directly promote the correct fruit token. Head 9's output writes no fruit signal at all — it is supporting the structural setup that Head 4 requires.

The finding is not that Layer 12 is special in isolation. It is that the set-completion computation is redundantly distributed on clean inputs — multiple paths can each produce the answer — and corruption removes that redundancy, exposing which component is actually load-bearing.
