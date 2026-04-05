# Redundant under Normal Conditions, Causally Necessary under Corruption: Layer 12 Attention in Qwen1.5-1.8B

**Background note:** I have 17 years in tech, the last 8 in data science and ML
products. This is my first serious mech interp investigation. The findings may be
wrong, the framing may be naive. Sharing publicly because I learn better that way.
Feedback genuinely welcome.

---

## The Core Finding

Layer 12 attention in Qwen1.5-1.8B is not what makes the clean run correct — it
is the only thing that can save the corrupted run. The asymmetry between patching
and ablation reveals that this computation is redundant under normal conditions but
causally necessary when the input is corrupted.

---

## Task

I needed a minimal task where:
- The correct answer is a single token
- One small input change flips the answer
- The underlying rule is interpretable

**Set-completion task:**

**Clean:**
`Man has four fruits, Peach, Orange, Grape and Mango - 1 Peach 1 Orange 1 Grape and 1`
→ expected: `Mango`

**Corrupted:**
`Man has four fruits, Peach, Orange, Grape and Mango - 1 Peach 1 Mango 1 Grape and 1`
→ expected: `Orange`

Same four fruits in both prompts. Only the third listed fruit is swapped. This
changes which fruit is missing, and therefore changes the correct completion. The
model must track a latent variable: which element of the declared set has not yet
appeared.

All key tokens (`Mango`, `Orange`, `Peach`, `Grape`) verified as single tokens
before running experiments.

---

## Method

**Model:** Qwen1.5-1.8B-Chat via TransformerLens. 24 layers, 16 heads, d_model=2048. Running locally on Apple M3.

**Metric:**

```
logit_diff         = logit(clean_answer) - logit(corrupted_answer)
baseline_clean     = +9.35
baseline_corrupted = -10.72
full_range         = 20.07
```

**Recovery score:**

```
recovery = (patched_diff - baseline_corrupted) / full_range
```

0% = no improvement over corrupted baseline. 100% = clean behavior fully restored.

**Three experiment types:**
1. Residual stream patching — replace corrupted activations with clean ones at a given (layer, position), measure recovery
2. Component patching — patch attention output vs MLP output, layer by layer
3. Ablation — zero out components on clean run and corrupted run separately, compare damage

---

## Finding 1: Layer 12 is the decisive layer

Residual stream patching at the final token position across all 24 layers:

![Layer × component recovery table](https://raw.githubusercontent.com/metarun/images/main/Layer_%C3%97_component_recovery_table.png)


Recovery is near zero through Layer 11. Layer 12 produces a sharp jump to 97.1%
total — 79.9% from attention alone, 17.2% from MLP. Layers 13 and 14 show
secondary contributions (~48% and ~71% at the layer level) but these appear to be
downstream effects — they benefit from Layer 12's computation rather than doing
independent work.

**Layer 12 attention accounts for 79.9% of recovery. It is where the set-completion computation happens.**

**Note on Layer 0 MLP:** The patching table shows ~99.5% recovery for Layer 0 MLP,
which is anomalously high for the first layer. The most plausible explanation is
that Layer 0 MLP encodes low-level token identity features that are nearly
prompt-invariant — patching them in neither helps nor disrupts the corrupted run,
so the logit difference barely changes and the recovery score is spuriously high.
This warrants a sanity check: patching Layer 0 MLP from a semantically unrelated
prompt should produce similar "recovery," confirming the signal is noise.

---

## Finding 2: The computation is compositional - superadditive, no single hero head

Head sweep at Layer 12:

![Head patching heatmap](Head%20patching%20heatmap.png)

Individual head recoveries at Layer 12:
- Head 4: 17.2%
- Head 9: 17.2%
- Heads 4+9 together: 74.8%
- Full layer: 79.9%


No single head dominates. The 74.8% from Heads 4+9 together versus 17.2%
individually shows strong **superadditivity** — the joint contribution (74.8%) far
exceeds the sum of individual contributions (34.4%). These heads are doing something
together that neither does alone, consistent with sequential composition: one head's
output becomes meaningful input for the other.

---

## Finding 3: A structure-tracker and a content-tracker — functional dissociation in Heads 4 and 9

Attention pattern inspection from the final token position reveals that Heads 4 and
9 are not doing the same computation. They implement a clean functional dissociation.

**Head 4 — content-tracker (list identity):**

| Token | Attention weight |
| --- | --- |
| Man | 27% |
| Grape / pos 21 | 25% |
| Mango / pos 18 | 19% |
| Peach / pos 15 | 6% |

![Head 4 attention weights](Head4_attention_weights.png)
Head 4 distributes attention across the fruit token positions — the actual named
items in the list. It is doing set-membership work: gathering evidence about which
fruits have appeared.

**Head 9 — structure-tracker (counting skeleton):**

| Token | Attention weight |
| --- | --- |
| Man | 56% |
| digit `1` at pos 14, 17, 20 | ~39% combined |
| Fruit tokens | ~0% |

![Head 9 attention weights](Head9_attention_weights.png)
Head 9 completely ignores fruit identity. It attends to the structural skeleton of
the sequence — the subject noun and the repeated count digits.

**Interpretation:** Head 9 establishes positional and structural context (how many
items, where the counted slots are) that Head 4 then uses to resolve which content
item is missing. This is consistent with the superadditivity result: Head 4's
content-tracking becomes interpretable only given the structural frame Head 9 has
established. Neither head is sufficient alone.

---

## Finding 4: Redundant under normal conditions, causally necessary under corruption

This is the most interesting result.

![Ablation comparison table](Ablation%20comparison.png)

<!-- | Intervention | Run | Damage | % of baseline |
| --- | --- | --- | --- |
| Ablate Heads 4+9 | Clean | 0.41 | 4.4% |
| Ablate full L12 attention | Clean | 0.70 | 7.4% |
| Ablate full L12 attention | Corrupted | 9.27 | 99.1% | -->

On the clean run, Layer 12 attention is nearly expendable. Remove all 15 heads —
the model still produces the correct answer at 92.6% baseline strength. Other
computational paths compensate.

On the corrupted run, the same ablation causes near-total collapse. The logit
difference drops to +0.08, essentially random. **Layer 12 attention is the only
surviving path that carries the correct signal in the corrupted run. Every other
path is already propagating the corrupted answer.**

**Reconciling patching and ablation:** These experiments ask different questions.
- Patching measures sufficiency: can this component carry the clean signal into a corrupted context?
- Ablation measures necessity: is this component required when the clean run is intact?

Layer 12 attention is sufficient (high patching recovery) but not necessary in the
clean run (low ablation damage). In the corrupted run it becomes necessary because
it is the only component not yet contaminated.

The computation is redundantly distributed on the clean run. Corruption removes
that redundancy and exposes which component is load-bearing.

---

## Finding 5: Head 4 writes a small, correctly-directed fruit signal. Head 9 writes none.

Direct logit attribution — projecting each head's output activations through the
unembedding matrix W_U — reveals what each head actually contributes to the final
prediction.

**Head 4 fruit-specific contributions:**

| Fruit | Clean run (answer = Mango) | Corrupted run (answer = Orange) |
| --- | --- | --- |
| Mango | **+0.087** | +0.062 |
| Orange | -0.067 | **+0.141** |
| Peach | +0.063 | +0.117 |
| Grape | -0.167 | +0.062 |

The correct answer fruit is most promoted in both cases. This is not a copy
operation — the magnitudes are small relative to the baseline logit difference of
9.35 — but the signal is correctly discriminative. Head 4 writes a weak, accurate
vote toward the missing fruit. Downstream layers amplify it.

The top promoted tokens in vocabulary space are noise (random Chinese characters,
punctuation). Head 4's contribution is small across most of vocabulary space but
correctly directed in the fruit subspace.

**Head 9 fruit contributions (corrupted prompt):** Mango −0.18, Orange −0.13,
Grape −0.26, Peach +0.04. Head 9 suppresses fruit tokens or is indifferent to them.
It writes no fruit-directed signal. This confirms it is not a content head — its
contribution is structural context, not answer selection.

**Note:** W_V @ W_O SVD against the token embedding matrix W_E showed fruit tokens
at ranks 82–140,000. This is expected — by Layer 12 the residual stream is in a
transformed representation space that does not align with raw token embeddings.
Direct logit attribution through W_U is the correct analysis for output-side questions.

**Mechanistic picture, revised:**
- Head 9 establishes structural context by attending to the counting skeleton
- Head 4 reads that context alongside fruit positions, writes a weak discriminative fruit signal
- Layers 13 and 14 amplify the signal into the strong logit difference observed at the final position

---

## Open Questions

**What amplifies Head 4's signal?** Head 4 writes a correctly-directed but weak
fruit contribution (max +0.14). The final logit difference is 9.35. Layers 13 and
14 account for the amplification — identifying what they are doing to Head 4's
residual stream write is the next mechanical question.

**Is the redundancy general or specific to this task?** The clean-run robustness
might reflect that set-completion is trivially easy for this model on this prompt,
not that set-completion circuits are generally redundant.

**Relation to IOI?** The multi-head compositional structure — one head tracking
structure, another tracking content — looks superficially similar to the
name-mover / S-inhibition head division in GPT-2's indirect object identification
circuit. I have not done the analysis to compare mechanisms.

---

## Code

Full pipeline (TransformerLens): residual stream patching, component patching, head
sweeps, ablation, attention visualization, direct logit attribution. Happy to share
the notebook on request.

---

*Feedback especially welcome on: whether the ablation interpretation is correct,
whether the clean-run redundancy is an interesting finding or an expected baseline
result, the Layer 0 MLP anomaly explanation, and whether the structure/content
dissociation framing for Heads 4 and 9 is mechanistically justified or
over-interpreted from attention weights alone.*
