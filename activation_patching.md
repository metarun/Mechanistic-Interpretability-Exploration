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

![Head patching heatmap](https://raw.githubusercontent.com/metarun/images/main/Head_patching_heatmap.png)

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

![Head4 attention weights](https://raw.githubusercontent.com/metarun/images/main/Head4_attention_weights.png)
Head 4 distributes attention across the fruit token positions — the actual named
items in the list. It is doing set-membership work: gathering evidence about which
fruits have appeared.

**Head 9 — structure-tracker (counting skeleton):**

| Token | Attention weight |
| --- | --- |
| Man | 56% |
| digit `1` at pos 14, 17, 20 | ~39% combined |
| Fruit tokens | ~0% |

![Head9 attention weights](https://raw.githubusercontent.com/metarun/images/main/Head9_attention_weights.png)
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

![Ablation comparison](https://raw.githubusercontent.com/metarun/images/main/Ablation_comparison.png)

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

## Finding 6: Computation and output are dissociated — Layer 12 computes, Layers 17/19/23 write

Direct logit attribution on both clean and corrupted runs reveals a two-stage circuit.

**Stage 1 — Computation (Layers 0–13):**
Layer DLA scores are nearly identical between clean and corrupted runs through
Layer 13. The early circuit is prompt-invariant — it is gathering structure and
content information without yet committing to an answer. Layer 12 (+1.08 clean,
+1.24 corrupted) and Layer 13 (+1.33 clean, +1.14 corrupted) show up symmetrically
in both runs. This confirms the patching finding: Layer 12 computes the
set-completion representation but does not directly vote for a token.

**Stage 2 — Output (Layers 16–23):**
Layer 16 is the first sign flip: clean +1.25, corrupted -1.59. Layer 17 is the
decisive split: clean +13.12, corrupted -13.58 — nearly perfectly mirrored.

Head DLA at the three dominant output layers:

| Layer | Head | Clean DLA | Note |
|-------|------|-----------|------|
| 17    | 5    | +6.57     | dominant output writer |
| 17    | 15   | +5.71     | dominant output writer |
| 17    | 9    | +1.12     | small secondary |
| 19    | 13   | +10.69    | single head, all others net negative |
| 23    | 15   | +2.03     | positive |
| 23    | 14   | -1.66     | actively suppressing correct answer |

Each output layer has a different structure: L17 is a two-head collaboration,
L19 is a single dominant head, L23 has internal competition between heads.

**On the corrupted run, layers 17–22 cascade uniformly negative.** The same
heads that write the correct answer on the clean run write the wrong answer
on the corrupted run with equal force. They are not biased toward any token —
they faithfully amplify whatever the Layer 12 computation produced.

**The amplification question is partially answered.** Head 4 at Layer 12 writes
a weak but correctly-directed fruit signal (~+0.14). L17 Heads 5 and 15 amplify
this into +13 in token space. The mechanism of that amplification — what these
heads are reading from the residual stream and why — remains the open question.
![Layer DLA](https://raw.githubusercontent.com/metarun/images/main/layer_dla.png)

---
## Finding 7: L17 Heads 5 and 15 implement a targeted copy mechanism

Attention pattern inspection at Layer 17 on both clean and corrupted runs
reveals the output mechanism directly.

![Clean Attention Pattern](https://raw.githubusercontent.com/metarun/images/main/clean_attention_pattern.png)
**Clean run (answer = Mango):**
- Head 15 final token attends strongly to position 11 — Mango in the declaration
- Head 5 final token attends to positions 9–11 — Grape/Mango area in the declaration

![Corrupt Attention Pattern](https://raw.githubusercontent.com/metarun/images/main/corrupt_Attention_pattern.png)
**Corrupted run (answer = Orange):**
- Head 15 final token attends strongly to position 7 — Orange in the declaration
- Head 5 final token attends to positions 7–9 — Orange area in the declaration

The declaration section is identical across both prompts. The fruit list never
changes position. Yet the attention of both heads shifts to a different declaration
token depending on which fruit is absent from the counted list.

These heads are not doing abstract computation. Their attention pattern encodes
the set-completion answer directly — they attend to whichever declared fruit has
not appeared in the counting sequence, then copy its token representation into
logit space. This is why their DLA scores are large (+6.57 and +5.71): a targeted
copy of a concrete token produces a strong, clean logit signal.

**The circuit is now fully described:**

| Component | Role |
|-----------|------|
| L12 H9 | Structure-tracker — attends to Man and count digits, establishes counting skeleton |
| L12 H4 | Content-tracker — attends to counted fruit positions, tracks what has appeared |
| L17 H5+H15 | Copy mechanism — attend to missing fruit in declaration, write it to output |

L12 computes the absence. L17 resolves it to a token. The two stages are
mechanistically distinct and operate on different parts of the prompt.

**What remains open:** How does L17 know which declaration token to attend to?
Its attention pattern already encodes the answer — meaning it is reading
something from the residual stream at that point that tells it where to look.
That intermediate representation, written by L12 and read by L17, is the
remaining unknown in this circuit.

---


## Open Questions

**What does the residual stream at L17 encode that tells H5+H15 where to attend?**
Head 4 at Layer 12 writes a weak but correctly-directed fruit signal (~+0.14).
Layer 17 Heads 5 and 15 implement a copy mechanism — they attend to the missing
fruit in the declaration and write it to output. But their attention pattern
already encodes the answer, meaning they are reading something from the residual
stream that tells them which declaration token to attend to. What that intermediate
representation looks like — and how L12's weak signal gets transformed into a
precise attention target by L17 — is the remaining unknown in this circuit.

**Is the redundancy general or specific to this task?**
On the clean run, ablating all of Layer 12 attention causes only 7.4% damage.
The model has redundant paths to the correct answer when the input is intact.
Whether this reflects a general property of how set-completion circuits are
organized, or simply that this particular prompt is easy enough that many
independent paths converge on the same answer, remains untested. Running the
same ablation on harder or more ambiguous prompts would distinguish these.

**How does this circuit compare to IOI in GPT-2?**
The structure is now concrete enough to compare directly. This circuit has:
a structure-tracker (L12 H9), a content-tracker (L12 H4), and copy heads
(L17 H5+H15) that attend to the correct token and write it to output. In GPT-2's
indirect object identification circuit, name-mover heads perform the analogous
copy operation — attending to the correct name and writing it to the output
position. The parallel is specific enough to be worth a direct mechanistic
comparison: do L17 H5+H15 implement the same QK and OV circuit structure as
GPT-2 name-mover heads? That comparison would either strengthen the claim that
copy heads are a general transformer primitive or reveal important differences
in how the two models solve structurally similar tasks.

---

## Code

Full pipeline (TransformerLens): residual stream patching, component patching, head
sweeps, ablation, attention visualization, direct logit attribution. 
[View Notebook (nbviewer)](https://nbviewer.org/github/metarun/Mechanistic-Interpretability-Exploration/blob/main/Activation_Patching_.ipynb)

---

*Feedback especially welcome on: whether the ablation interpretation is correct,
whether the clean-run redundancy is an interesting finding or an expected baseline
result, the Layer 0 MLP anomaly explanation, and whether the structure/content
dissociation framing for Heads 4 and 9 is mechanistically justified or
over-interpreted from attention weights alone.*
