# Direct Logit Attribution

## The One Example We Use Throughout

**Clean prompt:**
`A basket has three fruits: Apple, Mango, Orange. I counted one Apple, one Mango, and one`
→ correct answer: `Orange`

**Corrupted prompt:**
`A basket has three fruits: Apple, Mango, Orange. I counted one Apple, one Orange, and one`
→ correct answer: `Mango`

One word swapped. The missing fruit changes. The correct answer flips.

---

## What Question DLA Answers

Activation patching tells you *where* a computation happens — which layer, which head carries the signal.

Direct Logit Attribution (DLA) tells you *what* each component is actually writing — in token space, in units you can read.

Concretely: after running patching, you might know that Layer 12 Head 4 is important. DLA lets you ask — is Head 4 directly voting for `Orange`? By how much? Is it suppressing `Mango`? Or is it writing something structural that has no direct token meaning at all?

---

## Why The Math Allows This

The residual stream is a sum. Every layer takes the existing residual stream and adds its contribution:

```
resid_final = embed
            + layer_0_delta
            + layer_1_delta
            + ...
            + layer_N_delta
```

The unembedding matrix W_U converts the final residual stream into token logits:

```
logits = resid_final @ W_U
```

Because W_U is a linear operation, and the residual stream is a sum, you can distribute W_U across every term:

```
logits = (embed @ W_U)
       + (layer_0_delta @ W_U)
       + (layer_1_delta @ W_U)
       + ...
       + (layer_N_delta @ W_U)
```

Each term is that component's direct contribution to the final logits. **These add up exactly** — not approximately, exactly. No forward pass needed. No downstream interactions. Pure linear algebra applied to cached values.

---

## The Metric

For each component, we project its output through W_U and read off two numbers:

```
component_vote = component_output[position=-1] @ W_U

DLA_score = component_vote[orange_token_id] - component_vote[mango_token_id]
```

Positive score — this component is pushing toward `Orange` (the correct clean answer).
Negative score — this component is pushing toward `Mango`.
Near zero — this component is writing something with no direct bearing on either answer token.

---

## Three Levels of DLA

### Level 1 — Layer Level

**What you compute:** The delta each layer wrote to the residual stream.

```
layer_delta = resid_after_layer_L - resid_before_layer_L
```

For Layer 0, `resid_before` is the raw token embedding. For every other layer, it is the output of the previous layer.

**What you project:**

```
layer_delta[position=-1] @ W_U  →  score for every token in vocabulary
```

You slice position -1 because the model predicts the next token by reading only the final position. All information from earlier positions has already been gathered into position -1 by the attention heads.

**What you get:** A score per layer showing how much it directly pushed the final logit toward `Orange` versus `Mango`. Plot all layers as a bar chart — you see at a glance which layers contribute signal and which are neutral.

---

### Level 2 — Component Level (Attention vs MLP)

Within each layer, attention and MLP write to the residual stream separately:

```
resid = resid_before_layer
resid = resid + attention_output
resid = resid + mlp_output
```

TransformerLens caches each write directly — `hook_attn_out` and `hook_mlp_out`. No delta subtraction needed. You read the write directly.

**Attention DLA:**

```
cache["blocks.12.hook_attn_out"][position=-1] @ W_U
→ DLA_score = orange_logit - mango_logit
```

**MLP DLA:**

```
cache["blocks.12.hook_mlp_out"][position=-1] @ W_U
→ DLA_score = orange_logit - mango_logit
```

**Sanity check:** `attention_DLA + mlp_DLA = layer_DLA` exactly. If not, something went wrong.

**What this tells you:** Whether attention or MLP is the direct driver of the correct answer at that layer. A layer can show high DLA overall while being entirely driven by attention with MLP contributing nothing — or the reverse.

---

### Level 3 — Head Level

Within attention, each head produces its own output. After W_O is applied, the total attention output is the sum of all head contributions.

TransformerLens caches the pre-W_O head outputs at `hook_z`. To get each head's individual write to the residual stream:

```
head_output = cache["blocks.12.attn.hook_z"][position=-1, head=4] @ W_O[layer=12, head=4]
```

This is the vector Head 4 actually wrote to the residual stream at position -1.

**Head DLA:**

```
head_output @ W_U
→ DLA_score = orange_logit - mango_logit
```

**What this tells you:** Whether a specific head is directly voting for the correct answer — and by how much.

In our example:
- A list-tracking head that gathered evidence across fruit positions would show a positive DLA score — directly promoting `Orange`
- A structural head that attended to count digits and subject nouns would show a near-zero DLA score — it wrote nothing fruit-related into the residual stream

This is the difference between a pattern observation and a mechanistic claim. Seeing that Head 4 attends to fruit positions is a pattern. Confirming that Head 4's output directly promotes `Orange` by +2.1 logits is a mechanism.

---

## What DLA Cannot Tell You

DLA measures direct contributions only. A component with a near-zero DLA score is not necessarily unimportant — it may be doing essential work that another component reads and amplifies downstream.

If Head 9 writes structural context into the residual stream that Head 4 needs to identify the missing fruit, Head 9's contribution never appears in token space directly. It shows up only through Head 4's output. DLA would score Head 9 near zero and miss its role entirely.

This is why DLA and patching are complementary, not redundant:

- Patching measures what each component *causes* — including indirect effects through downstream layers
- DLA measures what each component *writes* — direct token-space contributions only

You need both to distinguish "this head directly promotes the answer" from "this head enables another head to promote the answer."

---

## Summary

| Level | What you project | What you learn |
|---|---|---|
| Layer | `resid_delta[pos=-1] @ W_U` | Which layers directly push the correct answer |
| Component | `attn_out[pos=-1] @ W_U` or `mlp_out[pos=-1] @ W_U` | Attention vs MLP split within a layer |
| Head | `(hook_z[pos=-1, head=h] @ W_O[h]) @ W_U` | Which heads directly vote for the answer token |

All three levels decompose the same final logit. All three add up exactly by construction. The only thing that changes is how finely you slice the contribution.
