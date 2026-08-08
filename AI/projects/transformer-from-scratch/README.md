# Transformer From Scratch

Build Transformer blocks in PyTorch from low-level pieces.

Goal: know the tensor shapes, learned weights, and fixed buffers.

## Embedding Block

The embedding block maps token IDs to vectors.

For vocabulary size $V$ and model dimension $d_{\text{model}}$:

$$
W_E \in \mathbb{R}^{V \times d_{\text{model}}}
$$

For token ID $t$, return row $W_E[t]$.

Input shape:

$$
(\text{batch}, \text{seq\_len})
$$

Output shape:

$$
(\text{batch}, \text{seq\_len}, d_{\text{model}})
$$

Weights are initialized from a normal distribution.

### Padding Index

Use `<PAD>` when sequences in a batch have different lengths.

Example:

```text
It is a cat        -> It is a cat <PAD>
It is a yellow cat -> It is a yellow cat
```

`<PAD>` is not real content, so it should not learn meaning. If `padding_idx` is
provided:

1. The embedding vector at `padding_idx` is initialized to zero.
2. The gradient for that row is forced to zero during backpropagation.
3. The optimizer step does not update that row.

Conceptually:

```text
compute gradients
-> hook clears gradient at padding_idx
-> optimizer updates all non-padding rows
```

Use `register_hook` to clear the padding row gradient before the optimizer step:

$$
\nabla W_E[\text{padding\_idx}] = 0
$$

This keeps the padding vector zero.

## Positional Encoding Block

Self-attention alone does not know order.

Without position information, these look the same:

```text
Hello world
```

or:

```text
world Hello
```

Token embeddings say what the token is. Positional encodings say where it is.

The original Transformer uses fixed sinusoidal positional encodings:

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

$$
PE(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

Terms:

- $pos$ is the token position in the sequence.
- $i$ is the dimension pair index.
- $d_{\text{model}}$ is the embedding dimension.
- even dimensions use sine.
- odd dimensions use cosine.

Same shape as token embeddings:

$$
(\text{batch}, \text{seq\_len}, d_{\text{model}})
$$

Add token and position:

$$
X_{\text{input}} = X_{\text{token}} + PE
$$

### Why Use Different Frequencies?

The sinusoidal formula can also be written as:

$$
PE(pos, 2i) = \sin(pos \cdot w_i)
$$

where:

$$
w_i = \frac{1}{10000^{2i / d_{\text{model}}}}
$$

$w_i$ means how many radians the wave moves per token step.

Each dimension uses a different $w_i$.

- small $i$ -> large $w_i$ -> changes fast
- large $i$ -> small $w_i$ -> changes slowly

For intuition, one position step is one token step. We imply $v = 1$.

So the model gets both local and long-range position signals.

### Why $10000$?

The value $10000$ controls the wavelength range.

For a wave:

$$
w = \frac{2\pi}{\lambda}
$$

so:

$$
\lambda = \frac{2\pi}{w}
$$

If $w = 1$, then:

$$
\lambda = 2\pi \approx 6.28
$$

One cycle takes about 6 tokens.

For the slowest dimensions, $w$ approaches:

$$
\frac{1}{10000}
$$

so:

$$
\lambda = 2\pi \cdot 10000 \approx 62831
$$

One cycle takes about 62,000 tokens.

So PE mixes fast waves and slow waves. This helps avoid repeated position
patterns inside the context window.

Transformer has d_model = 1024, so this number is large enough to model position of tokens.

### RoPE

Video: https://www.youtube.com/watch?v=o29P0Kpobz0

[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)

RoPE means Rotary Positional Embedding.

#### Why RoPE?

Absolute PE tells the model **where** each token is, but it does not represent
the distance $j-i$ directly. Learned absolute PE stores one trainable vector
for each position, so the table size limits the maximum position. Sinusoidal PE
uses a fixed formula and can generate any position, but the model may still
generalize poorly beyond its training length.

Relative PE tells the model **how far apart** two tokens are. For example, when
position $i$ attends to the previous position $j=i-1$, their relative distance
is always $j-i=-1$, no matter where they appear in the sequence. Relative PE
often adds this information through a bias lookup inside attention. This adds
overhead, can complicate optimized attention kernels, and does not directly
preserve absolute position.

$$
\text{score}(i,j)=\frac{Q_iK_j^\top}{\sqrt{d_k}}+b_{j-i}
$$

Here, $b_{j-i}$ is a learned bias for the relative distance between tokens. For
a given sequence length, these values form a bias matrix $B$ that **depends only
on token positions, not token content**:

$$
\text{Attention scores}=\frac{QK^\top}{\sqrt{d_k}}+B
$$

After training, the bias matrix $B$ is **fixed for the same relative
positions**. Adding it inside attention requires extra work or explicit support
from the fused attention kernel.

#### Core intuition

RoPE groups the dimensions of each $Q$ and $K$ vector into 2D pairs, such as
$[x_{2i},x_{2i+1}]$. At position $p$, it rotates each dimension pair by:

$$
\theta_{p,i}=p \cdot w_i
$$

$w_i$ means how many radians the wave moves per token step. Each rotated vector
therefore **depends on its absolute position**.

#### Why rotation works

For tokens at positions $m$ and $n$, the angle difference is:

$$
\theta_{n,i}-\theta_{m,i}=(n-m)w_i
$$

$R_p$ is the rotation matrix for position $p$. For one 2D pair:

$$
R(\theta_{p,i})=
\begin{bmatrix}
\cos\theta_{p,i} & -\sin\theta_{p,i} \\
\sin\theta_{p,i} & \cos\theta_{p,i}
\end{bmatrix}
$$

The full $R_p$ applies one such rotation to every dimension pair, using a
different $w_i$ for each pair.

In practice, the full matrix $R_p$ **is not built explicitly**. Each pair is
rotated using cached sine and cosine values:

$$
(x_0,x_1)\mapsto
(x_0\cos\theta-x_1\sin\theta,\;
 x_0\sin\theta+x_1\cos\theta)
$$

For $n$ tokens with $d$ dimensions each, RoPE touches every dimension once.
Applying it to both $Q$ and $K$ **costs** $O(2nd)=O(nd)$. In contrast, attention
compares every pair of tokens, so $QK^\top$ **costs** $O(n^2d)$. The RoPE
rotation is therefore small in comparison.

The same difference appears in their attention score:

$$
\left(R_m Q_m\right)^\top\left(R_n K_n\right)
=Q_m^\top R_{n-m}K_n
$$

Therefore, **RoPE uses absolute positions for rotations, while attention scores
depend on relative distance**. After rotating $Q$ and $K$, attention still uses
the usual $QK^\top$ computation, so **RoPE does not require an extra bias matrix
inside the attention kernel**. RoPE is applied to $Q$ and $K$, not $V$, because
$QK^\top$ computes attention scores.

#### Distance effect

Nearby positions have small rotation differences. As distance grows, the
different 2D pairs tend to become less aligned, so positional correlation often
weakens. **This is not a strict rule:** content can still make distant tokens
attend strongly.

#### Why RoPE fits self-attention

In self-attention, $Q$ and $K$ come from the same sequence, so their position
difference $j-i$ is meaningful. RoPE therefore works naturally in both encoder
and decoder self-attention.

For autoregressive decoding, RoPE also works efficiently with the KV cache:
each key is **rotated once before being cached**, while each new query is rotated
at its current position. Cross-attention needs more care because its queries and
keys come from different sequences, so $j-i$ may not be meaningful.

## Multihead Attention Block

# Tokenizers

A tokenizer turns text into token IDs. The embedding block needs IDs, not
strings.

Two bad extremes:

- **Word level**: vocabulary explodes. Any unseen word becomes `<UNK>`.
- **Character level**: tiny vocabulary, but sequences get very long. Attention
  costs $O(n^2)$, so long sequences are expensive.

Subword tokenization sits in the middle. Common words stay one token. Rare words
split into pieces.

## BPE and Byte-level BPE

### BPE

BPE means Byte Pair Encoding. It was a compression algorithm first, then reused
for tokenization.

Training is one loop. Each step below is shown on one corpus, used for the rest
of this section:

```text
low x5
log x3
her x4
per x3
```

Target vocabulary size: 13.

**Step 1. Start with a vocabulary of single characters.**

Split every word into characters. The counts stay attached.

```text
l o w   x5
l o g   x3
h e r   x4
p e r   x3
```

Vocabulary so far is just the distinct characters — 8 tokens:

```text
e g h l o p r w
```

This is the **base vocabulary**. It never shrinks. Every later token is built on
top of it.

Encoding the corpus at this point costs **45 tokens** — every character is one.

**Step 2. Count every adjacent pair.**

A pair is **two adjacent symbols** in the current sequence. A symbol is whatever
sits there right now: a single character at the start, an already merged token
later. Two limits:

- Pairs must be adjacent. In `l o w` the pairs are `l o` and `o w`. Not `l w`.
- Pairs never cross a **pre-token** boundary. Text is chopped into chunks
  before BPE runs, and merges stay inside a chunk. The `w` of `low` and the `h`
  of `her` never pair up.

##### Pre-tokenization is not "split on space"

Easy to get wrong. BPE does not throw spaces away.

The original BPE (`subword-nmt`) does split on whitespace and drops the space,
adding a `</w>` marker so you know where a word ended. The corpus in this
walkthrough is written that way — a list of words with counts — because it keeps
the example readable.

Modern byte-level BPE does not. GPT-2 uses a regex that attaches the space to
the **front of the next chunk**:

```python
r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
```

```text
"the low lower"  ->  ["the", " low", " lower"]
```

The space is *inside* the chunk, so merges run over it and it ends up inside
real tokens. `" low"` trains as one token, written `Ġlow` in the GPT-2 vocab.
That is why `"cat"` and `" cat"` are two different token IDs.

Why chunk at all? Without it, BPE would happily learn junk tokens spanning
`". The"` or `") {"`. The regex also stops merges from crossing letters into
digits into punctuation, which keeps the vocabulary cleaner.

SentencePiece takes a third route: replace every space with a visible `▁` and
run BPE on the raw stream. Same goal, no regex — the space becomes an ordinary
character the model can merge.

Count inside each word, then multiply by that word's frequency. `l o` appears in
`low` and `log`, so it scores $5+3=8$. `e r` appears in `her` and `per`, so it
scores $4+3=7$.

| Pair | Count |
|------|-------|
| `l o` | 8 |
| `e r` | 7 |
| `o w` | 5 |
| `h e` | 4 |
| `o g` | 3 |
| `p e` | 3 |

**Step 3. Merge the most frequent pair into one new token.**

`l o` wins at 8. Replace it everywhere:

```text
lo w   x5
lo g   x3
h e r  x4
p e r  x3
```

`lo` is now a single symbol. The vocabulary grows by exactly one token, from 8
to 9:

```text
e g h l o p r w  +  lo
```

The old `l` and `o` stay in the vocabulary. They are still needed for any word
that never forms `lo`.

**Step 4. Add that merge to the merge list.**

```text
1. l + o -> lo
```

The list is ordered. Rank 1 must always be applied before rank 2, or encoding
gives different tokens than training did.

**Step 5. Repeat until the vocabulary hits the target size.**

Go back to step 2 and recount. Two numbers move in opposite directions every
round, so track both:

> **After round 1** — vocab 9 tokens, corpus 37 tokens.

#### Rounds 2 to 5

The key rule: one merge per round, then **recount everything**. A pair's count
is not stable across rounds.

**Round 2.** Words are `lo w`, `lo g`, `h e r`, `p e r`.

| Pair | Count |
|------|-------|
| `e r` | 7 |
| `lo w` | 5 |
| `h e` | 4 |
| `lo g` | 3 |
| `p e` | 3 |

Two things happened here.

`lo w` appeared with a count of 5. In round 1 it had no count at all — `l o w`
was three symbols, not a pair. Merging `lo` is what put it on the board. **A
count can rise by becoming reachable.**

`e r` still sits at 7, untouched. Round 1 merged inside `low` and `log`, and
neither contains an `e` or an `r`, so nothing about `e r` changed. **A pair only
loses count when a merge eats one of its own symbols.**

So `e r` wins at 7. Merge `e r` -> `er`.

Note what just happened: `er` is a 2-character token and it merged **before**
`low`, a 3-character token. Length is not part of the rule. The only test is
which count is highest right now — 7 beat 5.

> **After round 2** — vocab 10 tokens, corpus 30 tokens.

**Round 3.** Words are `lo w`, `lo g`, `h er`, `p er`.

| Pair | Count |
|------|-------|
| `lo w` | 5 |
| `h er` | 4 |
| `lo g` | 3 |
| `p er` | 3 |

Merge `lo w` -> `low`.

> **After round 3** — vocab 11 tokens, corpus 25 tokens.

**Round 4.**

| Pair | Count |
|------|-------|
| `h er` | 4 |
| `lo g` | 3 |
| `p er` | 3 |

Merge `h er` -> `her`. A merged token can merge again — `er` is one symbol now,
so `h` + `er` is a normal pair.

> **After round 4** — vocab 12 tokens, corpus 21 tokens.

**Round 5.** Now a tie at the top.

| Pair | Count |
|------|-------|
| `lo g` | 3 |
| `p er` | 3 |

There is no correct choice between the two. Any tie-break gives a working
tokenizer. The only hard requirement is that it is **deterministic** — the same
corpus must always produce the same merge list, or encoding and decoding will
not match.

| Tie-break rule | Used by |
|------|---------|
| Smallest pair in lexicographic order | `subword-nmt` (original BPE paper code) |
| First pair seen in iteration order | HuggingFace `tokenizers` |

Never break ties randomly. Never break ties on dictionary order alone if your
dictionary is unordered — Python `dict` order depends on insertion, so a
different corpus reading order would silently change the vocabulary.

Taking lexicographic order, `lo g` wins. Merge it into `log`.

> **After round 5** — vocab 13 tokens, corpus 18 tokens.

Vocabulary hit the target of 13. Stop. `p er` never got merged.

#### What training produced

The merge list from step 4, all 5 rounds:

```text
1. l  + o  -> lo    (2 chars)
2. e  + r  -> er    (2 chars)
3. lo + w  -> low   (3 chars)
4. h  + er -> her   (3 chars)
5. lo + g  -> log   (3 chars)
```

The corpus is now:

```text
low    x5   1 token
log    x3   1 token
her    x4   1 token
p er   x3   2 tokens
```

The two numbers moved together the whole way:

| Round | Merge | Vocab | Corpus tokens |
|-------|-------|-------|---------------|
| — | base: `e g h l o p r w` | 8 | 45 |
| 1 | `lo` | 9 | 37 |
| 2 | `er` | 10 | 30 |
| 3 | `low` | 11 | 25 |
| 4 | `her` | 12 | 21 |
| 5 | `log` | 13 | 18 |

One merge adds exactly one token, so the final size is predictable:

$$
V = |{\text{base characters}}| + (\text{number of merges}) = 8 + 5 = 13
$$

That is the trade. Each merge costs one vocabulary slot and buys shorter
sequences. The three most frequent words collapsed to one token each. `per`, the
rarest, ran out of budget and stayed split as `p` + `er` — it never earned a
slot of its own.

Two things this example shows about merge order:

- **Length never enters the rule.** `er` is 2 characters and merged at round 2.
  `low` is 3 characters and merged at round 3. Not because one is shorter, but
  because 7 beat 5 at that moment. BPE compares counts and nothing else.
- **A pair keeps its count when merges happen elsewhere.** `e r` held 7 through
  round 1 because that merge was inside `low` and `log`, which contain no `e`
  or `r`. A count only drops when a merge eats one of the pair's own symbols.

Real training stops at a target $V$ (32k, 128k, 200k), exactly like the target
of 13 above. The merges you can afford go to whatever is most frequent, which is
why common words and code patterns become single tokens and rare words stay
split.

At encode time there are no ties. Each merge has a fixed rank from training, so
you always apply the lowest-rank merge available:

```python
pair = min(pairs, key=lambda p: merge_ranks.get(p, float("inf")))
```

Training produces two files:

- **vocab**: token string -> ID
- **merges**: the ordered list of merge rules

Encoding replays the merges in the same order on new text. Order matters. A
merge learned early must be applied early, or you get different tokens.

Complexity:

| Step | Cost |
|------|------|
| Train (naive) | $O(N \cdot M)$ for $N$ symbols, $M$ merges |
| Encode one word | $O(k^2)$ naive, $O(k \log k)$ with a heap |

The problem: plain BPE works on Unicode characters. Unicode has ~150,000
characters. You cannot put them all in the base vocabulary. Anything left out
becomes `<UNK>`, and `<UNK>` is unrecoverable — you cannot decode back to the
original text.

### Byte-level BPE

Byte-level BPE fixes this. Run BPE on **raw UTF-8 bytes** instead of characters.

The base vocabulary is exactly 256 tokens: byte `0` through byte `255`. Every
possible string is a sequence of bytes, so:

- No `<UNK>` token, ever.
- Encode then decode always returns the original text.
- Emoji, Chinese, code, binary garbage all work.

#### The same example, on bytes

Take the corpus from above. Step 1 splits into bytes instead of characters:

```text
6c 6f 77   x5    (low)
6c 6f 67   x3    (log)
68 65 72   x4    (her)
70 65 72   x3    (per)
```

Nothing else changes. Round 1 still counts `6c 6f` at 8 and merges it. The
algorithm is identical — only the starting symbols differ.

For plain ASCII, one character is one byte, so you get exactly the same 5
merges. The base vocabulary is the difference: 256 fixed byte tokens instead of
the 8 characters this corpus happened to contain. So the final size is
$256 + 5 = 261$.

That fixed base is the whole point. Swap `per` for `pér`:

```text
70 c3 a9 72   x3    (pér)
```

Character-level BPE has to ask whether `é` is in the vocabulary. If not,
`<UNK>`, and the text is unrecoverable. Byte-level BPE never asks — `c3` and
`a9` are already there, like every other byte.

The cost is length. `pér` starts as 4 symbols, not 3. A Chinese character is 3
bytes, an emoji is 4. Merges fix most of this during training, since frequent
multi-byte sequences become single tokens — but only for languages that appear
often enough in the training corpus to earn merges.

One detail: GPT-2 maps the 256 bytes to printable Unicode characters before
running BPE. Space (byte 32) becomes `Ġ`, newline becomes `Ċ`. This is only so
the vocab and merges files stay readable text — no invisible or control bytes in
them. It does not change the algorithm.

This is why the vocab lists `Ġlow` rather than `" low"`, and why `"cat"` and
`" cat"` are separate token IDs. See the pre-tokenization note above.

It also explains a familiar artifact: `é` is bytes `c3 a9`, which map to `Ã` and
`©`. A vocab file full of `Ã©` is not a bug — it is UTF-8 bytes shown one at a
time.

### What models use today

Byte-level BPE is the default everywhere.

| Model | Tokenizer |
|-------|-----------|
| GPT-2/3/4, o-series | byte-level BPE (`tiktoken`) |
| Llama 3, Qwen 2.5/3 | byte-level BPE |
| DeepSeek, GLM, Kimi | byte-level BPE |
| Llama 1/2, Mistral | SentencePiece BPE |
| BERT | WordPiece |
| T5, ALBERT | SentencePiece Unigram |

Vocabulary sizes grew over time: 32k (Llama 2) -> 128k (Llama 3) -> ~150k-200k
(GPT-4o, DeepSeek, GLM, Kimi). Bigger vocab compresses non-English text and code
into fewer tokens, which cuts sequence length and inference cost.

The trade-off is the embedding matrix. It is $V \times d_{\text{model}}$, so
doubling $V$ doubles that matrix and the output projection with it.

### BPE vs WordPiece vs Unigram

All three make subwords. They differ in how they pick merges.

| Method | Rule for picking |
|--------|------------------|
| BPE | Merge the most frequent pair |
| WordPiece | Merge the pair that most increases corpus likelihood |
| Unigram | Start with a big vocab, drop tokens that hurt likelihood least |

WordPiece scores a pair by $\frac{\text{count}(xy)}{\text{count}(x)\,\text{count}(y)}$
instead of raw count. This favors pairs that really belong together over pairs
that are just both common.

BPE won because it is simple, fast, and byte-level BPE removes `<UNK>` entirely.

# Training Pipeline

Tokenizer, data, and architecture are not enough to train. This section covers
what sits between them.

## What goes where

The project follows one rule, the same one behind `data.py` and `dataset.py`:

- **From-scratch code is for study.** Each piece has a test proving it matches
  the library version. `train.py` never imports it.
- **The real pipeline uses libraries.**
- **The model is the exception.** `transformer.py` stays hand-written.

| Study (tested against) | Real pipeline uses |
|---|---|
| `optim.py` — Adam, AdamW | `torch.optim.AdamW` |
| `schedule.py` — Noam | `torch.optim.lr_scheduler.LambdaLR` |
| `loss.py` — CE + smoothing | `nn.CrossEntropyLoss` |
| `metrics.py` — BLEU, perplexity | `sacrebleu` |

## Teacher forcing and the shift

Training does not generate. It shows the decoder the correct prefix and asks
for the next token. That is teacher forcing, and it means one target sequence
becomes two tensors:

```text
stored:   <bos>  5  1  4  <eos>

tgt_in:   <bos>  5  1  4          decoder reads
tgt_out:     5   1  4  <eos>      decoder must predict
```

The same tensor, offset by one. At every position the decoder reads `tgt_in[t]`
and is scored against `tgt_out[t]`.

Get the shift backwards and the model learns to copy its own input. Loss drops,
the curve looks healthy, and generation produces garbage. This lives in
`collate.py` and is the most-tested file in the project.

Generation cannot use teacher forcing — there is no target. Tokens come out one
at a time and feed back in, which is why a model with a shift bug looks fine
until you decode.

## Why warmup is not optional

Adam divides the update by $\sqrt{v}$, the running mean of squared gradients.
At step 1 that estimate comes from a single sample, so it is noise. Dividing by
the square root of noise gives huge, badly-aimed steps at the moment the model
is most fragile.

Warmup starts the learning rate near zero and ramps it up. By the time the rate
is high, $v$ is a real estimate. The Noam schedule:

$$
lr(step) = d_{\text{model}}^{-0.5} \cdot \min(step^{-0.5},\; step \cdot warmup^{-1.5})
$$

Two branches crossing at `step == warmup`:

- `step < warmup` → `step * warmup^-1.5`, grows linearly
- `step > warmup` → `step^-0.5`, decays as inverse sqrt

The $d_{\text{model}}^{-0.5}$ factor shrinks the peak for wider models, so the
same schedule transfers across sizes.

**Practical trap:** warmup is counted in *steps*, not epochs. The first toy run
here did 250 steps total against a 400-step warmup — the learning rate never
reached its peak, and the model looked broken when it was only undertrained.

## Padding appears in three places

`padding_idx` is not one setting. Miss any of the three and training quietly
degrades:

| Where | What it does | If missed |
|---|---|---|
| `Embedding` | zeroes the pad row's gradient | pad token learns a meaning |
| `CrossEntropyLoss(ignore_index=0)` | drops pad from loss and count | model rewarded for predicting `<pad>` |
| attention mask | blocks attending to pad | real tokens attend to nothing |

## Label smoothing

The hard target says $p_{\text{correct}} = 1$. To reach it the model must push
one logit to infinity — it becomes overconfident, and confident mistakes are
expensive. Smoothing spreads $\epsilon$ across all classes:

$$
target_{\text{correct}} = 1 - \epsilon + \frac{\epsilon}{V}, \quad
target_{\text{other}} = \frac{\epsilon}{V}
$$

The paper uses $\epsilon = 0.1$. It slightly *worsens* perplexity and reliably
*improves* BLEU, because translation has many valid outputs and total
confidence in one is wrong.

Evaluate perplexity with smoothing off, or the number is not comparable.

## The overfit gate

The highest-value test in the pipeline:

```bash
python train.py --task toy --overfit
```

Train on one batch and expect loss near zero. A model that cannot memorize a
single batch has a bug, not a tuning problem. Three causes show up here:

- target shift wrong → loss plateaus well above zero
- causal mask wrong → loss drops suspiciously fast, generation is garbage
- pad handling wrong → loss drops but decode emits `<pad>`

**Turn dropout off for this.** Dropout exists to prevent memorization, which is
the exact thing being measured. With dropout at 0.1 the gate stalls near 0.7
and reports a bug that does not exist — a mistake made while building this.

## Two fixes the model needed

**Embedding scale.** Multiply embeddings by $\sqrt{d_{\text{model}}}$ before
adding positional encoding. PE values sit in $[-1, 1]$ while embedding rows
start near unit normal. Without the scale the positional signal is
proportionally too loud, and early training is spent undoing it.

**Xavier init.** `nn.Linear` defaults to Kaiming, tuned for ReLU fan-in.
Transformers stack many residual projections; Xavier keeps forward activation
variance and backward gradient variance both near 1.

## Diagnosing a weak model

When the toy model produced the right digits in the wrong order, the question
was whether decoding or the model was at fault. The split test:

Run the model **teacher-forced** and measure token accuracy. Compare to greedy
decode.

- teacher-forced good, greedy bad → decoding bug
- both bad → the model itself is weak

Here both were ~0.5, so the decoder was fine and the model was data-starved.
Check this before tuning anything.

## Running it

```bash
python train.py --task toy --overfit      # correctness gate, seconds
python train.py --task toy                # synthetic reversal
python train.py --task multi30k           # real de->en translation

python -m pytest test_study_modules.py    # from-scratch vs library
python -m pytest test_pipeline.py         # collate, data, checkpoints
```
