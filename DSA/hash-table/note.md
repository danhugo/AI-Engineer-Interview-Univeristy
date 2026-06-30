# Hash Table — Interview Knowledge Sheet

## What is a Hash Table?

Think of it like a coat check: you hand over your coat (a **value**) and get a numbered ticket (the **index**). To get your coat back, you use the ticket — no searching required.

Hash tables map **keys → values** in **O(1) average** time for get/put/delete.
Internally: a plain array where the position (index) is calculated from the key using a *hash function*.

---

## 1. Hashing Algorithms

### Goal
Turn any key (string, int, tuple …) into a valid array index.

A good hash function must be:
- **Fast** to compute
- **Uniform** — spreads keys evenly, no hotspots
- **Deterministic** — same key always gives the same index

### Common approaches

| Name | Formula | Notes |
|---|---|---|
| Division | `h(k) = k % m` | Simple; m should be prime |
| Multiplication | `h(k) = floor(m * frac(k·A))`, A ≈ 0.618 | Less sensitive to choice of m |
| Polynomial rolling | `h = Σ ord(cᵢ) · BASE^i % MOD` | Standard for strings; BASE = 31 or 37 |
| Universal hashing | Randomly chosen from a family | Prevents adversarial worst-case |
| SipHash (Python) | Keyed with a secret per-process seed | Default in CPython; blocks hash-flooding attacks |

**Interview rule**: for hand-rolled string hashing, use polynomial rolling with a large prime modulus (`10^9+9`).

---

## 2. Collisions

Two different keys can produce the same index — that's a **collision**. Two main ways to handle it:

### 2a. Separate Chaining
Each bucket holds a **list** of all key-value pairs that land there.

```
buckets[3] → [ ("apple", 5), ("cherry", 7) ]
```

- Lookup: hash the key → go to bucket → scan the list
- Average list length = load factor λ
- Simple to implement; Python `dict` is conceptually similar

### 2b. Open Addressing
Everything lives in **one flat array**. On collision, probe (step forward) to find the next open slot.

| Variant | Probe sequence | Trade-off |
|---|---|---|
| Linear probing | `(h + i) % m` | Cache-friendly; causes *primary clustering* |
| Quadratic probing | `(h + i²) % m` | Reduces clustering; may skip some slots |
| Double hashing | `(h + i·h₂(k)) % m` | Best spread; slightly more work |

**Deletion** requires a **tombstone** marker — you cannot just blank the slot, or future lookups that probed past this spot will break.

#### Tombstones explained

A tombstone is needed when you delete a key from the middle of a probe chain.

```
put(A), put(B), put(C) — all hash to index 2
slots: [_][_][A][B][C][_EMPTY]   ← B and C were placed forward due to collisions

remove(B)
slots: [_][_][A][💀][C][_EMPTY]  ← slot 3 marked as tombstone, not empty
```

**Why get() must walk past tombstones**
If `get(C)` saw `💀` at slot 3 and treated it as empty, it would stop and return "not found" — even though C is alive at slot 4. The tombstone means: *something was here, keep looking*.

**Why put() inserts at the first tombstone, not the first empty slot**
```
put(D) — also hashes to index 2
probe 2 → A        (keep going)
probe 3 → 💀       (bookmark first_tomb=3, keep going)
probe 4 → C        (keep going)
probe 5 → _EMPTY   (stop — key D is definitely not here)

insert at first_tomb=3, not empty slot 5
→ future get(D): 2 steps instead of 4
```
Inserting at the closest tombstone keeps probe chains short.

**Why not insert at the tombstone immediately when we first see it?**
The key might already exist further along — we'd create a duplicate. We must reach `_EMPTY` (proof that the key isn't stored) before inserting.

#### Sentinels — `object()` as a unique marker

```python
_EMPTY     = object()   # slot was never used
_TOMBSTONE = object()   # slot was deleted
```

Each `object()` creates a unique value that equals **only itself** — it can never accidentally match a real key (unlike `None`, `0`, or `""`).

**Why use `is` instead of `==`?**
`is` checks identity (same object in memory), not equality. A key with a custom `__eq__` could return `True` for anything, fooling `==`. It can never fool `is`.

**The `default=_EMPTY` pattern**
```python
result = m.get("x", _EMPTY)
if result is _EMPTY:
    # key does not exist
else:
    # key exists — result could be None, 0, "" — all valid values
```

---

## 3. Load Factor

```
λ = n / m          (n = number of stored keys, m = number of buckets/slots)
```

| λ | Chaining | Open Addressing |
|---|---|---|
| < 0.5 | Very fast | Very fast |
| ≈ 0.75 | Still good (Python's resize threshold) | Noticeably slower |
| > 0.9 | Chains getting long | Near O(n) |

**Rule of thumb**: resize when λ > 0.75 (chaining) or λ > 0.5–0.7 (open addressing).

---

## 4. Dynamic Resizing

When λ exceeds the threshold:
1. Allocate a **new array** twice as large
2. **Rehash** every existing key into the new positions
3. Swap out the old array

One resize costs O(n), but spread across all inserts using the doubling strategy, it's **O(1) amortised per insert**.

---

## 5. Complexity Summary

| Operation | Average | Worst case |
|---|---|---|
| put | O(1) | O(n) |
| get | O(1) | O(n) |
| delete | O(1) | O(n) |
| resize | O(n) | O(n) |

Worst case happens when all keys collide (bad hash function or adversarial input).

---

## 6. Common Interview Patterns

| Pattern | How hash table helps |
|---|---|
| Count frequencies (Top K) | `count[x] += 1` |
| Two Sum | Store seen values; look up `target - x` |
| Group Anagrams | Key = sorted string, value = list of words |
| LRU Cache | Hash map + doubly linked list |
| Subarray sum equals k | Prefix sum → hash map |
| Sliding window (unique chars) | Map of char → last seen index |

---

## 7. Python `dict` Facts
- Uses open addressing (compact table since Python 3.6)
- Insertion-ordered since Python 3.7 (guaranteed by the spec)
- Resize threshold ≈ 2/3 of capacity
- `hash()` uses SipHash-1-3 with a random seed per process (`PYTHONHASHSEED`)

---

## 8. Why Hash Functions Spread Keys Evenly

### Division method — `h(k) = k % m`
When `m` is **prime**, every remainder 0..m-1 is equally reachable regardless of the key pattern. If `m` is a power of 2 instead, only the lowest bits of `k` matter — keys with similar endings all land in the same bucket. A prime `m` forces all bits to participate.

### Polynomial rolling hash — `h = Σ ord(cᵢ) · BASE^i % MOD`
Each character gets a **unique weight** (`BASE^i`), so two strings differing in any one position produce different sums. A prime `BASE` (31, 37) ensures no two positions can accidentally cancel each other. The large prime `MOD` (10⁹+9) wraps the result into a wide range, spreading strings evenly.

### Multiplication method — `h(k) = floor(m · frac(k · A))`, A ≈ 0.618
Multiplying by the **golden ratio** (irrational) means the fractional parts of `k·A` fill the range [0,1) evenly as k grows — they never bunch up. Related keys like k, 2k, 3k end up far apart.

### Universal hashing — `h(k) = ((a·k + b) % p) % m`
Random values `a` and `b` are picked at startup. For **any** two distinct keys, the chance they collide is exactly 1/m — no adversary can predict it without knowing `a` and `b`.

### SipHash (Python's default for `str`, `bytes`)
A **secret-keyed** hash: output depends on both the data and a random seed chosen at process start. Even knowing the algorithm, an attacker can't force collisions without the seed. This blocks **hash-flooding attacks** — where an attacker sends many inputs that all hash to the same bucket, degrading performance to O(n).
