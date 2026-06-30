# Hash Table — Interview Knowledge Sheet

## What is a Hash Table?
Maps keys → values in **O(1) average** time for get/put/delete.
Internally: an array where the index is derived from the key via a *hash function*.

---

## 1. Hashing Algorithms

### Goal
Turn any key (string, int, tuple …) into an array index.
A good hash function:
- Is **fast** to compute
- **Distributes** keys uniformly (avoids hotspots)
- Is **deterministic** (same key → same index every time)

### Common approaches

| Name | Formula | Notes |
|---|---|---|
| Division | `h(k) = k % m` | Simple; m should be prime |
| Multiplication | `h(k) = floor(m * frac(k·A))`, A ≈ 0.618 | Less sensitive to m |
| Polynomial rolling | `h = Σ ord(cᵢ) · BASE^i % MOD` | Standard for strings; BASE=31 or 37 |
| Universal hashing | Randomly chosen from a family | Prevents adversarial worst-case |
| SipHash (Python) | Keyed with a secret per-process seed | Default in CPython; guards against hash-flooding attacks |

**Interview rule**: for hand-rolled string hashing, use polynomial rolling with a large prime modulus (`10^9+9`).

---

## 2. Collisions

Two keys can produce the same index.  Two main strategies to handle this:

### 2a. Separate Chaining
Each bucket holds a **linked list** (or Python list) of all keys that hash there.

```
buckets[3] → [ ("apple", 5), ("cherry", 7) ]
```

- Lookup: hash → go to bucket → scan chain
- Average chain length = load factor λ
- Simple to implement; used in Python's `dict` (conceptually)

### 2b. Open Addressing
All entries live in **one flat array**.  On collision, probe forward.

| Variant | Probe sequence | Trade-off |
|---|---|---|
| Linear probing | `(h + i) % m` | Cache-friendly; causes *primary clustering* |
| Quadratic probing | `(h + i²) % m` | Reduces clustering; may miss some slots |
| Double hashing | `(h + i·h₂(k)) % m` | Best distribution; slightly more work |

**Deletion** requires a **tombstone** sentinel — you cannot blank the slot or it breaks the probe chain for other keys.

---

## 3. Load Factor

```
λ = n / m          (n = stored keys, m = buckets / slots)
```

| λ | Chaining | Open Addressing |
|---|---|---|
| < 0.5 | Very fast | Very fast |
| ≈ 0.75 | Still good (Python resize threshold) | Noticeably slower |
| > 0.9 | Chains getting long | Near O(n) |

**Rule of thumb**: resize when λ > 0.75 (chaining) or λ > 0.5–0.7 (open addressing).

---

## 4. Dynamic Resizing

When λ exceeds the threshold:
1. Allocate a **new array** twice as large
2. **Rehash** every existing key into the new array
3. Replace the old array

Single resize = O(n).
Amortised over all inserts = **O(1) per insert** (doubling strategy).

---

## 5. Complexity Summary

| Operation | Average | Worst case |
|---|---|---|
| put | O(1) | O(n) |
| get | O(1) | O(n) |
| delete | O(1) | O(n) |
| resize | O(n) | O(n) |

Worst case happens when all keys collide (bad hash or adversarial input).

---

## 6. AI-Engineer Interview Patterns

| Pattern | Hash table role |
|---|---|
| Count frequencies (Top K) | `count[x] += 1` |
| Two Sum | Store seen values; look up `target - x` |
| Group Anagrams | Key = sorted string, value = list of words |
| LRU Cache | Hash map + doubly linked list |
| Subarray sum equals k | Prefix sum → hash map |
| Sliding window (unique chars) | Map of char → last index |

---

## 7. Python `dict` Facts
- Implemented with open addressing (compact hash table since Python 3.6)
- Insertion-ordered since Python 3.7 (guaranteed)
- Resize threshold ≈ 2/3 of capacity
- `hash()` uses SipHash-1-3 with a random seed per process (PYTHONHASHSEED)
