# Hash Table — Q&A

---

## Basic Concept

**Q: What is a hash table?**
A: It maps keys to values using a hash function to compute an array index directly — like a coat check that gives you a numbered ticket so you can skip the search. Average time for get, put, and delete is O(1).

**Q: Why is lookup O(1) on average but O(n) worst case?**
A: On average, keys spread evenly so each slot holds ~1 item. Worst case is when all keys collide into the same bucket, forcing a scan of all n items.

**Q: What are the three properties of a good hash function?**
A: Fast to compute, deterministic (same key always gives the same index), and uniform (spreads keys evenly with no hotspots).

---

## Hashing Algorithms

**Q: What is the division method and when does it fail?**
A: `h(k) = k % m`. Fails when m is a power of 2 — only the low bits of k matter, so keys with similar endings cluster. m should be prime so all bits influence the result.

**Q: What is polynomial rolling hash and when do you use it?**
A: `h = Σ ord(cᵢ) · BASE^i % MOD`. Used for strings. Each character gets a unique weight so strings differing at any position produce different hashes. Use BASE = 31 or 37 and MOD = 10⁹+9.

**Q: What is the multiplication method?**
A: `h(k) = floor(m · frac(k · A))` where A ≈ 0.618 (golden ratio). Multiplying by an irrational number spreads keys evenly and is less sensitive to the choice of m than the division method.

**Q: What is universal hashing?**
A: Pick random values a and b at startup and compute `h(k) = ((a·k + b) % p) % m` where p is a large prime. For any two distinct keys, collision probability is exactly 1/m — no adversary can predict it.

**Q: What is SipHash and why does Python use it?**
A: A secret-keyed hash — output depends on both the input and a random per-process seed. Even knowing the algorithm, an attacker can't force collisions without the seed. Python uses it to block hash-flooding attacks.

**Q: What is a hash-flooding attack?**
A: An attacker sends many keys that all hash to the same bucket, degrading the server to O(n) per request. SipHash prevents this because the attacker doesn't know the secret seed.

---

## Collisions

**Q: What is a collision?**
A: When two different keys produce the same array index. Inevitable once keys outnumber buckets.

**Q: What are the two main collision resolution strategies?**
A: Separate chaining (each bucket holds a list of all key-value pairs that land there) and open addressing (one flat array; on collision, probe forward to find an open slot).

**Q: What are the three open addressing probe sequences?**
A: Linear probing `(h + i) % m` (cache-friendly, causes clustering), quadratic probing `(h + i²) % m` (reduces clustering, may skip slots), double hashing `(h + i·h₂(k)) % m` (best spread, slightly more work).

**Q: What is primary clustering?**
A: In linear probing, long runs of occupied slots form and grow — any key landing anywhere in the run probes into it, extending it further. This inflates average lookup time.

---

## Tombstones

**Q: Why can't you just blank a deleted slot in open addressing?**
A: Keys that were placed by probing past this slot would become unreachable — a blank looks like "never used" and stops the search early, wrongly returning "not found".

**Q: What is a tombstone?**
A: A special marker placed in a deleted slot. It means "something was here, keep probing" so the probe chain stays intact.

**Q: Walk through why `get(C)` needs tombstones given: `[_][_][A][💀][C][_EMPTY]`**
A: `get(C)` starts at index 2 (home). Slot 2 = A (not C, keep going). Slot 3 = 💀 (not C, but must keep going — C may be ahead). Slot 4 = C ✓. If slot 3 were blank, the search would stop at slot 3 and wrongly return "not found".

**Q: Why does `put()` bookmark the first tombstone instead of inserting there right away?**
A: The key might already exist further along — inserting at the tombstone immediately would create a duplicate. We must reach `_EMPTY` (proof the key isn't stored anywhere) before inserting.

**Q: Why insert at the first tombstone rather than the empty slot?**
A: The first tombstone is closer to the home index. Inserting there means fewer probe steps on every future lookup of that key.

**Q: What are the three cases `put()` handles when probing?**
A: 1) Slot is `_EMPTY` → insert here (or at `first_tomb` if one was bookmarked). 2) Slot is `_TOMBSTONE` → bookmark if it's the first one seen, keep probing. 3) Slot holds the exact key → update value in place and return.

---

## Sentinels

**Q: What is a sentinel and why use `object()` for it?**
A: A private value that signals a special internal state. `object()` creates a unique instance equal only to itself — no real user value can accidentally match it.

**Q: Why is `None` unsafe as a sentinel?**
A: Users can legitimately store `None` as a value. Using `None` to mean "slot is empty" makes it impossible to tell an empty slot from a slot holding the real value `None`.

**Q: Why use `is` instead of `==` when checking sentinels?**
A: `is` checks identity (same object in memory) and cannot be fooled by a custom `__eq__`. A class that overrides `__eq__` to always return `True` would match `== _TOMBSTONE` incorrectly; `is` never would.

**Q: What is the `default=_EMPTY` pattern in `get()`?**
A: Call `result = m.get(key, _EMPTY)`. If `result is _EMPTY`, the key doesn't exist. Otherwise the key exists and `result` is its value — which may legitimately be `None`, `0`, or `""`.

---

## Load Factor

**Q: What is the load factor?**
A: `λ = n / m` — number of stored keys divided by number of buckets/slots. Measures how full the table is.

**Q: When should a hash table resize?**
A: When λ > 0.75 for chaining, or λ > 0.5–0.7 for open addressing. Higher load means longer probe chains and performance closer to O(n).

**Q: Why does chaining tolerate a higher load factor than open addressing?**
A: Chaining spills into linked lists so the flat array never fills up. Open addressing has no overflow — once slots fill, probes must walk the whole table.

---

## Dynamic Resizing

**Q: What happens during a resize?**
A: Allocate a new array twice as large, rehash every existing key into the new positions, then swap out the old array.

**Q: Why must keys be rehashed, not just copied?**
A: The index formula is `hash(key) % size`. Doubling the size changes the modulus, so all old indices are wrong — every key must be recomputed.

**Q: Why is resize O(n) but amortised O(1) per insert?**
A: One resize moves all n keys — O(n). With doubling, a resize only happens after n more inserts. The O(n) cost spread over n inserts = O(1) amortised.

**Q: Why double the size instead of growing by a fixed amount?**
A: Fixed growth (e.g. +10) triggers O(n) resizes every 10 inserts, giving O(n²) total. Doubling makes resize events exponentially less frequent, keeping total cost O(n).

---

## Complexity Summary

**Q: What are the average and worst-case complexities for put, get, delete?**
A: Average O(1) for all three. Worst case O(n) when all keys collide into one bucket or chain.

**Q: What causes worst-case O(n)?**
A: A bad hash function that sends all keys to the same bucket, or adversarial input designed to cause maximum collisions — which is why SipHash uses a random secret seed.

---

## Python `dict` Facts

**Q: How is Python's `dict` implemented?**
A: Open addressing with a compact hash table (since Python 3.6). Not separate chaining.

**Q: Is Python's `dict` ordered?**
A: Yes — insertion order is guaranteed since Python 3.7.

**Q: At what load factor does Python's `dict` resize?**
A: Approximately 2/3 (≈ 0.67) of capacity.

**Q: What hash algorithm does Python use for strings?**
A: SipHash-1-3, keyed with a random per-process seed controlled by `PYTHONHASHSEED`.

---

## Interview Patterns

**Q: How does a hash map solve Two Sum?**
A: Scan the array and store each value seen so far. For each new value x, check if `target - x` is already in the map. If yes, you found the pair. O(n) time, O(n) space.

**Q: How does a hash map solve Top K Frequent Elements?**
A: First pass: count frequency of each element with `count[x] += 1`. Second pass: find the top K using a heap or sort. The map makes counting O(n).

**Q: How does a hash map solve Group Anagrams?**
A: Use the sorted version of each string as the key, and a list of originals as the value. Strings with the same letters produce the same key and group together automatically.

**Q: How does a hash map enable an LRU Cache?**
A: Combine a hash map (key → node, for O(1) access) with a doubly linked list (maintains recency order for O(1) eviction at either end).

**Q: How does a hash map solve Subarray Sum Equals K?**
A: Store prefix sums in a hash map. For each prefix sum S, check if `S - k` exists in the map — if yes, the subarray between those two indices sums to k.

**Q: How does a hash map help with the sliding window for unique characters?**
A: Map each character to its most recent index. When a duplicate is found, jump the left pointer past the previous occurrence — O(1) duplicate detection per step.
