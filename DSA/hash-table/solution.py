"""
============================================================
  DSA — Hash Map | SOLUTION FILE
  Do NOT open this until you've attempted the practice file.
============================================================
"""


# ======================================================================
# HASHING ALGORITHMS — reference notes
# ======================================================================
# A hash function turns any key into an integer index.
# Good ones are fast, deterministic, and distribute keys uniformly.
#
# Common approaches:
#   Division method      h(k) = k % m          — simple; m should be prime
#   Polynomial rolling   h = Σ ord(cᵢ)·BASE^i  — standard for strings
#   Universal hashing    random family of funcs — prevents adversarial attacks
#   SipHash (CPython)    keyed with secret seed — default for str/bytes in Python
#
# Polynomial rolling hash — the hand-rolled standard:
#   BASE = 31 (small prime), MOD = 10^9+9 (large prime)
#   h = 0
#   for c in s: h = (h * BASE + ord(c)) % MOD

def poly_hash(s: str, base: int = 31, mod: int = 10**9 + 9) -> int:
    h = 0
    for ch in s:
        h = (h * base + ord(ch)) % mod
    return h


# ======================================================================
# LEVEL 1 — Separate Chaining
# ======================================================================
# Each bucket is a list of [key, value] pairs.
# Collision: two keys hash to the same index → both live in that bucket.
# Average lookup cost = O(1 + λ) where λ = load factor.

class HashMapChaining:
    def __init__(self, size: int = 8):
        self.size = size
        self.buckets = [[] for _ in range(size)]
        self.count = 0

    def _index(self, key) -> int:
        return abs(hash(key)) % self.size

    def put(self, key, value):
        i = self._index(key)
        for pair in self.buckets[i]:
            if pair[0] == key:
                pair[1] = value
                return
        self.buckets[i].append([key, value])
        self.count += 1

    def get(self, key, default=None):
        i = self._index(key)
        for pair in self.buckets[i]:
            if pair[0] == key:
                return pair[1]
        return default

    def contains(self, key) -> bool:
        i = self._index(key)
        return any(p[0] == key for p in self.buckets[i])

    def remove(self, key):
        i = self._index(key)
        for j, pair in enumerate(self.buckets[i]):
            if pair[0] == key:
                self.buckets[i].pop(j)
                self.count -= 1
                return

    def __len__(self):
        return self.count


# ======================================================================
# LEVEL 2 — Chaining with Load Factor & Resize
# ======================================================================
# Load factor λ = n / m   (n = stored keys, m = buckets)
# When λ > 0.75 → double m and rehash every key.
# Single resize = O(n). Amortised over all inserts = O(1) per insert.

LOAD_THRESHOLD = 0.75


class HashMapWithResize:
    def __init__(self, initial_size: int = 4):
        self.size = initial_size
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0

    def _index(self, key) -> int:
        return abs(hash(key)) % self.size

    def _load_factor(self) -> float:
        return self.count / self.size

    def _resize(self):
        old_buckets = self.buckets
        self.size *= 2
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)

    def put(self, key, value):
        i = self._index(key)
        for pair in self.buckets[i]:
            if pair[0] == key:
                pair[1] = value
                return
        self.buckets[i].append([key, value])
        self.count += 1
        if self._load_factor() > LOAD_THRESHOLD:
            self._resize()

    def get(self, key, default=None):
        i = self._index(key)
        for pair in self.buckets[i]:
            if pair[0] == key:
                return pair[1]
        return default

    def contains(self, key) -> bool:
        i = self._index(key)
        return any(p[0] == key for p in self.buckets[i])

    def remove(self, key):
        i = self._index(key)
        for j, pair in enumerate(self.buckets[i]):
            if pair[0] == key:
                self.buckets[i].pop(j)
                self.count -= 1
                return

    def __len__(self):
        return self.count


# ======================================================================
# LEVEL 3 — Open Addressing (linear probing + tombstone deletion)
# ======================================================================
# All entries in ONE flat array. On collision, probe forward.
# Probe sequence: index = (start + i) % size
#
# _EMPTY     — slot never used → stop probing
# _TOMBSTONE — slot was deleted → continue probing
#
# Deletion must use a tombstone, not a blank, or the probe chain breaks.

_EMPTY     = object()
_TOMBSTONE = object()


class HashMapOpenAddressing:
    def __init__(self, initial_size: int = 8):
        self.size = initial_size
        self.slots = [_EMPTY] * self.size
        self.vals  = [None]   * self.size
        self.count = 0

    def _index(self, key) -> int:
        return abs(hash(key)) % self.size

    def put(self, key, value):
        i = self._index(key)
        first_tomb = None

        for _ in range(self.size):
            if self.slots[i] is _EMPTY:
                target = first_tomb if first_tomb is not None else i
                self.slots[target] = key
                self.vals[target]  = value
                self.count += 1
                return
            if self.slots[i] is _TOMBSTONE:
                if first_tomb is None:
                    first_tomb = i
            elif self.slots[i] == key:
                self.vals[i] = value
                return
            i = (i + 1) % self.size

    def get(self, key, default=None):
        i = self._index(key)
        for _ in range(self.size):
            if self.slots[i] is _EMPTY:
                return default
            if self.slots[i] is not _TOMBSTONE and self.slots[i] == key:
                return self.vals[i]
            i = (i + 1) % self.size
        return default

    def contains(self, key) -> bool:
        return self.get(key, _EMPTY) is not _EMPTY

    def remove(self, key):
        i = self._index(key)
        for _ in range(self.size):
            if self.slots[i] is _EMPTY:
                return
            if self.slots[i] is not _TOMBSTONE and self.slots[i] == key:
                self.slots[i] = _TOMBSTONE
                self.count -= 1
                return
            i = (i + 1) % self.size

    def __len__(self):
        return self.count


# ======================================================================
# Tests
# ======================================================================

def test_chaining_basic():
    m = HashMapChaining()
    m.put("a", 1); m.put("b", 2)
    assert m.get("a") == 1
    assert m.get("b") == 2
    m.put("a", 100)
    assert m.get("a") == 100
    assert len(m) == 2
    assert m.get("z") is None
    assert m.get("z", 0) == 0
    assert m.contains("b") is True
    assert m.contains("z") is False
    m.remove("a")
    assert m.contains("a") is False
    assert len(m) == 1
    print("PASS  chaining basic")


def test_chaining_counting():
    nums = [1, 1, 1, 2, 2, 3]
    c = HashMapChaining()
    for x in nums:
        c.put(x, c.get(x, 0) + 1)
    assert c.get(1) == 3
    assert c.get(2) == 2
    assert c.get(3) == 1
    print("PASS  chaining counting (Top K pattern)")


def test_resize():
    m = HashMapWithResize(initial_size=4)
    for i in range(20):
        m.put(f"k{i}", i)
    assert len(m) == 20
    assert m.size > 4
    for i in range(20):
        assert m.get(f"k{i}") == i, f"k{i} lost after resize"
    print(f"PASS  resize (final size={m.size})")


def test_open_addressing_basic():
    oa = HashMapOpenAddressing()
    oa.put("x", 10); oa.put("y", 20)
    assert oa.get("x") == 10
    oa.put("x", 99)
    assert oa.get("x") == 99
    assert len(oa) == 2
    oa.remove("x")
    assert oa.contains("x") is False
    assert oa.get("y") == 20
    assert len(oa) == 1
    print("PASS  open addressing basic")


def test_open_addressing_tombstone():
    oa = HashMapOpenAddressing()
    oa.put("a", 1); oa.put("b", 2); oa.put("c", 3)
    oa.remove("b")
    assert oa.get("c") == 3
    oa.put("d", 4)
    assert oa.get("d") == 4
    print("PASS  open addressing tombstone")


def test_poly_hash():
    h = poly_hash("hello")
    assert isinstance(h, int) and h >= 0
    assert poly_hash("hello") == poly_hash("hello")
    assert poly_hash("abc") != poly_hash("bca")
    print("PASS  poly_hash")


if __name__ == "__main__":
    print("=" * 48)
    print("  Hash Map — SOLUTION")
    print("=" * 48)
    test_chaining_basic()
    test_chaining_counting()
    test_resize()
    test_open_addressing_basic()
    test_open_addressing_tombstone()
    test_poly_hash()
    print("=" * 48)
    print("  All tests passed.")
    print("=" * 48)
