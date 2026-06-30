"""
============================================================
  DSA — Hash Map | PRACTICE FILE
  Write every method yourself. Tests will tell you if
  you got it right. Do NOT open the solution file first.
============================================================

HOW TO USE
----------
1. Read the hint for each method.
2. Delete the `raise NotImplementedError` and write your code.
3. Run:  python practice.py
4. A test PASS means your logic is correct. Fix until all pass.
5. Only open solution.py after you finish or are truly stuck.

This file covers three levels:
  LEVEL 1 — Separate Chaining         (the core idea)
  LEVEL 2 — Load Factor & Resize      (why O(1) stays O(1))
  LEVEL 3 — Open Addressing           (no linked lists)
  BONUS   — Polynomial Rolling Hash   (hand-roll a hash function)
"""


# ======================================================================
# LEVEL 1 — Separate Chaining  (no resize yet)
# ======================================================================
# Each bucket is a list of [key, value] pairs.
# Collision = two keys land on the same index → both live in same bucket.

class HashMapChaining:
    def __init__(self, size: int = 8):
        self.size = size
        self.buckets = [[] for _ in range(size)]
        self.count = 0

    def _index(self, key) -> int:
        """
        Return the bucket index for this key.

        HINT: abs(hash(key)) % self.size
        abs() guards against negative hash values (possible in Python).
        % self.size squeezes the big integer into [0, size).
        """
        # TODO
        pass

    def put(self, key, value):
        """
        Insert key→value, or update value if key already exists.

        HINT:
          1. i = self._index(key)
          2. Loop self.buckets[i]. If pair[0] == key → update pair[1], return.
          3. If not found → append [key, value], increment self.count.
        """
        # TODO
        pass

    def get(self, key, default=None):
        """
        Return the value for key, or default if not found.

        HINT:
          Get the bucket with _index. Loop it.
          Return pair[1] when pair[0] == key.
          Fall through to return default.
        """
        # TODO
        pass

    def contains(self, key) -> bool:
        """
        Return True if key exists, False otherwise.

        HINT: reuse get() with a sentinel that is guaranteed not to be
        a real value.  Example: get(key, _SENTINEL) is not _SENTINEL.
        Or just loop the bucket directly.
        """
        # TODO
        pass

    def remove(self, key):
        """
        Delete key if it exists. No error if key is absent.

        HINT:
          i = self._index(key)
          Loop with enumerate. When pair[0] == key → bucket.pop(j),
          decrement self.count, return.
        """
        # TODO
        pass

    def __len__(self):
        return self.count


# ======================================================================
# LEVEL 2 — Chaining WITH Load Factor & Resize
# ======================================================================
# Load factor λ = n / m   (n = stored keys, m = buckets)
# When λ > 0.75 → double the bucket array and REHASH every key.
# This keeps average chain length ≤ 1, preserving O(1) lookups.

LOAD_THRESHOLD = 0.75


class HashMapWithResize:
    def __init__(self, initial_size: int = 4):
        self.size = initial_size
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0

    def _index(self, key) -> int:
        """Same as Level 1: abs(hash(key)) % self.size"""
        # TODO
        pass

    def _load_factor(self) -> float:
        """
        Return the current load factor.

        HINT: self.count / self.size
        """
        # TODO
        pass

    def _resize(self):
        """
        Double the bucket array and rehash every existing key.

        HINT:
          1. old_buckets = self.buckets
          2. self.size *= 2
          3. self.buckets = [[] for _ in range(self.size)]
          4. self.count = 0
          5. For every [key, value] in old_buckets: call self.put(key, value).
             put() will compute the new index automatically.
        """
        # TODO
        pass

    def put(self, key, value):
        """
        Same logic as Level 1, THEN trigger a resize when needed.

        HINT: after inserting (not updating), check:
          if self._load_factor() > LOAD_THRESHOLD:
              self._resize()
        """
        # TODO
        pass

    def get(self, key, default=None):
        """Same as Level 1."""
        # TODO
        pass

    def contains(self, key) -> bool:
        """Same as Level 1."""
        # TODO
        pass

    def remove(self, key):
        """Same as Level 1."""
        # TODO
        pass

    def __len__(self):
        return self.count


# ======================================================================
# LEVEL 3 — Open Addressing  (linear probing + tombstone deletion)
# ======================================================================
# No linked lists. All entries live in ONE flat array.
# Collision → probe forward until an empty slot is found.
#
# Probe sequence (linear):  index = (start + i) % size
#
# TWO sentinel objects mark slot state:
#   _EMPTY     — never used → STOP probing (key definitely not here)
#   _TOMBSTONE — was deleted → CONTINUE probing (key may be further along)
#
# WHY tombstones?
#   If you blank a deleted slot, the probe chain breaks and later
#   lookups incorrectly return "not found".
#
# TOMBSTONE INTUITION:
#
#   Step 1 — collision forces keys to spread out via probing:
#     put(A), put(B), put(C) all hash to index 2
#     slots: [_][_][A][B][C][_EMPTY]   (B and C probed forward)
#
#   Step 2 — a delete creates a tombstone in the middle:
#     remove(B)
#     slots: [_][_][A][💀][C][_EMPTY]
#
#   Step 3 — now get(C) must walk PAST the tombstone:
#     probe 2 → A  (not C, keep going)
#     probe 3 → 💀 (not C, but don't stop! C is at 4)
#     probe 4 → C  ✓ found
#     If slot 3 were _EMPTY instead of 💀, we'd stop at 3
#     and wrongly return "not found".
#
#   Step 4 — put(D) with same home index 2, after the delete above:
#     slots: [_][_][A][💀][C][_EMPTY]
#     probe 2 → A        (not D, not empty → keep going)
#     probe 3 → 💀       (not D → keep going, but bookmark 3 as first_tomb)
#     probe 4 → C        (not D, not empty → keep going)
#     probe 5 → _EMPTY   (stop — D is not in the table)
#     insert at first_tomb=3, not at empty slot 5
#     → slot 3 is closer to home, so future get(D) only needs 2 probes
#     → if we inserted at 5, get(D) would need 4 probes every time
#     WHY not insert at tombstone 3 immediately when we first see it?
#     → because D might already exist at slot 4 or 5. We must check
#       first to avoid duplicates. Only _EMPTY guarantees "not here".

_EMPTY     = object()
_TOMBSTONE = object()


class HashMapOpenAddressing:
    def __init__(self, initial_size: int = 8):
        self.size = initial_size
        self.slots = [_EMPTY] * self.size   # stores keys
        self.vals  = [None]   * self.size   # stores values
        self.count = 0

    def _index(self, key) -> int:
        """abs(hash(key)) % self.size"""
        # TODO
        pass

    def put(self, key, value):
        """
        Insert or update using linear probing.

        HINT — tombstones only exist because of past deletes:
          Example: insert A, B, C all hashing to index 2, then delete B:
            slots: [_][_][A][💀][C][_EMPTY]
          Now put("D") also hashes to 2. You must walk past the tombstone
          because C is still alive further along. But you bookmark the
          tombstone so you can reuse that slot for D instead of the
          distant empty slot — keeps probe chains short.

          Three cases as you walk slot by slot:
          1. Slot is _EMPTY
             → key not in table. Insert here — BUT if you bookmarked a
               tombstone earlier, insert THERE instead (closer to home).
               Increment count and return.

          2. Slot is _TOMBSTONE
             → something was deleted here. Keep probing — key might be
               further along. Bookmark this index if it's the first
               tombstone you've seen (you'll reuse it in case 1).

          3. Slot holds your exact key
             → update value in place and return.

          Use `for _ in range(self.size)` to cap the loop at size steps.
        """
        # TODO
        pass
                


    def get(self, key, default=None):
        """
        Probe from _index(key).
        Stop on _EMPTY (not found). Skip _TOMBSTONE. Return val on match.

        HINT:
          i = self._index(key)
          for _ in range(self.size):
              if self.slots[i] is _EMPTY: return default
              if self.slots[i] is not _TOMBSTONE and self.slots[i] == key:
                  return self.vals[i]
              i = (i + 1) % self.size
          return default
        """
        # TODO
        pass

    def contains(self, key) -> bool:
        """
        HINT: use _EMPTY as a sentinel so values of None work correctly.
          self.get(key, _EMPTY) is not _EMPTY
        """
        # TODO
        pass

    def remove(self, key):
        """
        Find the key with the same probe as get().
        DO NOT blank the slot — set self.slots[i] = _TOMBSTONE.
        Decrement self.count.
        """
        # TODO
        pass

    def __len__(self):
        return self.count


# ======================================================================
# BONUS — Polynomial Rolling Hash
# ======================================================================

def poly_hash(s: str) -> int:
    """
    Hand-roll a hash for string s.

    Formula:
      h = 0
      for each char c in s:
          h = (h * 31 + ord(c)) % (10**9 + 9)
      return h

    Why 31? Small prime — reduces clustering.
    Why 10^9+9? Large prime modulus — keeps h in a safe integer range.
    """
    # TODO
    pass


# ======================================================================
# TESTS — do not edit
# ======================================================================

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_chaining_basic():
    m = HashMapChaining()
    m.put("a", 1); m.put("b", 2)
    check(m.get("a") == 1, "get('a') should be 1")
    check(m.get("b") == 2, "get('b') should be 2")

    m.put("a", 100)
    check(m.get("a") == 100, "get('a') should be 100 after update")
    check(len(m) == 2, f"len should be 2 after update, got {len(m)}")

    check(m.get("z") is None, "missing key should return None")
    check(m.get("z", 0) == 0, "missing key with default should return 0")
    check(m.contains("b") is True, "contains('b') should be True")
    check(m.contains("z") is False, "contains('z') should be False")

    m.remove("a")
    check(m.contains("a") is False, "'a' should be gone after remove")
    check(len(m) == 1, f"len should be 1 after remove, got {len(m)}")
    print("PASS  chaining basic")


def test_chaining_counting():
    nums = [1, 1, 1, 2, 2, 3]
    c = HashMapChaining()
    for x in nums:
        c.put(x, c.get(x, 0) + 1)
    check(c.get(1) == 3, "1 should appear 3 times")
    check(c.get(2) == 2, "2 should appear 2 times")
    check(c.get(3) == 1, "3 should appear 1 time")
    print("PASS  chaining counting (Top K pattern)")


def test_resize():
    m = HashMapWithResize(initial_size=4)
    for i in range(20):
        m.put(f"k{i}", i)
    check(len(m) == 20, f"expected 20 items, got {len(m)}")
    check(m.size > 4, "bucket array should have grown beyond 4")
    for i in range(20):
        check(m.get(f"k{i}") == i, f"k{i} lost after resize")
    print(f"PASS  resize (final size={m.size})")


def test_open_addressing_basic():
    oa = HashMapOpenAddressing()
    oa.put("x", 10); oa.put("y", 20)
    check(oa.get("x") == 10, "get('x') should be 10")

    oa.put("x", 99)
    check(oa.get("x") == 99, "get('x') should be 99 after update")
    check(len(oa) == 2, f"len should be 2, got {len(oa)}")

    oa.remove("x")
    check(oa.contains("x") is False, "'x' should be gone after remove")
    check(oa.get("y") == 20, "probe must continue past tombstone to find 'y'")
    check(len(oa) == 1, f"len should be 1 after remove, got {len(oa)}")
    print("PASS  open addressing basic")


def test_open_addressing_tombstone():
    oa = HashMapOpenAddressing()
    oa.put("a", 1); oa.put("b", 2); oa.put("c", 3)
    oa.remove("b")
    check(oa.get("c") == 3, "get('c') must work after 'b' tombstoned")
    oa.put("d", 4)
    check(oa.get("d") == 4, "insert after tombstone should work")
    print("PASS  open addressing tombstone")


def test_poly_hash():
    h = poly_hash("hello")
    check(isinstance(h, int) and h >= 0, "hash must be a non-negative int")
    check(poly_hash("hello") == poly_hash("hello"), "must be deterministic")
    check(poly_hash("abc") != poly_hash("bca"), "order of chars must matter")
    print("PASS  poly_hash")


TESTS = [
    ("chaining basic",             test_chaining_basic),
    ("chaining counting",          test_chaining_counting),
    ("resize",                     test_resize),
    ("open addressing basic",      test_open_addressing_basic),
    ("open addressing tombstone",  test_open_addressing_tombstone),
    ("poly_hash",                  test_poly_hash),
]

if __name__ == "__main__":
    print("=" * 48)
    print("  Hash Map Practice")
    print("=" * 48)

    passed = 0
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
        except NotImplementedError:
            print(f"TODO  {name}  ← not implemented yet")
        except AssertionError as e:
            print(e)
        except Exception as e:
            print(f"ERR   {name}  — {type(e).__name__}: {e}")

    print("=" * 48)
    print(f"  {passed}/{len(TESTS)} tests passed")
    if passed == len(TESTS):
        print("  All done. Now go read the solution and")
        print("  check if your approach matches.")
    else:
        print("  Keep going. One method at a time.")
    print("=" * 48)
