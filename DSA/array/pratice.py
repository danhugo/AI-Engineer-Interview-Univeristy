"""
============================================================
  DSA — Dynamic Array (Vector) | PRACTICE FILE
  Write every method yourself. Tests will tell you if
  you got it right. Do NOT open the solution file first.
============================================================

HOW TO USE
----------
1. Read the hint for each method.
2. Delete the `pass` and write your code.
3. Run:  python vector_practice.py
4. A test PASS means your logic is correct. Fix until all pass.
5. Only open vector_solution.py after you finish or are truly stuck.
"""

import ctypes


# ============================================================
#  YOUR IMPLEMENTATION
#  Fill in every method marked with TODO.
# ============================================================

class Vector:
    def __init__(self):
        """
        Set up the three fields every vector needs:
          _capacity  — how many slots the raw array has right now
          _size      — how many items are actually stored
          _data      — the raw C array (use _make_array)

        Start with capacity=16, size=0.
        """
        # TODO
        pass

    def _make_array(self, capacity):
        """
        Allocate a raw fixed-size C array that can hold `capacity`
        Python objects.

        HINT: (capacity * ctypes.py_object)()
        This gives you a dumb block of memory — no append, no resize.
        That's intentional. You manage it manually.
        """
        # TODO
        pass

    # ── Accessors ──────────────────────────────────────────

    def size(self):
        """Return the number of items currently stored."""
        # TODO
        pass

    def capacity(self):
        """Return the number of slots in the underlying array."""
        # TODO
        pass

    def is_empty(self):
        """Return True if no items are stored."""
        # TODO
        pass

    def at(self, index):
        """
        Return the item at `index`.
        Raise IndexError if index < 0 or index >= size.

        HINT: Direct array access — self._data[index].
        Check bounds FIRST. Accessing out-of-bounds on a ctypes
        array gives undefined behaviour, not a clean error.
        """
        # TODO
        pass

    def find(self, item):
        """
        Return the FIRST index where item is found.
        Return -1 if not found.

        HINT: Linear scan from 0 to self._size (not self._capacity).
        """
        # TODO
        pass

    # ── Mutators ───────────────────────────────────────────

    def push(self, item):
        """
        Append item to the end.
        If size == capacity, resize to double capacity first.
        HINT: implement _resize function in advance.
        After resizing (or not), store the item at self._data[self._size]
        and increment self._size.
        """
        # TODO
        pass

    def insert(self, index, item):
        """
        Insert item at position `index`, shifting everything from
        `index` onward one slot to the RIGHT.

        Valid range: 0 <= index <= size  (inserting at size == append)
        Raise IndexError otherwise.

        HINT — shift direction matters:
          Shift from the END toward index, not from index toward the end.
          If you go left-to-right you overwrite elements before moving them.

          for i in range(self._size, index, -1):
              self._data[i] = self._data[i - 1]

        Don't forget to resize if needed BEFORE shifting.
        """
        # TODO
        pass

    def prepend(self, item):
        """
        Insert item at the front (index 0).
        HINT: one line — reuse insert().
        """
        # TODO
        pass

    def pop(self):
        """
        Remove and return the last item.
        Raise IndexError if empty.

        Shrink rule: after decrementing size, if
            size > 0  AND  size == capacity // 4
        then resize to capacity // 2.

        WHY capacity//4 and not capacity//2?
        If you shrank at capacity//2, alternating push/pop at
        the boundary would trigger a resize on every operation — O(n) each time.
        Waiting until 1/4 full gives a buffer zone.
        """
        # TODO
        pass

    def delete(self, index):
        """
        Remove item at `index`, shifting everything after it one slot LEFT.
        Raise IndexError if index is out of bounds (0 <= index < size).

        HINT — shift direction:
          for i in range(index, self._size - 1):
              self._data[i] = self._data[i + 1]

        Apply the same shrink rule as pop().
        """
        # TODO
        pass

    def remove(self, item):
        """
        Remove ALL occurrences of `item`. No error if item not found.

        HINT — tricky index management:
          Use a while loop, not a for loop.
          After deleting at index i, do NOT increment i.
          The element that was at i+1 just slid into i — you must
          check it again. Only increment if you did NOT delete.
        """
        # TODO
        pass

    # ── Private ────────────────────────────────────────────

    def _resize(self, new_capacity):
        """
        Allocate a new array of size `new_capacity`.
        Copy all self._size elements into it.
        Replace self._data and update self._capacity.

        HINT: allocate with _make_array, then loop over range(self._size).
        Do not try to copy the whole ctypes array at once — copy element by element.
        """
        # TODO
        pass

    # ── Python protocols ───────────────────────────────────

    def __len__(self):
        """
        Makes len(v) work.
        HINT: one line.
        """
        # TODO
        pass

    def __iter__(self):
        """
        Makes  `for x in v`  and  `list(v)`  work.
        HINT: use `yield` inside a loop over range(self._size).
        """
        # TODO
        pass

    def __repr__(self):
        items = [self._data[i] for i in range(self._size)]
        return f"Vector({items}, size={self._size}, cap={self._capacity})"


# ============================================================
#  TESTS
#  Do not edit these. They are your feedback mechanism.
#  Each test prints PASS or raises AssertionError with a message.
# ============================================================

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_initial_state():
    v = Vector()
    check(v.size() == 0,     "size() should be 0 on init")
    check(v.capacity() == 16, "capacity() should be 16 on init")
    check(v.is_empty() is True, "is_empty() should be True on init")
    print("PASS  initial state")


def test_push_and_at():
    v = Vector()
    v.push(10); v.push(20); v.push(30)
    check(v.size() == 3, "size should be 3 after three pushes")
    check(v.at(0) == 10, "at(0) should be 10")
    check(v.at(2) == 30, "at(2) should be 30")
    check(not v.is_empty(), "is_empty() should be False after pushes")

    # at() must raise on out-of-bounds
    raised = False
    try:    v.at(3)
    except IndexError: raised = True
    check(raised, "at(3) should raise IndexError when size=3")

    raised = False
    try:    v.at(-1)
    except IndexError: raised = True
    check(raised, "at(-1) should raise IndexError")

    print("PASS  push / at")


def test_insert_and_prepend():
    v = Vector()
    for x in [1, 2, 4]: v.push(x)

    v.insert(2, 3)          # [1, 2, 3, 4]
    check(v.size() == 4, "size should be 4 after insert")
    check(v.at(2) == 3, "inserted value should be at index 2")
    check(v.at(3) == 4, "element after insert point should shift right")

    v.prepend(0)            # [0, 1, 2, 3, 4]
    check(v.at(0) == 0, "prepend should place item at index 0")
    check(v.at(1) == 1, "existing items should shift right after prepend")

    v.insert(v.size(), 99)  # insert at end — valid
    check(v.at(v.size() - 1) == 99, "insert at size should append")

    raised = False
    try:    v.insert(v.size() + 1, 0)
    except IndexError: raised = True
    check(raised, "insert beyond size should raise IndexError")

    print("PASS  insert / prepend")


def test_pop_and_shrink():
    v = Vector()
    for i in range(16): v.push(i)
    check(v.capacity() == 16, "capacity should still be 16 when full")

    v.push(99)              # 17th item → resize
    check(v.capacity() == 32, "capacity should double to 32 after resize")
    check(v.size() == 17, "size should be 17")

    popped = v.pop()
    check(popped == 99, "pop should return the last pushed value")

    # Pop until size == capacity//4 == 8 → shrink to 16
    for _ in range(8): v.pop()
    check(v.size() == 8, "size should be 8")
    check(v.capacity() == 16,
          f"capacity should shrink to 16 when size==cap//4, got {v.capacity()}")

    # Pop from empty must raise
    raised = False
    try:    Vector().pop()
    except IndexError: raised = True
    check(raised, "pop on empty vector should raise IndexError")

    print("PASS  pop / shrink")


def test_delete():
    v = Vector()
    for x in [10, 20, 30, 40, 50]: v.push(x)

    v.delete(2)             # remove 30 → [10, 20, 40, 50]
    check(v.size() == 4, "size should decrease after delete")
    check(v.at(2) == 40, "element after deleted index should shift left")

    v.delete(0)             # remove first → [20, 40, 50]
    check(v.at(0) == 20, "after deleting index 0, next element becomes first")

    v.delete(v.size() - 1) # remove last → [20, 40]
    check(v.at(v.size() - 1) == 40, "last element should be 40")

    raised = False
    try:    v.delete(v.size())
    except IndexError: raised = True
    check(raised, "delete at index==size should raise IndexError")

    print("PASS  delete")


def test_remove():
    v = Vector()
    for x in [1, 2, 3, 2, 4, 2, 5]: v.push(x)
    v.remove(2)             # remove all 2s → [1, 3, 4, 5]
    check(v.size() == 4, "size should be 4 after removing all 2s")
    check(list(v) == [1, 3, 4, 5], f"expected [1,3,4,5], got {list(v)}")

    v.remove(999)           # no-op
    check(v.size() == 4, "remove of missing value should not change size")

    # Remove only element
    s = Vector(); s.push(42); s.remove(42)
    check(s.is_empty(), "vector should be empty after removing only element")

    # Remove at boundaries
    v2 = Vector()
    for x in [7, 1, 2, 3, 7]: v2.push(x)
    v2.remove(7)
    check(list(v2) == [1, 2, 3], f"expected [1,2,3], got {list(v2)}")

    print("PASS  remove")


def test_find():
    v = Vector()
    for x in [10, 20, 30, 20, 40]: v.push(x)
    check(v.find(10) == 0, "find should return index 0 for first element")
    check(v.find(20) == 1, "find should return FIRST occurrence")
    check(v.find(99) == -1, "find should return -1 when not found")
    check(Vector().find(1) == -1, "find on empty vector should return -1")
    print("PASS  find")


def test_resize():
    v = Vector()
    for i in range(16): v.push(i)
    check(v.capacity() == 16, "capacity should be 16 when exactly full")
    v.push(100)
    check(v.capacity() == 32, "capacity should double on resize")

    # Data must survive the resize intact
    for i in range(16):
        check(v.at(i) == i, f"data at index {i} corrupted after resize")
    check(v.at(16) == 100, "newly pushed item should be at index 16")

    # Trigger a second resize
    for i in range(15): v.push(i)  # fill to 32
    v.push(0)                       # triggers resize to 64
    check(v.capacity() == 64, "capacity should be 64 after second resize")
    print("PASS  resize")


def test_len_and_iter():
    v = Vector()
    for x in [1, 2, 3, 4, 5]: v.push(x)
    check(len(v) == 5, "__len__ should return 5")
    check(list(v) == [1, 2, 3, 4, 5], "__iter__ should yield items in order")
    check(sum(v) == 15, "__iter__ should work with sum()")
    print("PASS  __len__ / __iter__")


def test_stress():
    v = Vector()
    for i in range(200): v.push(i)
    check(v.size() == 200, "size should be 200 after 200 pushes")
    for i in range(199, -1, -1):
        val = v.pop()
        check(val == i, f"expected {i} from pop, got {val}")
    check(v.is_empty(), "vector should be empty after popping everything")

    # Mixed operations
    v2 = Vector()
    for i in range(10): v2.push(i)   # [0..9]
    v2.prepend(99)                    # [99,0..9]
    v2.insert(5, 55)                  # [99,0,1,2,3,55,4..9]
    v2.delete(0)                      # [0,1,2,3,55,4..9]
    v2.remove(55)                     # [0,1,2,3,4..9]
    check(v2.size() == 10, f"size should be 10, got {v2.size()}")
    check(list(v2) == list(range(10)),
          f"expected 0-9 in order, got {list(v2)}")
    print("PASS  stress")


# ============================================================
#  RUNNER
# ============================================================

TESTS = [
    ("Initial state",    test_initial_state),
    ("push / at",        test_push_and_at),
    ("insert / prepend", test_insert_and_prepend),
    ("pop / shrink",     test_pop_and_shrink),
    ("delete",           test_delete),
    ("remove",           test_remove),
    ("find",             test_find),
    ("resize",           test_resize),
    ("__len__ / __iter__", test_len_and_iter),
    ("stress",           test_stress),
]

if __name__ == "__main__":
    print("=" * 48)
    print("  Vector Practice")
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