"""
============================================================
  DSA — Dynamic Array (Vector) | SOLUTION FILE
  Do NOT open this until you've attempted the practice file.
============================================================
"""

import ctypes


class Vector:
    def __init__(self):
        self._capacity = 16
        self._size = 0
        self._data = self._make_array(self._capacity)

    def _make_array(self, capacity):
        return (capacity * ctypes.py_object)()

    # ── Accessors ──────────────────────────────────────────

    def size(self):
        return self._size

    def capacity(self):
        return self._capacity

    def is_empty(self):
        return self._size == 0

    def at(self, index):
        if not (0 <= index < self._size):
            raise IndexError(f"Index {index} out of bounds (size={self._size})")
        return self._data[index]

    def find(self, item):
        for i in range(self._size):
            if self._data[i] == item:
                return i
        return -1

    # ── Mutators ───────────────────────────────────────────

    def push(self, item):
        if self._size == self._capacity:
            self._resize(self._capacity * 2)
        self._data[self._size] = item
        self._size += 1

    def insert(self, index, item):
        if not (0 <= index <= self._size):
            raise IndexError(f"Index {index} out of bounds")
        if self._size == self._capacity:
            self._resize(self._capacity * 2)
        for i in range(self._size, index, -1):
            self._data[i] = self._data[i - 1]
        self._data[index] = item
        self._size += 1

    def prepend(self, item):
        self.insert(0, item)

    def pop(self):
        if self._size == 0:
            raise IndexError("Pop from empty vector")
        val = self._data[self._size - 1]
        self._size -= 1
        if self._size > 0 and self._size == self._capacity // 4:
            self._resize(self._capacity // 2)
        return val

    def delete(self, index):
        if not (0 <= index < self._size):
            raise IndexError(f"Index {index} out of bounds")
        for i in range(index, self._size - 1):
            self._data[i] = self._data[i + 1]
        self._size -= 1
        if self._size > 0 and self._size == self._capacity // 4:
            self._resize(self._capacity // 2)

    def remove(self, item):
        i = 0
        while i < self._size:
            if self._data[i] == item:
                self.delete(i)
                # Do NOT increment i — after deletion the next
                # element slides into position i, so we recheck it.
            else:
                i += 1

    # ── Private ────────────────────────────────────────────

    def _resize(self, new_capacity):
        new_data = self._make_array(new_capacity)
        for i in range(self._size):
            new_data[i] = self._data[i]
        self._data = new_data
        self._capacity = new_capacity

    # ── Python protocols ───────────────────────────────────

    def __len__(self):
        return self._size

    def __iter__(self):
        for i in range(self._size):
            yield self._data[i]

    def __repr__(self):
        items = [self._data[i] for i in range(self._size)]
        return f"Vector({items}, size={self._size}, cap={self._capacity})"


# ============================================================
#  Tests — every method, every edge case
# ============================================================

def test_initial_state():
    v = Vector()
    assert v.size() == 0
    assert v.capacity() == 16
    assert v.is_empty() is True
    print("PASS  initial state")


def test_push_and_at():
    v = Vector()
    v.push(10); v.push(20); v.push(30)
    assert v.size() == 3
    assert v.at(0) == 10
    assert v.at(2) == 30
    try:
        v.at(3)
        assert False
    except IndexError:
        pass
    print("PASS  push / at")


def test_insert_and_prepend():
    v = Vector()
    for x in [1, 2, 4]: v.push(x)
    v.insert(2, 3)                         # [1,2,3,4]
    assert [v.at(i) for i in range(4)] == [1, 2, 3, 4]
    v.prepend(0)                           # [0,1,2,3,4]
    assert v.at(0) == 0
    v.insert(v.size(), 99)                 # insert at end is valid
    assert v.at(v.size() - 1) == 99
    try:
        v.insert(v.size() + 1, 0)
        assert False
    except IndexError:
        pass
    print("PASS  insert / prepend")


def test_pop_and_shrink():
    v = Vector()
    for i in range(16): v.push(i)
    v.push(99)                             # triggers resize → cap=32
    assert v.capacity() == 32
    for _ in range(9): v.pop()            # size 17→8 == cap//4 → shrink to 16
    assert v.capacity() == 16
    try:
        Vector().pop()
        assert False
    except IndexError:
        pass
    print("PASS  pop / shrink")


def test_delete():
    v = Vector()
    for x in [10, 20, 30, 40, 50]: v.push(x)
    v.delete(2)                            # remove 30 → [10,20,40,50]
    assert [v.at(i) for i in range(4)] == [10, 20, 40, 50]
    v.delete(0)                            # remove first
    assert v.at(0) == 20
    v.delete(v.size() - 1)                # remove last
    assert v.at(v.size() - 1) == 40
    try:
        v.delete(v.size())
        assert False
    except IndexError:
        pass
    print("PASS  delete")


def test_remove():
    v = Vector()
    for x in [1, 2, 3, 2, 4, 2, 5]: v.push(x)
    v.remove(2)                            # removes all 2s → [1,3,4,5]
    assert list(v) == [1, 3, 4, 5]
    v.remove(999)                          # no-op
    assert v.size() == 4
    s = Vector(); s.push(42); s.remove(42)
    assert s.is_empty()
    print("PASS  remove")


def test_find():
    v = Vector()
    for x in [10, 20, 30, 20, 40]: v.push(x)
    assert v.find(10) == 0
    assert v.find(20) == 1                 # first occurrence
    assert v.find(99) == -1
    assert Vector().find(1) == -1
    print("PASS  find")


def test_resize():
    v = Vector()
    for i in range(16): v.push(i)
    assert v.capacity() == 16
    v.push(100)
    assert v.capacity() == 32
    for i in range(16): assert v.at(i) == i   # data intact after resize
    assert v.at(16) == 100
    print("PASS  resize")


def test_len_and_iter():
    v = Vector()
    for x in [1, 2, 3, 4, 5]: v.push(x)
    assert len(v) == 5
    assert list(v) == [1, 2, 3, 4, 5]
    assert sum(v) == 15                    # iter works with builtins
    print("PASS  __len__ / __iter__")


def test_stress():
    v = Vector()
    for i in range(200): v.push(i)
    assert v.size() == 200
    for i in range(199, -1, -1):
        assert v.pop() == i
    assert v.is_empty()

    v2 = Vector()
    for i in range(10): v2.push(i)
    v2.prepend(99)
    v2.insert(5, 55)
    v2.delete(0)
    v2.remove(55)
    assert v2.size() == 10
    assert list(v2) == list(range(10))
    print("PASS  stress")


if __name__ == "__main__":
    print("=" * 48)
    print("  Vector — SOLUTION")
    print("=" * 48)
    test_initial_state()
    test_push_and_at()
    test_insert_and_prepend()
    test_pop_and_shrink()
    test_delete()
    test_remove()
    test_find()
    test_resize()
    test_len_and_iter()
    test_stress()
    print("=" * 48)
    print("  All tests passed.")
    print("=" * 48)