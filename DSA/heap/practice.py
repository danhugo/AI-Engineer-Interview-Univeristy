"""
============================================================
  DSA — Heap | PRACTICE FILE
  Write every method yourself. Tests will tell you if
  you got it right. Do NOT open solution.py first.
============================================================

HOW TO USE
----------
1. Read the hint for each method.
2. Delete the `pass` and write your code.
3. Run:  python practice.py
4. A test PASS means your logic is correct. Fix until all pass.
5. Only open solution.py after you finish or are truly stuck.

This file covers two levels:
  LEVEL 1 — Min-Heap core   (push, pop, peek)
  LEVEL 2 — Build heap      (heapify from array)
"""


# ======================================================================
# LEVEL 1 — Min-Heap core
# ======================================================================
# Store values in a plain list (array).
# Parent of index i  →  (i - 1) // 2
# Left child of i    →  2 * i + 1
# Right child of i   →  2 * i + 2

class MinHeap:
    def __init__(self):
        pass

    def size(self) -> int:
        """Return the number of elements."""
        # TODO
        pass

    def is_empty(self) -> bool:
        """Return True if the heap has no elements."""
        # TODO
        pass

    def peek(self):
        """
        Return the smallest value without removing it.
        Raise IndexError if empty.

        HINT: smallest is always at index 0.
        """
        # TODO
        pass

    def push(self, val):
        """
        Add val to the heap.

        HINT:
          1. Append val to self._data.
          2. Call self._bubble_up(last index).
        """
        # TODO
        pass

    def pop(self):
        """
        Remove and return the smallest value.
        Raise IndexError if empty.

        HINT:
          1. Swap index 0 with the last element.
          2. Pop the last element (that's your return value).
          3. Call self._bubble_down(0) to restore the heap.
        """
        # TODO
        pass

    def _bubble_up(self, i: int):
        """
        Move element at index i up until the heap rule holds.

        HINT:
          While i > 0:
            parent = (i - 1) // 2
            if self._data[i] < self._data[parent]:
                swap them, set i = parent
            else:
                break
        """
        # TODO
        pass

    def _bubble_down(self, i: int):
        """
        Move element at index i down until the heap rule holds.

        HINT:
          While True:
            left  = 2 * i + 1
            right = 2 * i + 2
            smallest = i

            If left is in bounds and data[left] < data[smallest]:
                smallest = left
            If right is in bounds and data[right] < data[smallest]:
                smallest = right

            If smallest == i: break   (already in right place)
            swap data[i] and data[smallest], set i = smallest
        """
        # TODO
        pass



# ======================================================================
# LEVEL 2 — Build heap from array (heapify)
# ======================================================================

def heapify(arr: list) -> list:
    """
    Turn arr into a valid min-heap IN PLACE. Return arr.

    HINT:
      Start from the last non-leaf node: index = (n - 1 - 1) // 2 =   len(arr) // 2 - 1
      Loop from that index down to 0 (inclusive).
      Call _bubble_down_arr(arr, i) for each.

      You'll need to write _bubble_down_arr(arr, i) below —
      same logic as MinHeap._bubble_down but works on a plain list.
    """
    # TODO
    pass



def _bubble_down_arr(arr: list, i: int):
    """
    Bubble down element at index i within arr (min-heap order).

    HINT: same logic as MinHeap._bubble_down — just use arr directly.
    """
    # TODO
    pass



# ======================================================================
# TESTS — do not edit
# ======================================================================

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


# --- Level 1 ---

def test_push_peek():
    h = MinHeap()
    h.push(5); h.push(3); h.push(8); h.push(1)
    check(h.peek() == 1, "peek should return 1 (smallest)")
    check(h.size() == 4, "size should be 4")
    print("PASS  push / peek")


def test_pop_order():
    h = MinHeap()
    for v in [4, 1, 7, 3, 2]:
        h.push(v)
    out = [h.pop() for _ in range(5)]
    check(out == [1, 2, 3, 4, 7], f"pop order wrong: {out}")
    check(h.is_empty(), "heap should be empty after all pops")
    print("PASS  pop order")


def test_pop_empty():
    h = MinHeap()
    raised = False
    try:    h.pop()
    except IndexError: raised = True
    check(raised, "pop on empty heap should raise IndexError")
    print("PASS  pop empty raises IndexError")


def test_peek_empty():
    h = MinHeap()
    raised = False
    try:    h.peek()
    except IndexError: raised = True
    check(raised, "peek on empty heap should raise IndexError")
    print("PASS  peek empty raises IndexError")


# --- Level 2 ---

def test_heapify():
    arr = [5, 3, 8, 1, 4, 2, 7]
    heapify(arr)
    # verify heap property: every parent <= its children
    n = len(arr)
    for i in range(n // 2):
        left, right = 2*i+1, 2*i+2
        if left < n:
            check(arr[i] <= arr[left], f"heap violated at index {i} (left child)")
        if right < n:
            check(arr[i] <= arr[right], f"heap violated at index {i} (right child)")
    print("PASS  heapify")


if __name__ == "__main__":
    print("\n── Level 1: Min-Heap core ──")
    test_push_peek()
    test_pop_order()
    test_pop_empty()
    test_peek_empty()

    print("\n── Level 2: Heapify ──")
    test_heapify()

    print("\nAll tests passed!")
