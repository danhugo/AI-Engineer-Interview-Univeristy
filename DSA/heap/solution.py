"""
============================================================
  DSA — Heap | SOLUTION FILE
  Only open this after you've attempted practice.py.
============================================================
"""


# ======================================================================
# LEVEL 1 — Min-Heap core
# ======================================================================

class MinHeap:
    def __init__(self):
        self._data = []

    def size(self) -> int:
        return len(self._data)

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty heap")
        return self._data[0]

    def push(self, val):
        self._data.append(val)
        self._bubble_up(len(self._data) - 1)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty heap")
        # swap top with last, remove last, restore heap
        self._data[0], self._data[-1] = self._data[-1], self._data[0]
        val = self._data.pop()
        if self._data:
            self._bubble_down(0)
        return val

    def _bubble_up(self, i: int):
        while i > 0:
            parent = (i - 1) // 2
            if self._data[i] < self._data[parent]:
                self._data[i], self._data[parent] = self._data[parent], self._data[i]
                i = parent
            else:
                break

    def _bubble_down(self, i: int):
        n = len(self._data)
        while True:
            smallest = i
            left  = 2 * i + 1
            right = 2 * i + 2

            if left < n and self._data[left] < self._data[smallest]:
                smallest = left
            if right < n and self._data[right] < self._data[smallest]:
                smallest = right

            if smallest == i:
                break  # already in the right place

            self._data[i], self._data[smallest] = self._data[smallest], self._data[i]
            i = smallest


# ======================================================================
# LEVEL 2 — Build heap from array (heapify)
# ======================================================================

def heapify(arr: list) -> list:
    # start from last non-leaf, bubble down each node toward index 0
    # nodes from n//2 onward are leaves — no children, nothing to do
    for i in range(len(arr) // 2 - 1, -1, -1):
        _bubble_down_arr(arr, i)
    return arr


def _bubble_down_arr(arr: list, i: int):
    n = len(arr)
    while True:
        smallest = i
        left  = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] < arr[smallest]:
            smallest = left
        if right < n and arr[right] < arr[smallest]:
            smallest = right

        if smallest == i:
            break

        arr[i], arr[smallest] = arr[smallest], arr[i]
        i = smallest


# ======================================================================
# LEVEL 3 — Interview problems
# ======================================================================

import heapq


def top_k_frequent(nums: list, k: int) -> list:
    # count frequency of each number
    count = {}
    for n in nums:
        count[n] = count.get(n, 0) + 1

    # min-heap of (frequency, number) — keeps only top k
    h = []
    for num, freq in count.items():
        heapq.heappush(h, (freq, num))
        if len(h) > k:
            heapq.heappop(h)  # drop the least frequent

    return [num for freq, num in h]


def kth_largest(nums: list, k: int) -> int:
    # min-heap of size k — top is always the kth largest
    h = []
    for n in nums:
        heapq.heappush(h, n)
        if len(h) > k:
            heapq.heappop(h)  # drop anything smaller than kth largest
    return h[0]


def merge_k_sorted_lists(lists: list) -> list:
    result = []
    h = []

    # seed the heap with the first element of each list
    for li, lst in enumerate(lists):
        if lst:
            heapq.heappush(h, (lst[0], li, 0))  # (value, list_index, element_index)

    while h:
        val, li, ei = heapq.heappop(h)
        result.append(val)
        # push the next element from the same list
        if ei + 1 < len(lists[li]):
            heapq.heappush(h, (lists[li][ei + 1], li, ei + 1))

    return result


# ======================================================================
# TESTS — same as practice.py
# ======================================================================

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


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


def test_heapify():
    arr = [5, 3, 8, 1, 4, 2, 7]
    heapify(arr)
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
