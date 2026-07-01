# Heap — Interview Knowledge Sheet

## What is a Heap?

A heap is a **binary tree** with one rule: every parent must satisfy an ordering condition with its children.

- **Max-heap**: parent ≥ children. Biggest value is always at the top.
- **Min-heap**: parent ≤ children. Smallest value is always at the top.

Use a heap when you need to repeatedly grab the min or max value fast.

---

## Shape Rule: Nearly Complete Tree

A heap must be a **nearly complete binary tree**:
- Every level is fully filled except possibly the last.
- The last level fills from **left to right** with no gaps.

```
Valid heap shape:        Invalid heap shape:
       10                      10
      /  \                    /  \
     7    9                  7    9
    / \  /                    \  /
   3   2 8                     2 8   ← left child missing, gap on left
```

This rule is what makes the array trick work. A nearly complete tree maps to a contiguous array with no wasted slots. If gaps were allowed, the index math (`2i+1`, `2i+2`) would break.

It also guarantees the tree height is always **O(log n)** — which is why push and pop are O(log n).

---

## Stored as an Array

No pointers or nodes needed. A heap lives in a plain array — the position tells you the structure.

```
index:   0   1   2   3   4   5   6
value:  [10,  7,  9,  3,  2,  8,  4]

         10
        /  \
       7    9
      / \  / \
     3   2 8   4
```

For any node at index `i`:
- Left child → `2i + 1`
- Right child → `2i + 2`
- Parent → `(i - 1) // 2`

No extra memory. Cache-friendly. Simple math.

---

## Two Core Operations

### Push (insert) — O(log n)

Add the new value at the end, then **bubble up**: swap with parent until the heap rule holds.

```
push(15) into max-heap [10, 7, 9, 3, 2, 8, 4]:

append 15 → [10, 7, 9, 3, 2, 8, 4, 15]   (index 7)
15 > parent(index 3) = 3  → swap
15 > parent(index 1) = 7  → swap
15 > parent(index 0) = 10 → swap
result    → [15, 10, 9, 7, 2, 8, 4, 3]
```

### Pop (remove top) — O(log n)

The top is always index 0. To remove it:
1. Swap index 0 with the last element
2. Remove the last element
3. **Bubble down**: swap with the larger child (max-heap) until the rule holds

```
pop() from max-heap [15, 10, 9, 7, 2, 8, 4]:

swap 15 ↔ 4  → [4, 10, 9, 7, 2, 8, 15]
remove last  → [4, 10, 9, 7, 2, 8]
4 < child 10 → swap → [10, 4, 9, 7, 2, 8]
4 < child 7  → swap → [10, 7, 9, 4, 2, 8]
done
```

Why swap with the last element first? To keep the array contiguous and the tree shape valid.

---

## Build a Heap from an Array — O(n)

Naively pushing n items one by one costs O(n log n). There's a faster way.

Start from the last non-leaf node (`n // 2 - 1`) and bubble down each node toward index 0. Most nodes are near the bottom and barely move — the total work is O(n).

```python
import heapq
nums = [3, 1, 4, 1, 5]
heapq.heapify(nums)   # in-place, O(n)
```

---

## Complexity

| Operation | Time |
|-----------|------|
| Push | O(log n) |
| Pop top | O(log n) |
| Peek top | O(1) |
| Build from array | O(n) |
| Search | O(n) |

Peek is O(1) because the top is always index 0 — no work needed.
Search is O(n) because the heap only guarantees parent vs child order, not left vs right.

---

## Python `heapq`

Python only has a **min-heap**.

```python
import heapq

# build from existing array — in-place, O(n)
arr = [3, 1, 4, 1, 5]
heapq.heapify(arr)

h = []
heapq.heappush(h, 5)
heapq.heappush(h, 1)
heapq.heappush(h, 3)

heapq.heappop(h)   # → 1 (smallest first)
h[0]               # peek without removing
```

**Max-heap trick**: store negated values.

```python
heapq.heappush(h, -5)    # treat as "5 with max priority"
-heapq.heappop(h)        # negate back to get 5
```

**Push then pop in one step** (more efficient than two separate calls):

```python
heapq.heappushpop(h, x)   # push x, then pop and return smallest
heapq.heapreplace(h, x)   # pop smallest first, then push x (faster, but h must be non-empty)
```

---

## Interview Patterns

| Pattern | How heap helps |
|---------|---------------|
| Kth largest element | Min-heap of size k — pop when size exceeds k |
| Top K frequent elements | Count frequencies, then heap on counts |
| Merge K sorted lists | Min-heap of (value, list_index) — always pull the smallest |
| Sliding window median | Two heaps: max-heap (left half) + min-heap (right half) |
| Dijkstra's shortest path | Min-heap on (cost, node) — always expand the cheapest node next |
| Task scheduler | Max-heap on frequency — always schedule the most frequent task first |
