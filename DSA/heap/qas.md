# Heap — Q&A

---

## Basic Concept

**Q: What is a heap?**
A: A binary tree where every parent satisfies an ordering rule with its children. Max-heap: parent ≥ children (biggest on top). Min-heap: parent ≤ children (smallest on top).

**Q: When do you use a heap?**
A: When you need to repeatedly get the min or max value fast — like finding the top K items or always processing the highest-priority task next.

**Q: What shape must a heap have?**
A: A nearly complete binary tree. Every level is fully filled except the last, and the last level fills left to right with no gaps. This guarantees height is always O(log n).

**Q: Why must the last level fill left to right?**
A: So the tree maps cleanly to a contiguous array. The index math (`2i+1`, `2i+2`) only works when there are no gaps — a missing left child would break the formula for everything below it.

**Q: How is a heap stored in memory?**
A: As a plain array. No pointers or node objects needed. The index tells you the structure: left child at `2i+1`, right child at `2i+2`, parent at `(i-1)//2`.

**Q: Why store a heap as an array instead of a tree with pointers?**
A: Less memory, better cache performance, and no pointer overhead. The math to find parent/child is simple and fast.

---

## Core Operations

**Q: How does push work and what is its time complexity?**
A: Append the new value at the end of the array, then bubble up — swap with parent repeatedly until the heap rule holds. O(log n) because the tree height is log n.

**Q: How does pop work and what is its time complexity?**
A: Swap the top (index 0) with the last element, remove the last element, then bubble down — swap with the larger child (max-heap) until the rule holds. O(log n).

**Q: Why swap with the last element when popping instead of just removing the top?**
A: Removing index 0 directly would leave a gap. Swapping with the last element keeps the array contiguous and the tree shape valid, then bubbling down restores the order.

**Q: How do you peek at the top without removing it?**
A: Read index 0. O(1) — the top is always there.

**Q: Why is peek O(1)?**
A: The heap always keeps the min (or max) at index 0. No searching needed.

**Q: Why is search O(n) in a heap?**
A: The heap only guarantees parent vs child order. There's no rule about left vs right siblings, so you can't skip any part of the array when searching.

---

## Build Heap

**Q: What is heapify and why is it O(n) instead of O(n log n)?**
A: Heapify builds a heap from an existing array in-place. It starts from the last non-leaf node and bubbles down toward index 0. Most nodes are near the bottom and barely move, so total work is O(n) — much less than pushing n items one by one.

**Q: Why is heapify O(n) and not O(n log n)?**
A: Most nodes are near the bottom and do almost no work. Leaves (half the nodes) do 0 swaps. The level above does 1 swap each. Only the few nodes near the top do log n swaps. Total work = `n/2×0 + n/4×1 + n/8×2 + …` which sums to 2n → O(n).

**Q: What index do you start from when building a heap bottom-up?**
A: `n // 2 - 1` — the last non-leaf node. Leaf nodes don't need bubbling down since they have no children.

---

## Python `heapq`

**Q: How do you build a min-heap from an existing array in Python?**
A: Call `heapq.heapify(arr)` — it converts the list in-place in O(n).

**Q: What kind of heap does Python's `heapq` give you?**
A: Min-heap only. The smallest value is always at index 0.

**Q: How do you simulate a max-heap with `heapq`?**
A: Store negated values. Push `-x` instead of `x`, then negate the result when you pop: `-heapq.heappop(h)`.

**Q: What is the difference between `heappushpop` and `heapreplace`?**
A: Both push one value and pop one. `heappushpop` pushes first then pops — safe on an empty heap. `heapreplace` pops first then pushes — faster, but the heap must be non-empty.

---

## Complexity Summary

**Q: What are the time complexities for heap operations?**
A: Push O(log n), pop O(log n), peek O(1), build from array O(n), search O(n).

---

## Interview Patterns

**Q: How do you find the Kth largest element using a heap?**
A: Use a min-heap of size k. Push each element; when size exceeds k, pop the smallest. At the end, the top of the heap is the Kth largest. O(n log k).

**Q: How do you merge K sorted lists using a heap?**
A: Push the first element of each list into a min-heap along with its list index. Each time you pop the smallest, push the next element from that same list. O(n log k) where n is total elements.

**Q: How do you find the sliding window median using two heaps?**
A: Keep a max-heap for the left half and a min-heap for the right half. Balance them so sizes differ by at most 1. The median is the top of the larger heap (or average of both tops if equal size).

**Q: Why does Dijkstra's algorithm use a min-heap?**
A: You always want to expand the node with the smallest known cost next. A min-heap gives you that node in O(log n) instead of scanning all nodes in O(n).
