# Array — Q&A

---

## Basic Concept

**Q: What is an array?**
A: A contiguous block of memory where all elements sit side by side with no gaps. This layout lets you access any element in O(1) using address math: `start + i × element_size`.

**Q: Why is index access O(1)?**
A: The computer computes the exact memory address directly — no searching. It just multiplies the index by the element size and adds the starting address.

**Q: What's the difference between how C and Python store array elements?**
A: A C array stores the actual values side by side in memory (one jump to access). A Python list stores pointers to objects — the objects live elsewhere in the heap (two jumps: read pointer, then follow it).

**Q: Why can Python lists hold mixed types but C arrays cannot?**
A: Python lists store pointers, and every pointer is the same size regardless of what it points to. C arrays store raw values directly, so the compiler needs a fixed type and size for each slot.

---

## Static vs Dynamic Arrays

**Q: What is a static array?**
A: A fixed-size array allocated at creation. It cannot grow or shrink. If you need more space, you must allocate a new array and copy everything over manually.

**Q: What is a dynamic array?**
A: An array that grows automatically. When it fills up, it allocates a larger block (~2×), copies all old items over, and frees the old block. Examples: Python `list`, Java `ArrayList`, C++ `std::vector`, Go `slice`.

**Q: What triggers a resize in a dynamic array?**
A: When `size == capacity` and you try to append. The array allocates a new block about twice as large, copies all elements, then frees the old block.

---

## Python `list` Internals

**Q: What is the difference between size and capacity in a Python list?**
A: `size` is how many elements are actually stored (`len(list)`). `capacity` is how many slots are currently allocated. Extra slots are kept in reserve so most appends don't need a resize. `size` is always ≤ `capacity`.

**Q: What is Python's list growth factor?**
A: Approximately 2×. The exact CPython formula is `new_capacity = old_capacity + (old_capacity >> 3) + (6 if old_capacity < 9 else 0)`.

---

## Operations

**Q: What is the time complexity of `arr[i]`?**
A: O(1). Direct address calculation — no searching.

**Q: What is the time complexity of `arr.append(x)` and why?**
A: O(1) amortised. Most appends just write to the next free slot. Occasionally a resize copies all n elements (O(n)), but this happens rarely enough that the average cost per append is O(1).

**Q: What is the time complexity of `arr.pop()` and why?**
A: O(1) amortised. Just decrements the size counter. Optionally shrinks capacity when `size ≤ capacity // 4`, but this is rare.

**Q: What is the time complexity of `arr.insert(i, x)` and why?**
A: O(n). All elements from index i onward must shift right by one position to make room.

**Q: What is the time complexity of `del arr[i]` and why?**
A: O(n). All elements from index `i+1` onward must shift left by one position to fill the gap.

**Q: What is the time complexity of `x in arr`?**
A: O(n). Linear scan from the start; stops on the first match.

**Q: What is the time complexity of `arr[i:j]`?**
A: O(k) where k = j − i. Allocates a new list and copies k pointers from the source range.

**Q: What is the time complexity of `len(arr)`?**
A: O(1). Python stores the count as a field — no counting needed.

---

## Amortised Analysis

**Q: What does "amortised O(1)" mean for append?**
A: Individual appends may cost O(n) during a resize, but if you average the cost across all n appends, each one costs O(1) on average.

**Q: Why does doubling keep amortised cost O(1) for append?**
A: After n appends, total resize cost = 1 + 2 + 4 + … ≤ 2n → O(n) total. Spread over n appends = O(1) each. The key: each element is copied at most once per doubling, and doublings become rarer as the array grows.

**Q: Why not grow by a fixed amount (e.g. +10) instead of doubling?**
A: Fixed growth triggers O(n) resizes every 10 inserts — O(n²) total. Doubling makes resize events exponentially less frequent, keeping total cost O(n) and amortised cost O(1).
