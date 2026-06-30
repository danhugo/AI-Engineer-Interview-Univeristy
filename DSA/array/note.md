# Array — Interview Knowledge Sheet

## What is an Array?

An array is a **contiguous block of memory** — all elements sit next to each other, no gaps.
This is what makes index access O(1): the computer calculates the address with simple math instead of searching.

```
address of arr[i] = start_address + i × element_size
```

---

## Static vs Dynamic Typed Languages

How arrays work depends on the language:

**Statically typed (C, Java, C++)** — the array stores the actual values side by side.

```c
int arr[4] = {1, 2, 3, 4};
// 16 bytes in memory: [1][2][3][4]
```

To get `arr[i]`: one jump — start + i × 4. Very fast.

**Dynamically typed (Python, JavaScript)** — the array stores *pointers* to objects, not the objects themselves.

```python
arr = [1, 2, 3, 4]
# memory: [ptr→1][ptr→2][ptr→3][ptr→4]
# each pointer leads to an object somewhere else in memory
```

To get `arr[i]`: two jumps — read pointer, then follow it. This is why Python lists can hold mixed types (`[10, "hi", 3.14, True]`) — every slot is just a pointer regardless of what it points to.

![C Array vs Python List in memory mapping](c_array_vs_python_list_white_bg.png)

---

## Static Array vs Dynamic Array

**Static array** — fixed size set at creation. Cannot grow or shrink.

```c
int arr[5];   // always 5 slots, no more
```

**Dynamic array** — grows automatically as you add items. Under the hood it's still a contiguous block, but when it gets full it:
1. Allocates a new block (~2× bigger)
2. Copies all old items over
3. Frees the old block

```python
arr = []
arr.append(1)   # grows automatically
arr.append(2)
```

Examples: Python `list`, Java `ArrayList`, C++ `std::vector`, Go `slice`.

---

## Python `list` Internals

Python `list` is a dynamic array with two sizes:
- **capacity** — how many slots are currently allocated
- **size** — how many elements are actually stored (`len(list)`)

`size` is always ≤ `capacity`. Extra slots are reserved so most appends don't need a resize.

**Growth factor**: ~2× in CPython.
Precise formula: `new_capacity = old_capacity + (old_capacity >> 3) + (6 if old_capacity < 9 else 0)`

---

## Operations & Complexity

| Operation | Code | Time |
|-----------|------|------|
| Index access | `arr[i]` | O(1) |
| Append | `arr.append(x)` | O(1) amortised |
| Pop from end | `arr.pop()` | O(1) amortised |
| Insert at i | `arr.insert(i, x)` | O(n) |
| Delete at i | `del arr[i]` | O(n) |
| Search | `x in arr` | O(n) |
| Slice | `arr[i:j]` | O(k) |
| Length | `len(arr)` | O(1) |

### Index access — O(1)
Direct address math: `start + i × element_size`. Just a bounds check, then one memory read.

### Append — O(1) amortised
- If `size < capacity`: write at `arr[size]`, increment size. O(1).
- If `size == capacity`: resize (~2×) + copy everything → O(n), then write. Rare, averages to O(1) per append.

### Pop from end — O(1) amortised
Decrement size and return the element. Optionally shrink capacity when `size ≤ capacity // 4` to avoid wasting memory (but never below size).

### Insert at i — O(n)
Resize if needed, then shift all elements from index i onward one position right, then write the new value at i.

### Delete at i — O(n)
Shift all elements from `i+1` onward one position left, then decrement size.

### Search — O(n)
Linear scan from the start. Returns on the first match.

### Slice `arr[i:j]` — O(k) where k = j − i
Allocates a new list of length k and copies k references (pointers) from the source range.

### Length — O(1)
`len()` reads a stored counter — no counting needed.

---

## Amortised O(1) — Why Append is Fast on Average

Imagine starting with capacity 1 and doubling each time:

```
capacity 1  → 2  → 4  → 8  → 16 ...
resize cost    1    2    4    8   ...
```

After n appends, total resize cost = 1 + 2 + 4 + … ≤ 2n → O(n) total.
Spread over n appends = **O(1) per append**.

The key insight: each element is copied at most once per doubling, and doublings get rarer as the array grows.
