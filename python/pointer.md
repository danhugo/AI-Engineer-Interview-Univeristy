In Python, a variable is a pointer, not an object. Many pointers (name) can point to the same object.

### 1. `b = a` does NOT copy 
```python
a = [1,2,3]
b = a

b.append(4) # a = [1,2,3,4]
```

To get a real copy: `b = a.copy()` or `b = a[:]` or `b = list(a)`.

### 2. The `[x] * 3` trap - `x` is mutable object

This code does:
- create `x`
- create 3 pointers point to `x`
  
**BUG** only appears for MUTABLE objects (list, dict, set, your class), because only those can be changed in place. Immutable objects (int, str, tuple) cannot be changed in place.

```python
a = [[]] * 3
a[0].append[1] # a = [[1], [1], [1]]

# class C() with attribute C.x
a = [C()] * 3
a[0].x = 99 # a[0].x, a[1].x, a[2].x = 99, 99, 99

# TOFIX:
a = [[] for _ in range(3)] # create 3 different object []
a[0].append[1] # a = [[1], [], []]

# immutable object
a = [1000] * 3
a[0] = 5 # a = [5, 1000, 1000]. change a[0] = 5 re-point the pointer at slot 0 to other object.

a = ['str'] * 3
a[0] = 's' # a = ['s', 'str', 'str']
```

### 3. Shallow copy - deep copy

```python
a = [[1, 2], [3, 4]] # a contains 2 pointers p1, p2 pointing to list [1, 2] & [3, 4]
b = a.copy()         # other new object b copying 2 pointers p1, p2
# same with b = a[:] or b = list(a)
b[0].append(99)
print(a)             # [[1, 2, 99], [3, 4]]  <- inner still shared!

# TOFIX:
import copy; b = copy.deepcopy(a)
```

### 4. Mutable default argument

```python
def add(item, box=[]):    # box created ONCE, reused forever
    box.append(item)
    return box

add(1)    # [1]
add(2)    # [1, 2]  <- not [2]! same list every call
```

The default `[]` is made one time, not each call. Fix: use `box=None`, then `if box is None: box = []` inside.

### 5. `is` vs `==`

```python
a = [1, 2]
b = [1, 2]
c = b
a == b     # True  (same value)
a is b     # False (different objects)
c is b     # True  (same object)
```

