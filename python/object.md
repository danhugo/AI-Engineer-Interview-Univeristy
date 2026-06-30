# What is `object`?

`object` is the root class. Every type in Python is built on top of it. Numbers, strings, lists, your own classes — all inherit from `object`. It's the ancestor of everything.

## So what does `object()` give you?

A plain instance with nothing added. No useful data, no useful methods of its own — just the bare minimum every Python thing has.

```python
x = object()
print(x)  # <object object at 0x7f...>
```

It can't do much:

```python
x = object()
x.name = "hi"  # ERROR — a bare object can't even hold attributes
```

## Why does it exist / when is it useful?

`object()` is used when you need a unique `private marker`, and normal values like `None`, `0`, `""` are not safe (a user might use them as real data).

```python
a = object()
b = object()
a is a    # True
a is b    # False — different objects
```

**Why not** `None` `0` and `""`
```python
a = None
b = None
a is b # True, None is an object

a = 0
b = 0
a is b # often True, but not reliable. Python often caches small interger but for larger integer for example 1000, a is b gives False

# same idea with ""

```

**Main real uses:**

**1. "Nothing was given" marker — when `None` is a valid input.**

```python
_MISSING = object()

def get(data, key, default=_MISSING):
    if default is _MISSING:   # user passed nothing
        raise KeyError(key)
    return default            # user passed something (maybe None)
```

**2. Stop signal for loops.**

```python
STOP = object()

for x in iter(read_next, STOP):   # stop only on this exact marker
    use(x)
```

Same job every time: a value that is 100% yours and can never clash with real data.





## Summary

- `object` = the base class all types come from.
- `object()` = one bare, empty instance of it.
- Main use = a unique marker that can't be confused with anything else.
