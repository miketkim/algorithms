# Algorithms and Data Structures

## Searching Algorithms

### Binary Search

**Code (Recursive)**

```python
def binary_search(items, start, end, target):
    if end >= start:
        middle = start + (end - start) // 2
        if items[middle] == target:
            return middle
        elif items[middle] > target:
            return binary_search(items, start, middle - 1, target)
        else:
            return binary_search(items, middle + 1, end, target)
    else:
        return -1
```

**Code (Iterative)**

```python
def binary_search(items, start, end, target):
    while start <= end:
        middle = start + (end - start) // 2
        if items[middle] == target:
            return middle
        elif items[middle] > target:
            start = middle + 1
        else:
            end = middle - 1
    else:
        return -1
```
