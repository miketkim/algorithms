# Data Structures

## Python `set()`

### Time Efficiency

| Operation   | Running Time |
|-------------|--------------|
| len(set)    | O(1)         |
| item in set | O(n)         |
| get item    | O(n)         |
| set item    | O(n)         |
| iteration   | O(n)         |

## Singly Linked List

```python
class Node:
    def __init__(self, element, next_node):
        self.element = element
        self.next_node = next_node
```

```python
class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def __len__(self):
        return self.size

    def is_empty(self):
        return self.size == 0

    def add_first(self, element):
        node = Node(element, self.head)
        self.head = node
        self.size += 1

    def add_last(self, element):
        node = Node(element, None)
        self.tail.next_node = node
        self.tail = node
        self.size += 1

    def remove_first(self):
        if self.is_empty():
            raise Exception('This list is empty.')
        element = self.head.element
        self.head = self.head.next_node
        self.size -= 1
        return element
```

## Stack (Singly Linked List Implementation)

```python
class LinkedStack:
    def __init__(self):
        self.head = None
        self.size = 0

    def __len__(self):
        return self.size

    def is_empty(self):
        return self.size == 0

    def push(self, element):
        self.head = Node(element, self.head)
        self.size += 1

    def top(self):
        if self.is_empty():
            raise Exception('This stack is empty.')
        return self.head.element

    def pop(self):
        if self.is_empty():
            raise Exception('This stack is empty.')
        element = self.head.element
        self.head = self.head.next_node
        self.size -= 1
        return element
```

### Time Efficiency

| Operation        | Running Time |
|------------------|--------------|
| len(stack)       | O(1)         |
| stack.is_empty() | O(1)         |
| stack.push()     | O(1)         |
| stack.top()      | O(1)         |
| stack.pop()      | O(1)         |

## Queue (Singly Linked List Implementation)

```python
class LinkedQueue:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def __len__(self):
        return self.size

    def is_empty(self):
        return self.size == 0

    def first(self):
        if self.is_empty():
            raise Exception('This queue is empty.')
        return self.head.element

    def dequeue(self):
        if self.is_empty():
            raise Exception('This queue is empty.')
        element = self.head.element
        self.head = self.head.next_node
        self.size -= 1
        if self.is_empty():
            self.tail = None
        return element

    def enqueue(self, element):
        node = Node(element, None)
        if self.is_empty():
            self.head = node
        else:
            self.tail.next_node = node
        self.tail = node
        self.size += 1
```

### Time Efficiency

| Operation        | Running Time |
|------------------|--------------|
| len(queue)       | O(1)         |
| queue.is_empty() | O(1)         |
| queue.enqueue()  | O(1)         |
| queue.first()    | O(1)         |
| queue.dequeue()  | O(1)         |

## Queue (Circularly Linked List Implementation)

```python
class CircularQueue:
    def __init__(self):
        self.tail = None
        self.size = 0

    def __len__(self):
        return self.size

    def is_empty(self):
        return self.size == 0

    def first(self):
        if self.is_empty():
            raise Exception('This queue is empty.')
        head = self.tail.next_node
        return head.element

    def dequeue(self):
        if self.is_empty():
            raise Exception('This queue is empty.')
        old_head = self.tail.next_node
        if self.size == 1:
            self.tail = None
        else:
            self.tail.next_node = old_head.next_node
        self.size -= 1
        return old_head.element

    def enqueue(self, element):
        node = Node(element, None)
        if self.is_empty():
            node.next_node = node
        else:
            node.next_node = self.tail.next_node
            self.tail.next_node = node
        self.tail = node
        self.size += 1

    def rotate(self):
        if self.size > 0:
            self.tail = self.tail.next_node
```

### Problems

#### Detect Cycle

```python
def has_cycle(head):
    """Time efficiency O(n2); Auxiliary space O(n)"""
    visited = set()
    node = head
    while node is not None:
        if node in visited:
            return True
        visited.add(node)
        node = node.next
    return False
```

```python
def has_cycle(head):
    """Time efficiency O(n); Auxiliary space O(n)"""
    node = head
    while node is not None:
        if hasattr(node, 'visited'):
            return True
        setattr(node, 'visited', True)
        node = node.next
    return False
```

```python
def has_cycle(head):
    """Time efficiency O(n); Auxiliary space O(1)"""
    slow_pointer = head
    fast_pointer = head
    while slow_pointer and fast_pointer and fast_pointer.next_node:
        slow_pointer = slow_pointer.next_node
        fast_pointer = fast_pointer.next_node.next_node
        if slow_pointer == fast_pointer:
            return True
    return False
```

## Doubly Linked List

```python
class Node:
    def __init__(self, element, prev_node, next_node):
        self.element = element
        self.prev_node = prev_node
        self.next_node = next_node
```

```python
class DoublyLinkedList:
    def __init__(self):
        self.head = Node(None, None, None)
        self.tail = Node(None, None, None)
        self.head.next_node = self.tail
        self.tail.prev_node = self.head
        self.size = 0

    def __len__(self):
        return self.size

    def is_empty(self):
        return self.size == 0

    def insert(self, element, prev_node, next_node):
        node = Node(element, prev_node, next_node)
        prev_node.next_node = node
        next_node.prev_node = node
        self.size += 1
        return node

    def remove(self, node):
        prev_node = node.prev_node
        next_node = node.next_node
        prev_node.next_node = next_node
        next_node.prev_node = prev_node
        self.size -= 1
        element = node.element
        node.prev_node = None
        node.next_node = None
        node.element = None
        return element
```

## Deque (Doubly Linked List Implementation)

```python
class LinkedDeque(DoublyLinkedList):
    def first(self):
        if self.is_empty():
            raise Exception('This deque is empty.')
        return self.head.next_node.element

    def last(self):
        if self.is_empty():
            raise Exception('This deque is empty.')
        return self.tail.prev_node.element

    def insert_first(self, element):
        self._insert(element, self.head, self.head.next_node)

    def insert_last(self, element):
        self._insert(element, self.tail.prev_node, self.tail)

    def remove_first(self):
        if self.is_empty():
            raise Exception('This deque is empty.')
        return self.remove(self.head.next_node)

    def remove_last(self):
        if self.is_empty():
            raise Exception('This deque is empty.')
        return self.remove(self.tail.prev_node)
```

## Tree

### Binary Tree

**Properties:**
- The maximum number of nodes at level *L* of a binary tree is 2<sup>L-1</sup>. The root is at level *L* = 1.
- The maximum number of nodes in a binary tree of height *H* is 2<sup>H-1</sup>. A binary tree with a single node (the root) is of height *H* = 1.
- In a binary tree with *N* nodes, the minimum possible height or minimum number of leaf nodes is log<sub>2</sub>(*N*+1).
- A binary tree with *L* leaves has at least log<sub>2</sub>*L* + 1 levels.
- In a binary tree, the number of leaf nodes is always one more than the number of nodes with two children.

**Types of Binary Trees:**
- Full Binary Tree--Every node has 0 or 2 children.
- Complete Binary Tree--All levels are filled except maybe the last level. The last level has all nodes as left as possible.
- Perfect Binary Tree--All internal nodes have 2 children and all leaf nodes are on the same level.
- Balanced Binary Tree--Tree height is O(log n).
- Pathological Binary Tree--Every internal node has one child. This is essentially a linked list.

```python

```

### Binary Search Tree

A node-based binary tree data structure with the following properties:
- the left subtree of a node only contains nodes with data less than the node's data
- the right subtree of a node only contains nodes with data greater than the node's data
- the left and right subtrees must also be binary search trees
- there must be no duplicate nodes

*Facts:*
- in-order traversal of binary search tree yields sorted values
- can do binary search

```python
class Node:
    def __init__(self, element, left=None, right=None):
        self.element = element
        self.left = left
        self.right = right
```

```python
def depth(node):
    if node.parent is None:
        return 0
    else:
        return 1 + depth(node.parent)

def height(node):
    if node.left is None and node.right is None:
        return 0
    if node.left and node.right:
        return 1 + max(height(node.left), height(node.right))
    if node.left:
        return 1 + height(node.left)
    if node.right:
        return 1 + height(node.right)

# Preorder traversal.
# Executes in O(n) time.
def preorder(node, memo):
    memo.append(node.element)
    if node.left:
        preorder(node.left, memo)
    if node.right:
        preorder(node.right, memo)

def preorder_iterative(node):
    stack = [node]
    while stack:
        node = stack.pop()
        yield node.element
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

# Postorder traversal.
# Executes in O(n) time.
def postorder(node, memo):
    if node.left:
        postorder(node.left, memo)
    if node.right:
        postorder(node.right, memo)
    memo.append(node.element)

def postorder_iterative(node):
    out = []
    stack = [node]
    while stack:
        node = stack.pop()
        out.append(node.element)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    while out:
        yield out.pop()

# Inorder traversal.
# Executes in O(n) time.
def inorder(node, memo):
    if node.left:
        inorder(node.left, memo)
    memo.append(node.element)
    if node.right:
        inorder(node.right, memo)

def inorder_iterative(root):
    stack = []
    while stack or node:
        if node:
            stack.append(node)
            node = node.left
        else:
            node = stack.pop()
            yield node.element
            node = node.right

# Breadthfirst traversal.
# Executes in O(n) time.
def breadthfirst(node):
    queue = [node]
    while queue:
        node = queue.pop(0)
        yield node.element
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
```

See [Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view)  
See [Average of Levels in Binary Tree](https://leetcode.com/problems/average-of-levels-in-binary-tree/)  
See [Find Largest Value in Each Tree Row](https://leetcode.com/problems/find-largest-value-in-each-tree-row/)  
See [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)  
See [Check Completeness of a Binary Tree](https://leetcode.com/problems/check-completeness-of-a-binary-tree/)  
See [Binary Tree Level Traversal II](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)  
See [Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)  
See [Maximum Level Sum](https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/)  

```python
def levelorder(root):
    # If you need to return something, then instantiate the data structure here.
    # rightmost_values = list()
    # level_sums = list()
    # levels = list()
    
    queue = deque([root])
    while queue:
        count = len(queue)
        while count > 0:
            root = queue.popleft()
            
            # Do something with the value here.
            # val = root.val
            # If you want to find right-side view of tree, then do:
            # rightmost_value = root.val
            # If you want to find the sum of each row, then do:
            # level_sum += root.val
            # If you want to store the values in order, then do:
            # level.append(root.val)
            
            if root.left:
                queue.append(root.left)
            if root.right:
                queue.append(root.right)
            count -= 1
            
        # If you need to build a data structure to return, then do it here.
        # rightmost_values.append(rightmost_value)
        # level_sums.append(level_sum)
        # levels.append(level)
        
    # Return any data structure.
    # return rightmost_values
    # return level_sums
    # return levels
            

# Search binary search tree.
# Executes in O(h) time.
def search_bst(root, element):
    if root is None or root.element == element:
        return root
    if element < root.element:
        return search_bst(root.left, element)
    return search_bst(root.right, element)


# Insert element into binary search tree.
# Executes in O(h) time.
def insert_bst(root, element):
    if element < root.element:
        if root.left is None:
            root.left = Node(element)
        else:
            insert_bst(root.left, element)
    else:
        if root.right is None:
            root.right = Node(element)
        else:
            insert_bst(root.right, element)


# Delete node from binary search tree.
# Executes in O(h) time.
def delete_bst(root, element):
    if root is None:
        return None
    if element < root.element:
        root.left = delete_bst(root.left, element)
    elif element > root.element:
        root.right = delete_bst(root.right, element)
    else:
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left
        node = root.right
        while node.left is not None:
            node = node.left
        root.element = node.element
        root.right = delete_bst(root.right, node.element)
    return root
```

## Heap

```python
class Heap:
    def __init__(self, elements=None):
        self.heap = elements or []
        if elements:
            self.heapify()

    def heapify(self):
        start = self.parent(len(self) - 1)
        for j in range(start, -1, -1):
            self.bubble_down(j)

    def __len__(self):
        return len(self.heap)

    def is_empty(self):
        return len(self.heap) == 0

    def parent(self, j):
        return (j - 1) // 2

    def left(self, j):
        return 2 * j + 1

    def right(self, j):
        return 2 * j + 2

    def has_left(self, j):
        return self.left(j) < len(self.heap)

    def has_right(self, j):
        return self.right(j) < len(self.heap)

    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def add(self, element):
        self.heap.append(element)
        self.bubble_up(len(self.heap) - 1)

    def min(self):
        if self.is_empty():
            raise Exception('This heap is empty.')
        element = self.heap[0]
        return element

    def remove_min(self):
        if self.is_empty():
            raise Exception('This heap is empty.')
        self.swap(0, len(self.heap) - 1)
        element = self.heap.pop()
        self.bubble_down(0)
        return element

    def bubble_up(self, index):
        parent = self.parent(index)
        if index > 0 and self.heap[index] < self.heap[parent]:
            self.swap(index, parent)
            self.bubble_up(index)

    def bubble_down(self, index):
        if self.has_left(index):
            left = self.left(index)
            small_child = left
            if self.has_right(index):
                right = self.right(index)
                if self.heap[right] < self.head[left]:
                    small_child = right
                if self.heap[small_child] < self.heap[index]:
                    self.swap(index, small_child)
                    self.bubble_down(small_child)
```

### Time Efficiency

| Operation         | Running Time |
|-------------------|--------------|
| len(heap)         | O(1)         |
| heap.is_empty()   | O(1)         |
| heap.min()        | O(1)         |
| heap.add()        | O(log n)     |
| heap.remove_min() | O(log n)     |
| heap.heapify()    | O(n)         |

# Sorting

## Insertion Sort

### Discussion

Although it is one of the elementary sorting algorithms with O(n²) worst-case time, insertion sort is the algorithm of choice either when the data is nearly sorted (because it is adaptive) or when the problem size is small (because it has low overhead).

For these reasons, and because it is also stable, insertion sort is often used as the recursive base case (when the problem size is small) for higher overhead divide-and-conquer sorting algorithms, such as merge sort or quick sort.

### Properties

* Stable
* O(1) extra space
* O(n²) comparisons and swaps
* Adaptive: O(n) time when nearly sorted
* Very low overhead

### Algorithm

```python
def insertion_sort(array):
    for i in range(1, len(array)):
        current = array[i]
        j = i
        while j > 0 and array[j - 1] > current:
            array[j] = array[j - 1]
            j -= 1
        array[j] = current
```

## Selection Sort

### Discussion

From the comparions presented here, one might conclude that selection sort should never be used. It does not adapt to the data in any way (notice that the four animations above run in lock step), so its runtime is always quadratic.

However, selection sort has the property of minimizing the number of swaps. In applications where the cost of swapping items is high, selection sort very well may be the algorithm of choice.

### Properties

* Not stable
* O(1) extra space
* Θ(n²) comparisons
* Θ(n) swaps
* Not adaptive

### Algorithm

```python
def selection_sort(array):
    for i in range(len(array) - 1, 0, -1):
        pointer = 0
        for j in range(1, i + 1):
            if array[j] > array[pointer]:
                pointer = j
        array[i], array[pointer] = array[pointer], array[i]
```

## Bubble Sort

### Discussion

Bubble sort has many of the same properties as insertion sort, but has slightly higher overhead. In the case of nearly sorted data, bubble sort takes O(n) time, but requires at least 2 passes through the data (whereas insertion sort requires something more like 1 pass).

### Properties

* Stable
* O(1) extra space
* O(n²) comparisons and swaps
* Adaptive: O(n) when nearly sorted

### Algorithm

```python
def bubble_sort(array):
    for i in range(len(array) - 1, 0, -1):
        for j in range(i):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
```

## Merge Sort

### Discussion

Merge sort is very predictable. It makes between 0.5lg(n) and lg(n) comparisons per element, and between lg(n) and 1.5lg(n) swaps per element. The minima are achieved for already sorted data; the maxima are achieved, on average, for random data. If using Θ(n) extra space is of no concern, then merge sort is an excellent choice: It is simple to implement, and it is the only stable O(n·lg(n)) sorting algorithm. Note that when sorting linked lists, merge sort requires only Θ(lg(n)) extra space (for recursion).

Merge sort is the algorithm of choice for a variety of situations: when stability is required, when sorting linked lists, and when random access is much more expensive than sequential access (for example, external sorting on tape).

There do exist linear time in-place merge algorithms for the last step of the algorithm, but they are both expensive and complex. The complexity is justified for applications such as external sorting when Θ(n) extra space is not available.

### Properties

* Stable
* Θ(n) extra space for arrays (as shown)
* Θ(lg(n)) extra space for linked lists
* Θ(n·lg(n)) time
* Not adaptive
* Does not require random access to data

### Algorithm

```python
def merge(array1, array2, array):
    i = 0
    j = 0
    while i + j < len(array):
        if i == len(array1):
            array[i + j] = array2[j]
            j += 1
        elif j == len(array2):
            array[i + j] = array1[i]
            i += 1
        elif array1[i] < array2[j]:
            array[i + j] = array1[i]
            i += 1
        else:
            array[i + j] = array2[j]
            j += 1
        # if j == len(array2) or (i < len(array1) and array1[i] < array2[j]):
        #     array[i + j] = array1[i]
        #     i += 1
        # else:
        #     array[i + j] = array2[j]
        #     j += 1


def merge_sort(array):
    n = len(array)
    if n < 2:
        return

    # Divide.
    mid = n // 2
    array1 = array[0 : mid]
    array2 = array[mid : n]

    # Conquer.
    merge_sort(array1)
    merge_sort(array2)

    # Merge.
    merge(array1, array2, array)
```

## Quick Sort

### Discussion

The 3-way partition variation of quick sort has slightly higher overhead compared to the standard 2-way partition version. Both have the same best, typical, and worst case time bounds, but this version is highly adaptive in the very common case of sorting with few unique keys.

The 3-way partitioning code shown above is written for clarity rather than optimal performance; it exhibits poor locality, and performs more swaps than necessary. A more efficient but more elaborate 3-way partitioning method is given in Quicksort is Optimal by Robert Sedgewick and Jon Bentley.

When stability is not required, quick sort is the general purpose sorting algorithm of choice. Recently, a novel dual-pivot variant of 3-way partitioning has been discovered that beats the single-pivot 3-way partitioning method both in theory and in practice.

### Properties

* Not stable
* O(lg(n)) extra space
* O(n²) time, but typically O(n·lg(n)) time
* Adaptive: O(n) time when O(1) unique keys

### Algorithm

```python
def inplace_quick_sort(S, a, b):
    if a >= b: 
        return 
    pivot = S[b]
    left = a
    right = b - 1
    while left <= right:
        while left <= right and S[left] < pivot: 
            left += 1
        while left <= right and pivot < S[right]: 
            right -= 1
        if left <= right:
            S[left], S[right] = S[right], S[left] 
            left, right = left + 1, right - 1
    S[left], S[b] = S[b], S[left]
    inplace_quick_sort(S, a, left - 1) 
    inplace_quick_sort(S, left + 1, b)
```

# Dynamic Programming

## Longest Common Subsequence

### Stage 1: Creating the initial matrix.

Steps:
1. Create an empty matrix (a two-dimensional array of zeroes). The width of the matrix should be one more than the number of characters in `s2`. The height of the matrix should be one more than the number of characters in `s1`.
2. Starting with the first character in `s1`, compare every character in `s2` to every character in `s1`. At the same time, start with the cell in the matrix that is one row down and one row to the right. If the characters are equal between the two strings, set the value of the current cell to the value of the top-left cell plus 1.
3. If the characters do not match, set the value of the current cell to whichever value is larger: the value of the cell one column up or the value of the cell one column to the left.

```python
def longest_common_subsequence(s1, s2):
    num_rows, num_cols = len(s1), len(s2)
    matrix = [[0] * (num_cols + 1) for row in range(num_rows + 1)]

    for row in range(num_rows):
        for col in range(num_cols):
            if s1[row] == s2[col]:
                matrix[row + 1][col + 1] = matrix[row][col] + 1
            else:
                matrix[row + 1][col + 1] = max(matrix[row][col + 1],
                                               matrix[row + 1][col])

    return matrix
```

### Stage 2: Using the matrix to find the solution.

Steps:
1. Starting from the bottom-right corner, compare the characters in each string.
2. If the characters are equal, append the character to the solution list, and then move one cell up and one cell to the left.
3. If the characters are not equal, then compare the value of the cell above to the value of the cell to the left.
4. If the value of the cell above is greater than or equal to the value of the cell to the left, then move up one cell.
5. If the value of the cell above is less than the value of the cell to the left, then move left one cell.
6. Keep comparing and moving until the value of the current cell is 0.

```python
def longest_common_subsequence_solution(s1, s2, matrix):
    solution = []
    row, col = len(s1), len(s2)

    while matrix[row][col] > 0:
        if s1[row - 1] == s2[col - 1]:
            solution.append(s1[row - 1])
            row -= 1
            col -= 1
        elif matrix[row - 1][col] >= matrix[row][col - 1]:
            row -= 1
        else:
            col -= 1

    return ''.join(reversed(solution))
```

# Graph

## Searches

```python
from collections import defaultdict


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, vertex1, vertex2):
        self.graph[vertex1].append(vertex2)

    # Time complexity O(V + E)
    def dfs_iterative(self, start):
        visited = set()
        stack = [start]
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                for adjacent_vertex in self.graph[vertex]:
                    if adjacent_vertex not in visited:
                        stack.append(adjacent_vertex)
        return visited

    # Time complexity O(V + E)
    def dfs_recursive(self, start, visited):
        visited.add(start)
        yield start
        for adjacent_vertex in self.graph[start]:
            if adjacent_vertex not in visited:
                self.dfs_recursive(adjacent_vertex, visited)

    # Time complexity O(V + E)
    def bfs(self, start):
        visited = set()
        queue = [start]
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                for adjacent_vertex in self.graph[vertex]:
                    if adjacent_vertex not in visited:
                        queue.append(adjacent_vertex)
        return visited
```

See [Keys and Rooms](https://leetcode.com/problems/keys-and-rooms/)

```python
def all_vertices_visited(start, edges):
    graph = Graph()
    vertices = set()
    for edge in edges:
        u, v = edge
        vertices.add(u)
        vertices.add(v)
        graph.add_edge(u, v)
    visited = graph.bfs(start)
    return len(visited) == len(vertices)
```

## Trends

### Subarray

**Examples:**
1. Given an array and a key, find the minimum subarray whose sum equals the key.
2. Find the maximum sum of a subarray from a positive integer array where any two numbers of the subarray are not adjacent to each other in the original array.
3. Given an array, write all the continuous subarrays in the array whose sums equal 0.
4. Given an array, write a function to write all triplets in the array whose sums equal 0.
5. Given an array, find the continuous subarray with the greatest sum.
6. Given an array of *n* elements and an integer *k*, write a function that returns whether the sum of any two elements in the array equals *k*.

**Look For:**
- Continuous or non-continuous subarray?
- Positive integers only or both positive and negative integers?
- One subarray or a number (all) of them?
- Target size or no size constraints?

**Techniques:**
1. Use a hash table to store values and do math to find if missing value exists.
2. Nested for-loops and O(n2) time are almost certainly not the most efficient way to solve a problem.

```python
def subarray_sum(array, target):
    """Brute force solution: try every possible continuous subarray."""
    length = len(array)
    for i in range(length):
        current_sum = array[i]
        for j in range(i + 1, length):
            if current_sum == target:
                return array[i:j]
            if current_sum > target or j == length:
                break
            current_sum += array[j]
    return None
```

```python
def subarray_sum(array, target):
    """Efficient solution (non-negative numbers): """
    start = 0
    current_sum = array[start]
    for i in range(1, len(array)):
        while current_sum > target and start < i - 1:
            current_sum -= array[start]
            start += 1
        if current_sum == target:
            return array[start:i]
        if i < len(array):
            current_sum += array[i]
    return None
```

```python
def subarray_sum(array, target):
    """Efficient solution (negative numbers): """
    index_by_sum = dict()
    current_sum = 0
    for i in range(len(array)):
        current_sum += array[i]
        if current_sum == target:
            return array[:i + 1]
        index = index_by_sum.get(current_sum - target)
        if index is not None:
            return array[index + 1: i + 1]
        index_by_sum[current_sum] = i
    return None
```

### Substring

**Examples:**
1. *Given a string and a pattern (with wildcards), find the first substring matching the pattern.*
2. Print all permutations of a given string.
3. Print all anagrams in buckets.
4. *Given a string, check if it has a substring.*
5. Print all unique subsets of a string.
6. Determine if two strings are one edit apart.
7. You are given a set of unique characters and a string. Find the smallest substring that uses all of the characters in the set.

**Techniques:**
1. The most efficient way to find a substring is Knuth-Morris-Pratt (KMP) algorithm, which has O(n) time efficiency.
2. The most efficient way to determine if two words are anagrams is to count the letters and their occurrences in each word and compare.
3. The number of permutations is always *n!*.

```python
def find_kmp(string, pattern):
    len_string, len_pattern = len(string), len(pattern)
    if len_pattern == 0:
        return 0
    fail = compute_kmp_fail(pattern)
    i, j = 0, 0
    while i < len_string:
        # If a character in the pattern matches a character in the string...
        if string[i] == pattern[j]:
            # If the last character in the pattern has been matched, the substring has been found.
            if j == len_pattern - 1:
                return i - len_pattern + 1
            # Go to the next character in both the string and the pattern.
            i += 1
            j += 1
        elif j > 0:
            j = fail[j - 1]
        # If no match, go to the next character in the string.
        else:
            i += 1
    return -1


def compute_kmp_fail(pattern):
    len_pattern = len(pattern)
    fail = [0] * len_pattern
    i, j = 1, 0
    while i < len_pattern:
        if pattern[i] == pattern[j]:
            fail[i] = j + 1
            i += 1
            j += 1
        elif j > 0:
            j = fail[j - 1]
        else:
            i += 1
    return fail
```

```python
def match_pattern(pattern, string):
    if len(pattern) == 0 and len(string) == 0:
        return True
    if len(pattern) > 1 and pattern[0] == '*' and len(string) == 0:
        return False
    if pattern[0] == string[0]:
        return match_pattern(pattern[1:], string[1:])
    if len(pattern) != 0 and pattern[0] == '*':
        return match_pattern(pattern[1:], string) or match_pattern(pattern, string[1:])
    return False
```

### Dynamic Programming

**Examples:**
1. Find the longest common substring between two strings.

### Binary Trees

**Examples:**
1. Given preorder traversal of a BST, find the leaf nodes.
2. Find first pair of mismatched nodes in two preorder traversals of BSTs.
3. Print the columns of a binary tree in order.
4. *Find the closest common parent of two nodes in a binary tree.*
5. For each node in a binary tree, find the next node to the right at the same depth.
6. Find the in-order successor of a node in a BST.
7. Convert a binary tree to a circular double linked list.
8. Print a tree in breadth first traversal.
9. Print all paths in a tree from root to leaf.
10. Determine if a binary tree is a BST.
11. Find the kth largest element in a BST.

```python
def find_leaf_nodes(preorder):
    if not preorder:
        return None
    leaf_nodes = []
    stack = Stack()
    for i in range(1, len(preorder)):
        if preorder[i - 1] > preorder[i]:
            stack.push(preorder[i - 1])
        else:
            found = False
            while not stack.is_empty():
                if preorder[i] > stack.top():
                    stack.pop()
                    found = True
                else:
                    break
            if found:
                leaf_nodes.append(preorder[i - 1])
    leaf_nodes.append(preorder[-1])
    return leaf_nodes
```

```python
def lowest_common_ancestor(node, element1, element2):
    if node is None:
        return None
    if node.element == element1 or node.element == element2:
        return node
    left = lowest_common_ancestor(node.left, element1, element2)
    right = lowest_common_ancestor(node.right, element1, element2)
    if left and right:
        return node
    if left:
        return left
    return right
```

```python
def inorder_successor(root, node):
    if node.right:
        while node.left is not None:
            node = node.left
        return node
    successor = None
    while root is not None:
        if node.element < root.element:
            successor = root
            root = root.left
        elif node.element > root.element:
            root = root.right
        else:
            break
    return successor
```

```python
class IntervalNode:
    def __init__(self, low, high, left=None, right=None):
        self.low = low
        self.high = high
        self.left = left
        self.right = right
        self.maximum = self.high


def insert_interval(root, low, high):
    if root is None:
        return IntervalNode(low, high)
    if low < root.low:
        root.left = insert_interval(root.left, low, high)
    else:
        root.right = insert_interval(root.right, low, high)
    if root.maximum < high:
        root.maximum = high
    return root


def intervals_overlap(low1, high1, low2, high2):
    return low1 < high2 and low2 < high1


def overlap_search(root, low, high):
    if root is None:
        return None
    if intervals_overlap(root.low, root.high, low, high):
        return root.low, root.high
    if root.left and root.left.maximum >= low:
        return overlap_search(root.left, low, high)
    return overlap_search(root.right, low, high)


def conflicting_intervals(intervals):
    conflicts = []
    root = insert_interval(None, intervals[0])
    for i in range(1, len(intervals)):
        temp = overlap_search(root, intervals[i])
        if temp:
            conflicts.append(intervals[i])
        root = insert_interval(root, interval[i])
    return conflicts
```

```python
class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end


def merge_intervals(intervals):
    intervals.sort(key=lambda i: i.start)
    stack = Stack()
    stack.push(intervals[0])
    for interval in intervals[1:]:
        top = stack.top()
        if top.end < interval.start:
            stack.push(interval)
        elif top.end < interval.end:
            top.end = interval.end
    merged = []
    while not stack.is_empty():
        merged.append(stack.pop())
    return merged
```

```python
class Graph:
    def __init__(self, graph, num_rows, num_cols):
        self.graph = graph
        self.num_rows = num_rows
        self.num_cols = num_cols

    def is_safe(self, row, col, visited):
        return (0 <= row < self.num_rows and
                0 <= col < self.num_cols and
                not visited[row][col] and
                self.graph[row][col] == 1)

    def dfs(self, row, col, visited):
        row_position = (-1, -1, -1, 0, 0, 1, 1, 1)
        col_position = (-1, 0, 1, -1, 1, -1, 0, 1)
        visited[row][col] = True
        for position in range(8):
            new_row = row + row_position[position]
            new_col = col + col_position[position]
            if self.is_safe(new_row, new_col, visited):
                self.dfs(new_row, new_col, visited)

    def count_islands(self):
        visited = [[False] * self.num_cols for _ in range(self.num_rows)]
        count = 0
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if not visited[row][col] and self.graph[row][col] == 1:
                    self.dfs(row, col, visited)
                    count += 1
        return count
```

## Search Algorithms

### Linear Search

#### Discussion

- Θ(n) time efficiency
- List does not have to be sorted

#### Steps

Loop through each item in the list and check to see if it matches the search item. If it does, return the index. If you reach the end of the list and none of the items matched the search item, return `-1`.

#### Example

```python
def linear_search(items, search_item):
    for index in range(len(items)):
        if items[index] == search_item:
            return index
    else:
        return -1
```

```python
def linear_search(items, search_item):
    for item, index in enumerate(items):
        if item == search_item:
            return index
    else:
        return -1
```

### Binary Search

#### Discussion

- Θ(lg(n)) time efficiency
- List has to be sorted

#### Steps

#### Example

**Recursive Implementation**

```python
def binary_search(items, start, end, search_item):
    if end >= start:
        middle = start + (end - 1) // 2
        if items[middle] == search_item:
            return middle
        elif items[middle] > search_item:
            return binary_search(items, start, middle - 1, search_item)
        else:
            return binary_search(items, middle + 1, end, search_items)
    else:
        return -1
```

**Iterative Implementation**

```python
def binary_search(items, start, end, search_item):
    while start <= end:
        middle = start + (end - 1) // 2
        if items[middle] == search_item:
            return middle
        elif items[middle] > search_item:
            end = middle - 1
        else:
            start = middle + 1
    else:
        return -1
```
