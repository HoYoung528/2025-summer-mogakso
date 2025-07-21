# DFS

## 개념

---

- 그래프 완전 탐색 기법 중 하나
- 시작 노드에서 출발하여 탐색할 분기를 정하여 최대 깊이까지 탐색을 마친 후 다른 쪽 분기로 이동하여 다시 탐색
- 시간 복잡도: O(V+E) → 이때 V는 노드 수, E는 엣지 수
- 스택과 재귀를 이용하여 구현

---

## 문제

---

### 백준11724 - 연결 요소의 개수 구하기(실버2)

```python
import sys
sys.setrecursionlimit(10**6)
input = sys.stdin.readline

node, edge = map(int, input().split())
graph = [[] for _ in range(node+1)]
visited = [False] * (node+1)
count = 0

for i in range(edge):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

def dfs(start):
    visited[start] = True
    child = graph[start]
    for c in child:
        if visited[c] is False:
            dfs(c)

for i in range(1, node+1):
    if visited[i] is False:
        count += 1
        dfs(i)

print(count)
```

DFS를 이용하여 한번의 DFS가 끝나면 count를 1 증가시키도록 하여 연결 요소의 개수를 구하였다.

### 백준2023 - 신기한 소수(골드5)

```python
import sys
sys.setrecursionlimit(10**6)
input = sys.stdin.readline

prime = [2, 3, 5, 7]
n = int(input())

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def dfs(num, depth):
    if is_prime(num):
        if depth == n:
            print(num)
            return
        for i in range(1, 10, 2):
            next_num = (num*10) + i
            dfs(next_num, depth+1)

for i in prime:
    dfs(i, 1)
```

### 백준13023 - ABCDE(골드5)

```python
import sys
sys.setrecursionlimit(10**6)
input = sys.stdin.readline

n, m = map(int, input().split())
graph = [[] for _ in range(n)]
visited = [False] * n
depth = 0
depth_4 = False

for i in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

def dfs(s, d):
    global depth_4
    if d == 4:
        depth_4 = True
        return
    visited[s] = True
    child = graph[s]
    for c in child:
        if visited[c] is False:
            dfs(c, d+1)
    visited[s] = False

    
for i in range(n):
    visited = [False] * n
    dfs(i, 0)
    if depth_4 is True:
        break

if depth_4 is True:
    print('1')
else:
    print('0')
```

graph의 깊이가 4인 경우의 수가 존재하면 1을 출력하도록 설계하였다. 모든 출발 지점을 고려해야 하므로 visited를 False로 초기화하고 반복문을 돌렸다.

# BFS

## 개념

---

- 그래프를 완전 탐색하는 방법 중 하나, 시작 노드에서 출발해 시작 노드를 기준으로 가장 가까운 노드를 먼저 방문하면서 탐색하는 알고리즘
- 시간복잡도: O(V+E) → 이때 V는 노드 수 E는 엣지 수
- 선입선출 방식이므로 큐를 이용해 구현

---

## 문제

---

### 백준1260 - DFS와 BFS (실버2)

```python
import sys
from collections import deque
sys.setrecursionlimit(10**6)
input = sys.stdin.readline

n, m, v = map(int, input().split())
graph = [[] for _ in range(n+1)]
visited = [False] * (n+1)
dfs_result = []
bfs_result = []

for i in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

for i in range(1, n+1):
    graph[i].sort()

def dfs(start):
    visited[start] = True
    dfs_result.append(start)
    for c in graph[start]:
        if visited[c] is False:
            dfs(c)

def bfs(start):
    queue = deque()
    queue.append(start)
    visited[start] = True   
    while queue:
        node = queue.popleft()
        bfs_result.append(node)
        for c in graph[node]:
            if visited[c] is False:
                visited[c] = True
                queue.append(c)

dfs(v)
visited = [False] * (n+1)
bfs(v)

for r in dfs_result:
    print(r, end=" ")
print()
for r in bfs_result:
    print(r, end=" ")
```

DFS, BFS 알고리즘 구현 문제

### 백준2178 - 미로찾기(실버1)

```python
import sys
from collections import deque
sys.setrecursionlimit(10**6)
input = sys.stdin.readline

n, m = map(int, input().split())
maze = []
visited = [([False] * m) for _ in range (n)]
move = [(-1, 0), (1, 0), (0, -1), (0, 1)]

for i in range(n):
    maze.append(list(map(int, input().strip())))

def bfs(start_x, start_y, depth):
    queue = deque()
    visited[start_x][start_y] = True
    queue.append((start_x, start_y, depth))
    while queue:
        x, y, depth = queue.popleft()
        if x == n-1 and y == m-1:
            return depth
        for (dx, dy) in move:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < n and 0 <= ny < m:
                if maze[nx][ny] == 1 and visited[nx][ny] is False:
                    visited[nx][ny] = True
                    queue.append((nx, ny, depth+1))

print(bfs(0, 0, 1))
```

### 백준1167 - 트리의 지름(골드2)

```python
import sys
sys.setrecursionlimit(10**6)
input = sys.stdin.readline

v = int(input())
graph = [[] for _ in range(v+1)]
result = []
far_node = 0

for i in range(v):
    idx = 1
    input_num = list(map(int, input().split()))
    v_num = input_num[0]
    while True:
        if input_num[idx] == -1:
            break
        dest_num = input_num[idx]
        dest_weight = input_num[idx+1]
        graph[v_num].append((dest_num, dest_weight))
        idx += 2

def dfs(start):
    global distance, max_distance, far_node
    visited[start] = True
    if distance > max_distance:
        max_distance = distance
        far_node = start

    for (dest, weight) in graph[start]:
        if visited[dest] is False:
            distance += weight
            dfs(dest)
            distance -= weight

visited = [False] * (v+1)
max_distance = 0
distance = 0
dfs(1)

visited = [False] * (v+1)
max_distance = 0
distance = 0
dfs(far_node)

print(max_distance)
```
# 이진 탐색

## 개념

---

- 데이터가 정렬돼 있는 상태에서 원하는 값을 찾아내는 알고리즘
- 중앙값과 찾고자 하는 값을 비교해 데이터 크기를 절반씩 줄이면서 대상을 찾는다
- 이진 탐색 과정
    1. 데이터셋의 중앙값을 선택
    2. 중앙값 > 타깃 데이터일 때 중앙값 기준으로 왼쪽 데이터셋 선택
    3. 중앙값 < 타깃 데이터일 때 중앙값 기준으로 오른쪽 데이터셋 선택
    4. 과정 1~3을 반복하다가 중앙값 == 타깃 데이터일 때 탐색 종료

---

## 문제

---

### 백준1920 - 원하는 정수 찾기(실버4)

```python
import sys
input = sys.stdin.readline

n = int(input())
n_nums = list(map(int, input().split()))
n_nums.sort()
result = []

m = int(input())
m_nums = list(map(int, input().split()))

def binary_search(arr, start, end, target):
    if start > end:
        return 0
    
    mid = (start + end) // 2
    if arr[mid] > target:
        return binary_search(arr, start, mid-1, target)
    elif arr[mid] < target:
        return binary_search(arr, mid+1, end, target)
    else:
        return 1

for i in  m_nums:
    result.append(binary_search(n_nums, 0, n-1, i))

for r in result:
    print(r)
```

이진 탐색을 사용 

start > end 조건이 중요, 무한 루프에 빠지지 않기 위해 종료 조건 걸어놓음

---

### 백준2343 - 기타 레슨(실버1)

```python
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
length = list(map(int, input().split()))

start = max(length)
end = sum(length)

def binary_search(arr, start, end, target):
    if start > end:
        return start
    
    sum = 0
    count = 0
    size = (start+end)//2
    for i in arr:
        sum += i
        if sum > size:
            sum = i
            count += 1
        elif sum == size:
            sum = 0
            count += 1
    if sum > 0:
        count += 1
        
    if count <= target:
        return binary_search(arr, start, size-1, target)
    else:
        return binary_search(arr, size+1, end, target)

print(binary_search(length, start, end, m))
```

블루레이의 최소 용량과 최대 용량을 구한 다음에 이진 탐색을 이용

---
