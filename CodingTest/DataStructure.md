## 구간 합

## 개념

---

i ~ j 까지의 합을 구할 때

- 배열을 돌면서 합을 구하는 경우 → 시간복잡도 O(N)
- 합 배열을 만들어서 구하는 경우 → 시간복잡도 O(1)

### 1. 1차원 구간합

1. 합 배열을 구한다(1부터 n까지의 합)

```python
s[0] = a[0]
s[1] = a[0] + a[1]
s[2] = a[0] + a[1] + a[2]
.
.
s[5] = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]

```

1. 합 배열에서 값을 뺴내어 구간 합을 구한다(i 부터 j 까지)

```python
# 2부터 3까지
s[3] - s[1]

# 1부터 4까지
s[4] - s[0]

# 1부터 3까지
s[3] - s[0]

# i부터 j까지
s[j] - s[i-1]
```

### 2. 2차원 구간합

1. 2차원 구간합 구하는법

```python
prefix[i][j] = num[i][j] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]
```

1. 2차원 누적합 구하는법

```python
sum = prefix[i][j] - prefix[i-1][j] - prefix[i][j-1] + prefix[i-1][j-1]
```

---

## 문제

---

### 백준11659 - 구간 합 구하기1(실버3)

```python
import sys
input = sys.stdin.readline

num_cnt, testcase = map(int, input().split())

num = list(map(int, input().split()))
sum = 0
sum_list = list()
result = list()

for n in num:
    sum += n
    sum_list.append(sum)

for i in range(testcase):
    start, end = map(int, input().split())
    end -= 1
    if (start -2) < 0:
        a = sum_list[end]
    else:
        start -= 2
        a = sum_list[end] - sum_list[start]
    result.append(a)

for r in result:
    print(r)
```

1차원 구간합을 구하는 문제. 해당 인덱스까지의 누적 합을 저장하는 누적 합 배열 생성. 누적 합끼리 빼서 특정 범위의 구간 합을 구함

---

### 백준11660 - 구간 합 구하기2(실버1)

```python
import sys
input = sys.stdin.readline

num = list()
result = list()

size, testcase = map(int, input().split())
sum = [[0]*(size+1) for i in range(size+1)]

for i in range(size):
    temp = 0
    sum_row = []
    row = list(map(int, input().split()))
    num.append(row)

for i in range(1, size+1):
    for j in range(1, size+1):
        sum[i][j] = num[i-1][j-1] + sum[i-1][j] + sum[i][j-1] - sum[i-1][j-1]

for i in range(testcase):
    x1, y1, x2, y2 = map(int, input().split())
    r = sum[x2][y2] - sum[x2][y1-1] - sum[x1-1][y2] + sum[x1-1][y1-1]
    result.append(r)

for r in result:
    print(r)
```

2차원 구간합을 구하는 문제. 1차원 구간합 구하는 문제와 동일하게 진행하되 차원만 다름. 2차원 누적합을 구하고 누적합끼리 빼서 구간합을 구함. 

누적합은 행과 열 기준으로 각각 한 번씩  함. 겹친 부분이 존재하기 때문에 겹친 부분은 한 번 빼줌. 구간합도 같은 방식으로 진행

---

### 백준10986 - 나머지 합 구하기(골드3)

```python
import sys
import math
input = sys.stdin.readline

m, n = map(int, input().split())
remainder = []
r = 0
result = 0
sum = 0
count = [0] * n

num = list(map(int, input().split()))

for i in num:
    sum += i
    r = sum % n
    remainder.append(r)
    count[r] += 1
    

result += count[0]
for i in range(n):
    if (count[i] > 1):
        result += math.comb(count[i], 2)
    
print(result)
```

M으로 나눈 값의 누적합을 구한다. 누적합이 같으면 각 누적합을 뺐을 때 나머지가 0이 되므로 해당 구간합은 M으로 나눠 떨어진다고 볼 수 있다. 

---

## 두 포인터

## 개념

---

두 개의 포인터를 지정하고 포인터를 움직이면서 각 루프마다 중복되는 계산은 하지 않고 추가적으로 더해지거나 빼지는 계산만 수행하는 원리. 

중복되는 계산을 하지 않기 때문에 시간 복잡도가 많이 줄어든다고 볼 수 있다.

---

## 문제

---

### 백준2018 - 연속된 자연수의 합 구하기(실버5)

```python
import sys
input = sys.stdin.readline

num = int(input())
count = 0
sum = 1
start, end = 1, 1

while start <= num:
    if sum == num:
        count += 1
        sum -= start
        start += 1
 
    elif sum < num:
        end += 1
        sum += end
    else:
        sum -= start
        start += 1

print(count)
```

start, end 포인터를 지정하고 start와 end만 조정하면서 값을 빼고 더해준다. 

---

### 백준1940 - 주몽의 명령(실버4)

```python
import sys
input = sys.stdin.readline

n = int(input())
m = int(input())
num = list(map(int, input().split()))
num.sort()
first_idx, second_idx = 0, n-1
sum, count = 0, 0

while first_idx < second_idx:
    sum = num[first_idx] + num[second_idx]
    if sum == m:
        count += 1
        first_idx += 1
        second_idx -= 1
    elif sum > m:
        second_idx -= 1
    else:
        first_idx += 1

print(count)
```

이전 문제와 동일한 방식으로 진행

---

### 백준1253 - 좋은 수 구하기(골드4)

```python
import sys
input = sys.stdin.readline

cnt = int(input())
num = list(map(int, input().split()))
num.sort()
count = 0

for n in range(cnt):
    start = 0
    end = cnt-1
    while start < end:
        if start == n:
            start += 1
            continue
        if end == n:
            end -= 1
            continue
        sum = num[start] + num[end]
        if num[n] == sum:
            count += 1
            break
        elif num[n] < sum:
            end -= 1
        else:
            start += 1

print(count)
```

이전 문제와 동일한 방식으로 진행하되 start나 end가 n과 같을 경우에는 스킵해야 하는 조건이 추가된다. 

---

## 슬라이딩 윈도우

## 개념

---

두 포인터와 매우 비슷한 개념이지만 슬라이딩 윈도우는 일정한 범위가 유지한 채로 이동함

---

## 문제

---

### 백준12891 - DNA 비밀번호(실버5)

```python
import sys
input = sys.stdin.readline

str_len, substr_len = map(int, input().split())
str = input().strip()
min_cnt = list(map(int, input().split()))
cnt = [0]*4
result = 0
start = 0
end = substr_len-1

for i in range(substr_len):
    if str[i] == "A": cnt[0] += 1
    elif str[i] == "C": cnt[1] += 1
    elif str[i] == "G": cnt[2] += 1
    elif str[i] == "T": cnt[3] += 1

if cnt[0] >= min_cnt[0] and cnt[1] >= min_cnt[1] and cnt[2] >= min_cnt[2] and cnt[3] >= min_cnt[3]:
    result += 1

while end < (str_len-1):
    c = str[start]
    if c == "A": cnt[0] -= 1
    elif c == "C": cnt[1] -= 1
    elif c == "G": cnt[2] -= 1
    elif c == "T": cnt[3] -= 1
    start += 1
    end += 1
    c = str[end]
    if c == "A": cnt[0] += 1
    elif c == "C": cnt[1] += 1
    elif c == "G": cnt[2] += 1
    elif c == "T": cnt[3] += 1

    if cnt[0] >= min_cnt[0] and cnt[1] >= min_cnt[1] and cnt[2] >= min_cnt[2] and cnt[3] >= min_cnt[3]:
        result += 1

print(result)
```

일정한 범위를 지정하고 해당 범위를 밀면서 윈도우의 첫번째 요소는 제거하고 마지막 다음 요소는 추가하는 형식이다.

---

## 스택과 큐

## 개념

---

### 스택

- 삽입과 삭제 연산이 후입선출(LIFO)
- top 값이 가장 최근에 들어온 값
- 깊이 우선 탐색, 백트래킹에 매우 효과적
- 스택은 리스트 사용

### 큐

- 삽입과 삭제 연산이 선입선출(FIFO)
- 삽입과 삭제가 양방향
- 너비 우선 탐색에 매우 효과적
- 큐는 컬렉션에서 deque 사용
- priority queue는 heapq를 이용해서 사용가능 (중요)

```python
# queue 선언
from collections import deque
queue = deque()

# priority queue 선언
import heapq

pq = []
heapq.heappush(pq, 3)
heapq.heappush(pq, 1)
heapq.heappush(pq, 2)

print(heapq.heappop(pq))  # 출력: 1
```

---

## 문제

---

### 백준1874 - 스택으로 수열 만들기(실버3)

```python
import sys
input = sys.stdin.readline

n = int(input())
stack = []
result = []
num = 1

for _ in range(n):
    target = int(input())
    
    while num <= target:
        stack.append(num)
        result.append('+')
        num += 1
    
    if stack[-1] == target:
        stack.pop()
        result.append('-')
    else:
        print("NO")
        exit()

for op in result:
    print(op)
```

---

### 백준17298 - 오큰수 구하기

```python
import sys
input = sys.stdin.readline

cnt = int(input())
result = list()
right = list()
stack = list(map(int, input().split()))

for i in range(cnt-1, -1, -1):
    while right and right[-1] <= stack[i]:
        right.pop()
    
    if not right:
        result.append(-1)
    else:
        result.append(right[-1])
    
    right.append(stack[i])

result.reverse()
print(*result)
```

오른쪽에 있는 수를 stack으로 구현해 추가하고 타겟 숫자보다 작으면 pop을 진행

---

### 백준2164 - 카드2(실버4)

```python
import sys
from collections import deque
input = sys.stdin.readline
n = int(input())
queue = deque()

for i in range(n, 0, -1):
    queue.append(i)

while True:
    if len(queue) == 1:
        break;
    queue.pop()
    temp = queue.pop()
    queue.appendleft(temp)

print(*queue)
```

queue를 사용하여 풀이

---

### 백준11286 - 절댓값 힙(실버1)

```python
import sys
import heapq
input = sys.stdin.readline

cnt = int(input())
pq = []
result = list()

for i in range(cnt):
    num = int(input())
    if num != 0:
        heapq.heappush(pq, (abs(num), num))
    else:
        if not pq:
            result.append(0)
        else:
            _, add = heapq.heappop(pq)
            result.append(add)

for r in result:    
    print(r)
```

priority queue를 선언하여 풀이하였다. heappush를 이용하여 정렬 기준을 num의 절댓값과 num 기본 값을 기준으로 삽입한다. 이후 heappop을 이용하여 정렬을 유지한 채 pop한다.
