## 버블 정렬

## 개념

---

- 인접한 데이터의 크기만 비교하여 정렬하는 방법
- 시간복잡도 O(N^2)
- 코테에서 거의 사용 안함

---

## 문제

---

### 백준2750 - 수 정렬하기(브론즈2)

```python
import sys
input = sys.stdin.readline

cnt = int(input())
nums = []
temp = 0
for i in range(cnt):
    nums.append(int(input()))

nums.sort()

for n in nums:
    print(n)
```

그냥 sort 내장함수 사용하면 됨

---

### 백준1377 - 버블소트(골드2)

```python
import sys 
input = sys.stdin.readline

cnt = int(input())
num = []
result = 0
max_move = 0

for i in range(cnt):
    value = int(input())
    num.append((value, i))

num.sort()

for i in range(cnt):
    move = num[i][1] - i
    if move > max_move:
        max_move = move

print(max_move + 1)
```

버블 소트의 원리를 정확히 이해해야 풀 수 있는 문제였다. 버블 소트는 인접한 수들끼리만 swap을 진행한다. 원리상 버블 소트에서 한번의 for문이 끝나면 인덱스가 하나씩 밀리기 때문에 sort 전과 후의 차이를 구하고 가장 큰 인덱스 차이값이 for문이 돌아간 횟수이다.

---

## 선택 정렬

## 개념

---

- 선택 정렬은 데이터의 최대나 최소값을 구하고 이를 순차적으로 나열하는 방법
- 시간 복잡도 O(N^2)
- 코테에서는 사용하지 않음

---

## 문제

---

### 백준1427 - 소트인사이드(실버5)

```python
import sys
input = sys.stdin.readline

arr = list(map(int, input().strip()))

# 선택 정렬 (내림차순)
n = len(arr)
for i in range(n):
    max_idx = i
    for j in range(i+1, n):
        if arr[j] > arr[max_idx]:  # 더 큰 값을 찾으면
            max_idx = j
    # 현재 위치(i)와 최대값 위치(max_idx) 교환
    arr[i], arr[max_idx] = arr[max_idx], arr[i]

# 공백 없이 출력
for num in arr:
    print(num, end='')
```

더 간단한 풀이 (선택 정렬 사용 X)

```python
import sys
input = sys.stdin.readline

list = list(map(int, input().strip()))

list.sort()
list.reverse()
for i in list:
    print(i, end='')
```

sort함수 사용하고 reverse로 바꾼 다음에 출력

---

## 삽입 정렬

## 개념

---

- 정렬이 되어있는 데이터에 데이터를 삽입하는 원리
- 이것 또한 코테에서는 잘 안 쓰임

---

## 문제

---

### 백준11399 - ATM(실버4)

```python
import sys
input = sys.stdin.readline

cnt = int(input())
num = list(map(int, input().split()))
sum = 0

for i in range(1, cnt):
    idx = 0
    target = num[i]
    for j in range(i):
        if num[j] < target:
            idx += 1
        else:
            break
    for j in range(i, idx-1, -1):
        num[j] = num[j-1]
    num[idx] = target
    
for i in range(cnt, 0, -1):
    sum += (num[cnt-i] * i)

print(sum)
```

선택정렬을 사용하고 누적합을 이용하여 값을 구함

더 간단한 풀이

```python
import sys
input = sys.stdin.readline

cnt = int(input())
num = list(map(int, input().split()))
sum = 0

num.sort()
    
for i in range(cnt, 0, -1):
    sum += (num[cnt-i] * i)

print(sum)
```

sort() 메서드를 사용하여 간단하고 풀이 가능

## 퀵 정렬

## 개념

---

- 평균 시간 복잡도: O(nlogn) → 최악의 경우 O(n^2)
- pivot을 중심으로 데이터를 2개의 집합으로 나누면서 정렬
- 동작 과정
    1. 배열에서 pivot 원소 하나 선택
    2. pivot보다 작은 값은 왼쪽, 큰 값은 오른쪽으로 나눔 (분할)
    3. 나뉜 두 구간을 재귀적으로 정렬
    4. 모든 재귀가 끝나면 정렬 완료

---

## 문제

---

### 백준11004 - K번째 수 구하기(실버5)

```python
import sys
input = sys.stdin.readline
sys.setrecursionlimit(10**6)

def quickselect(arr, k):
    if len(arr) == 1:
        return arr[0]

    pivot = arr[len(arr) // 2]  # 중앙값을 피벗으로

    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    if k < len(left):
        return quickselect(left, k)
    elif k < len(left) + len(mid):
        return pivot
    else:
        return quickselect(right, k - len(left) - len(mid))

# 입력 처리
n, k = map(int, input().split())
arr = list(map(int, input().split()))

print(quickselect(arr, k - 1))  # k는 1-based → 0-based로 변환

```

더 쉬운 풀이

```python
import sys
input = sys.stdin.readline

n, k = map(int, input().split())
nums = list(map(int, input().split()))

nums.sort()

print(nums[k-1])
```

## 병합 정렬

## 개념

---

- Divde and Conquer 방식의 안정 정렬 알고리즘
- 시간 복잡도: O(nlogn)
- 정렬 과정
    1. 배열 절반으로 나눔
    2. 재귀적으로 병합 정렬 수행
    3. 정렬된 두 배열을 병합

---

## 문제

---

### 백준2751 - 수 정렬하기2(실버5)

```python
import sys
input = sys.stdin.readline

def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    l = r = 0

    while l < len(left) and r < len(right):
        if left[l] <= right[r]:
            result.append(left[l])
            l += 1
        else:
            result.append(right[r])
            r += 1

    result.extend(left[l:])
    result.extend(right[r:])
    return result

# 입력 받기
n = int(input())
arr = [int(input()) for _ in range(n)]

sorted_arr = merge_sort(arr)

# 출력
for num in sorted_arr:
    print(num)
```

더 쉬운 방법

```python
import sys
input = sys.stdin.readline

n = int(input())
nums = []

for i in range(n):
    nums.append(int(input()))

nums.sort()
for n in nums:
    print(n)
```

## 기수 정렬

## 개념

---

- 자릿수 기준으로 낮은 자리수부터 차례대로 정렬하는 알고리즘
- 시간복잡도: O(kN) → 이때 k는 자릿수를 의미
- 예제

```python
def radix_sort(arr):
    max_num = max(arr)
    exp = 1  # 자릿수: 1, 10, 100, ...

    while max_num // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10

def counting_sort_by_digit(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10  # 0~9 자리수

    # 현재 자릿수 기준으로 카운팅
    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1

    # 누적합 → 정렬 위치 계산
    for i in range(1, 10):
        count[i] += count[i - 1]

    # output 배열에 정렬
    for i in reversed(range(n)):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    # 원래 배열에 복사
    for i in range(n):
        arr[i] = output[i]

# 사용 예시
arr = [170, 45, 75, 90, 802, 24, 2, 66]
radix_sort(arr)
print(arr)  # 출력: [2, 24, 45, 66, 75, 90, 170, 802]

```
