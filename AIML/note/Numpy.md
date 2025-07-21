### ndarray: N차원(dimension) 배열 객체

### ndarray 생성

- numpy 모듈의 array() 함수를 통해 생성
- list나 ndarray를 넣음

```python
import numpy as np

array1 = np.array([1, 2, 3])
array2 = np.array([[1,2,3],
										[4,5,6]]
```

### ndarray 형태와 차원

- `ndarray.shape()`  : 속성을 반환 → ex) (2, 3) 행: 2, 열: 3
- `ndarray.ndim()` : 차원을 반환 → ex) 1, 2, 3

### ndarray 타입 및 변환

- ndarray 데이터 값은 숫자, 문자열, bool 모두 가능
- ndarray 내의 데이터는 같은 데이터 타입이어야 한다.
- `ndarray.dtype` : 데이터 타입 반환
- `ndarray.astype` : 데이터 타입 변환

### ndarray의 axis 축

- ndarray의 shape는 행, 열, 높이가 아니라 axis0, axis1, axis2 와 같이 axis 단위로 부여된다

![image.png](attachment:eaed9907-3be5-4ceb-b385-4cafe49a8f93:image.png)

### ndarray 편하게 생성하기

- `np.arange()` : ndarray를 연속값으로 생성 → ex) np.arrange(5) = [0,1,2,3,4]
- `np.zeros()` : ndarray를 0으로 초기화 생성 → ex) np.zeros((3, 2)) = [[0,0], [0,0], [0,0]]
- `np.ones()` : ndarray를 1로 초기화 생성 → ex) np.ones((3, 2)) = [[1,1], [1,1], [1,1]]

### ndarray 차원과 크기 변경

- `reshape()` : 차원을 변환해주는 메서드 → ex) reshape(2, 5) = 2x5 ndarray로 변환
- reshape() 에서 인자에 -1을 부여하면 데이터의 크기에 맞춰서 알아서 변환 → ex) reshape(-1, 5) = 열의 크기는 5이되 행은 알아서 변환, 만약 데이터가 10개 있으면 행은 2로 변환됨

### ndarray 데이터 세트 선택

- 특정 위치 단일값 추출
    - ex) arr[0], arr[3]
- 슬라이싱: 연속된 인덱스 추출 방식, 시작인덱스:종료인덱스 → ex) [0:9]
    - ex) arr1[0:9], arr1[:], arr1[3:], arr2[0:2][0:2], arr2[:3][2:]
- 팬시 인덱싱: 일정한 인덱스 집합 → ex) [0, 1, 2]
    - ex) arr[[1,3,5]], arr[[2,3,5]]
- 불린 인덱싱: 특정 조건을 만족하는지
    - ex) arr[arr>5]

### sort 와 argsort

- `sort()`
    - `np.sort(arr)`: 원본 행렬은 보존한 채 정렬된 행렬 반환
    - `arr.sort()`: 원본 행렬을 정렬하고 반환 값은 None
    - 기본적으로 오름차순
    - 만약 내림차순으로 하고 싶다면 np.sort()[::-1]
- `argsort()`
    - 원본 행렬 정렬 시 정렬된 행렬의 원래 인덱스를 필요로 할 때 사용
    
    ![image.png](attachment:9e0ae87c-3d26-4f78-9c0c-111e146b7683:image.png)
    

### 행렬내적과 전치행렬

- 행렬 내적
    - `np.dot(A, B)`로 행렬 내적 가능
- 전치 행렬
    - 행과 열을 바꿈
    - `np.transpose(A)`를 통해 가능
