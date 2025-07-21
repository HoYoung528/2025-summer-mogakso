## Pandas

- 데이터 처리를 위해 존재하는 가장 인기 있는 라이브러리
- 데이터 세트는 형, 열 즉 2차원 데이터로 구성되어 있다

## 판다스 주요 구성 요소

- DataFrame: Column X Rows 로 구성된 2차원 데이터셋
- Series: Column 값으로만 구성된 1차원 데이터셋
    - Column명이 주어지지 않음
- Index: DataFrame/Series의 고유한 Key값 객체(데이터베이스의 PK 개념)

## 기본 API

### 1. `read_csv()`

- csv 파일을 편리하게 DataFrame으로 로딩, sep인자를 콤마가 아닌 다른 분리자로 변경하여 다른 유형의 파일도 로드 가능

### 2. `head()`와 `tail()`

- head는 DataFrame의 맨 앞부터 일부분, tail()은 DataFrame의 맨 뒤부터 일부분 출력
- default는 5개의 결과 출력

### 3. `shape`

- DataFrame의 행과 열 크기를 가지고 있는 속성

### 4. DataFrame 생성

- `DataFrame()`을 통해 DataFrame 생성
- 새로운 컬럼명(column= )을 추가하거나 인덱스를 새로운 값(index= )으로 할당 가능

### 5. DataFrame 컬럼명과 인덱스

- `df.columns`: 컬럼명을 반환
- `df.index`: index 정보를 반환
- `df.index.value`: index의 값들을 반환

### 6. `info()`

- DataFrame 내의 컬럼명, 데이터타입, Null건수, 데이터 건수 정보 제공

### 7. `describe()`

- 데이터값들의 평균, 표준편차, 4분위 분포도를 제공
- 숫자형 컬럼들에 대해서만 해당 정보를 제공

### 8. `value_counts()`

- 개별 데이터 값의 분포도를 제공
- values_counts()는 Null 값을 무시하고 결과값을 내놓기 때문에 null 포함 여부를 dropna인자로 설정해야 한다.
- 이때 오름차순 정렬
- dropna의 디폴트 값은 True

## DataFrame과 리스트, 딕셔너리, ndarray 상호 변환

| 변환 형태 | 설명 |
| --- | --- |
| list → DataFrame | df_list = pd.DataFrame(list, columns=col_name)
DataFrame 생성 인자로 리스트 객체와 매핑되는 컬럼명들 입력 |
| ndarray → DataFrame | df_array = pd.DataFrame(array, columns=col_name)
DataFrame 생성 인자로 ndarray와 매핑되는 컬럼명들을 입력 |
| dict → DataFrame | dict = {’col1’: [1, 11], ‘col2’: [2, 22], ‘col3’: [3, 33]}
df_dict = pd.DataFrame(dict)
딕셔너리의 키로 컬럼명을, 값들을 리스트 형식으로 입력 |
| DataFrame → ndarray | df_array.values
DataFrame 객체의 values 속성을 이용하여 ndarray 변환 |
| DataFrame → list | df.list.values.tolist()
ndarray로 변환 후 tolist()를 이용하여 list로 변환 |
| DataFrame → dict | df_dict.to_dict()
DataFrame 객체의 to_dict() 이용하여 변환 |

## DataFrame 데이터 삭제

### drop()

- row를 삭제할 때는 axis=0, column을 삭제할 때는 axis=1 설정
- inplace=False일 때는 원본 DataFrame은 유지하고 드롭된 DataFrame을 새롭게 객체 변수로 받음
- inplace=True일 때는 원본 DataFrame을 변환

## 판다스 index 개요

- 판다스의 index는 데이터베이스의 PK와 유사하게 DataFrame, Series를 고유하게 식별하는 객체
- DataFrame/Series 객체는 index 객체를 포함하지만 연산 함수를 적용할 때는 index를 연산에서 제외함(index는 오직 식별용)
- DataFrame.index, Series.index 속성을 통해 index 객체 추출 가능
- index는 숫자형뿐만 아니라 문자형/DateTime도 상관 없음

## DataFrame 인덱싱, 필터링 및 정렬

### [] 기능

- 컬럼 기반 필터링 또는 불린 인덱싱 필터링 제공
- 단일 컬럼명을 입력하면 컬럼명에 해당하는 Series 객체 반환
- 여러 개 컬럼명들을 list로 입력하면 컬럼명들에 해당하는 DataFrame 객체 반환

```python
# 단일 컬럼명 입력 -> Series 객체 반환
titanic_df['Name']

# 컬럼명 리스트 입력 -> DataFrame 객체 반환
tatinic_df[['Name', 'Age']]
```

### loc, iloc

- 명칭/위치 기반 인덱싱 제공
- 명칭 기반 인덱싱: 컬럼의 명칭을 기반으로 위치를 지정하는 방식(행 위치는 index를 이용)
- 위치 기반 인덱싱: 행, 열 위치값으로 정수가 입력됨(index를 이용하지 않음)

![image.png](attachment:80900363-06e5-4d6e-8c8d-7c93258e2f86:image.png)

### 불린 인덱싱

- 위치기반, 명칭기반 인덱싱 모두 사용하지 않고 조건식을 [] 안에 기입하여 간편하게 필터링 수행

### DataFrame 정렬

- sort_values() 메서드는 by 인자로 정렬하고자 하는 컬럼값을 입력 받아 해당 컬럼값으로 DataFrame을 정렬
- 오름차순 정렬이 기본

## DataFrame의 집합 연산 수행

### Aggregation

- sum(), max(), min(), count() 등은 DataFrame/Series에서 집합연산을 수행
- DataFrame에서 바로 aggregation을 호출할 경우 모든 컬럼에 해당 aggregation을 적용

### Group By

- group by 연산을 위해 groupby() 메소드 제공
- groupby() 메소드는 by 인자로 컬럼명을 입력 받으면 DataFrameBroupBy 객체 반환
- 반환된 DataFrameBroupBy 객체에 aggregation 함수 수행
- 서로 다른 aggregation을 적용하려면 agg()를 활용
- agg내의 인자로 들어가는 Dict객체에 동일한 Key값을 가지는 두개의 value가 있을 경우 마지막 value로 update됨 → 동일 컬럼에 서로 다른 aggregation을 가지면서 추가적인 컬럼 aggregation이 있을 경우 원하는 결과로 출력 X

## 결손 데이터 처리

- isna() : 주어진 컬럼 값들이 NaN인지 True/False 값을 반환
- fillna() : missing 데이터를 인자로 주어진 값으로 대체

## unique와 replace

- unique() : 컬럼 내 몇 건의 고유값이 있는지 파악
- replace() : 원본값을 특정값으로 대체
