## Scikit-learn 소개

- 머신러닝을 위한 매우 다양한 알고리즘과 개발을 위한 편리한 프레임워크와 API 제공
- 주로 Numpy, Scipy 기반 위에서 구축된 라이브러리

## 머신러닝 용어

- feature: 데이터 세트의 일반 속성, 타겟값을 제외한 나머지 속성을 모두 feature로 지칭
- label: 지도 학습 시 데이터의 학습을 위해 주어지는 정답 데이터

## 지도학습 - 분류

### 분류(Classification)

- 대표적인 지도학습 방법의 하나
- 다양한 feature와 분류 결정값인 레이블 데이터로 모델을 학습
- 별도의 테스트 데이터 세트에서 미지의 레이블 예측

## 사이킷런 기반 프레임워크

### Estimator

- 학습: fit(), 예측: predict()
- 분류(Classifier)
    - DecisionTree
    - RandomForest
    - GradientBoosting
    - GaussianNB
    - SVC
- 회귀(regression)
    - LinearRegression
    - Ridge
    - Lasso
    - RandomForest
    - GrandientBoosting

### 사이킷런 주요 모듈

![image.png](attachment:dadb1073-cb56-4c4d-befc-7b2b5cf016b5:image.png)

### 사이컷런 내장 예제 데이터셋

![image.png](attachment:c30a2ac3-d5f4-48b7-bf1e-557e4e643911:image.png)

## Model Selection

### 학습 데이터셋

- 머신러닝 알고리즘의 학습을 위해 사용
- 데이터의 속성들과 결정값을 모두 가지고 있음
- 학습 데이터를 기반으로 머신러닝 알고리즘이 데이터 속성과 결정값의 패턴을 인지하고 학습

### 테스트 데이터셋

- 테스트 데이터셋에서 학습된 머신러닝 알고리즘을 테스트
- 테스트 데이터는 속성 데이터만 머신러닝 알고리즘에 제공
- 머신러닝 알고리즘은 제공된 데이터를 기반으로 결정값 예측
- 테스트 데이터는 학습 데이터와 별도의 데이터셋으로 제공되어야 함

### 학습 데이터와 테스트 데이터 분리 - train_test_split()

- test_size: 테스트 데이터 크기를 얼마로 샘플링할 것인가 결정
- train_size: 학습용 데이터 세트 크기를 얼마로 샘플링할 것인가 결정
- shuffle: 데이터를 분리하기 전에 데이터를 머리 섞을지 결정(디폴트는 True)
- random_state: 호출할 때마다 동일한 학습/테스트용 데이터셋을 생성하기 위해 주어지는 난수

## 교차 검증

- 교차검증: 학습 데이터를 다시 분할하여 학습 데이터와 학습된 모델의 성능을 일차 평가하는 검증 데이터로 나눔

### K-fold 교차검증

![image.png](attachment:39121eff-df5e-4dc2-88ee-59f09d0131d1:image.png)

### startified K-fold

- 불균형한 분포도를 가진 레이블 데이터 집합을 위한 K-fold 방식
- 학습데이터와 검증 데이터셋이 가지는 레이블 분포도가 유사하도록 검증 데이터 추출

## 간편한 교차 검증

### cross_val_score()

1. 폴드 세트 설정
2. for루프에서 반복적으로 학습/검증 데이터 추출 및 학습과 예측 수행
3. 폴드 세트별로 예측 성능을 평균하여 최종 성능 평가
- **cross_val_score() 함수로 폴드 세트 추출, 학습/예측, 평가를 한번에 수행**

```python
iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

data = iris_data.data
label = iris_data.target

scores = cross_val_score(dt_clf, data, label, scoring='accuracy', cv=3)
```

### GridSearchCV

- 하이퍼 파라미터를 순차적으로 입력하면서 편리하게 최적의 파라미터를 도출할 수 있는 방안 제공
- 이때 하이퍼파라미터 후보는 dictionary 형태로 해야함
- refit은 최적의 파라미터로 재학습하도록 하는 옵션 (디폴트값 True)

```python
# parameter 들을 dictionary 형태로 설정
parameters = {'max_depth':[1, 2, 3], 'min_samples_split':[2,3]}

# refit=True 가 defulat, True이면 가장 좋은 파라미터 설정으로 재학습
grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True, return_train_score=True)
```

## 데이터 전처리

- 데이터 클린징
- 결손값 처리(Null/NaN 처리)
- 데이터 인코딩(label, one-hot)
- 데이터 스케일링
- 이상치 제거
- feature selection, 추출 및 가공

### 레이블 인코딩

![image.png](attachment:d11a7143-a9c1-4969-970f-6195028b4fda:image.png)

- 상품 분류를 레이블 인코딩함
- 데이터들간의 상하관계가 생기기 때문에 위험할 수 있다
    - ex) TV=0, 컴퓨터=5 → 0 < 5 → 따라서 TV < 컴퓨터
- LabelEncoder 클래스 사용
- fit()과 transform()을 이용하여 변환

```python
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
```

### 원핫 인코딩

![image.png](attachment:63704ce5-2397-4a72-b7cb-b222c2751855:image.png)

- 피처 값의 유형에 따라 새로운 피처를 추가해 고유 값에 해당하는 컬럼에만 1을 표시하고 나머지 컬럼에는 0을 표시하는 방식
- OneHotEncoder 클래스 사용
- it()과 transform()을 이용하여 변환
- 인자로 2차원 ndarray 입력 필요
- Sparse 배열 형태로 변환되므로 toarray()를 적용하여 다시 Dense 형태로 변환해야 함
- pd.get_dummies(DataFrame)을 이용

```python
# 2차원 ndarray로 변환합니다. 
items = np.array(items).reshape(-1, 1)

# 원-핫 인코딩을 적용합니다. 
oh_encoder = OneHotEncoder()
oh_encoder.fit(items)
oh_labels = oh_encoder.transform(items)
```

### 피처 스케일링

- 표준화는 평균이 0이고 분산이 1인 가우시안 정규 분포를 가진 값으로 변환하는 것을 의미
- 정규화는 서로 다른 피처의 크기를 통일하기 위해 크기를 변환해주는 개념
- StandardSclaer: 평균이 0이고 분산이 1인 정규 분포 형태로 변환
- MinMaxScaler: 데이터값을 0과 1사이의 범위 값으로 변환
- 유의사항
    - 학습할 때의 데이터 척도와 테스트할 때의 데이터 척도는 반드시 같아야 한다
    - test_array에 Scale 변환을 할 때는 반드시 fit()을 호출하지 않고 transform() 만으로 변환해야 함
