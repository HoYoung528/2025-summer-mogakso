## 분류 성능 평가 지표

- 정확도(Accuracy)
- 오차행렬(Confusion Matrix)
- 정밀도(Precision)
- 재현율(Recall)
- F1 score
- ROC AUC

## 정확도(Accuracy)

- 직관적으로 모델 예측 성능을 나타내는 평가 지표
- 이진 분류의 경우 데이터 구성에 따라 ML 모델의 성능을 왜곡할 수 있음
- 불균형한 레이블 값 분포에 부적합한 평가 지표

## 오차행렬(Confusion Matrix)

- 이진 분류의 예측 오류가 얼마인지와 더불어 어떠한 유형의 예측 오류가 발생하고 있는지를 함께 나타내는 지표

![image.png](attachment:6b34e4bd-2ea9-4555-82ee-b7a4b32e9560:image.png)

## 정밀도(Precision)과 재현율(Recall)

- 정밀도는 예측을 Positive로 한 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율
- 재현율은 실제 값이 Positive인 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율
- 정밀도는 precision_score(), 재현율은 recall_score() 제공

### Recall과 Precision의 상대적 중요도

- Recall이 더 중요한 경우: 실제 Positive인 데이터 예측을 Negative로 잘못 예측하면 큰 영향이 발생하는 경우 → 암 진단, 금융사기 판별
- Precision이 더 중요한 경우: 실제 Negative인 데이터 예측을 Positive로 잘못 예측하면 큰 영향이 발생하는 경우 → 스팸 메일

### Recall과 Precision의 Trade-off

- Precision과 Recall은 상호 보완적인 평가 지표이기 때문에 어느 한쪽을 강제로 높이면 다른 하나는 떨어지기 쉽다 (Trade-off)
- 분류 결정 Threshold 값이 낮아질수록 Positive로 예측할 확률 높아짐 → Recall 증가
- precision_recall_curve() 함수를 통해 threshold에 따른 precision, recall 변화값을 제공

## F1 Score

- Precision과 Recall을 결합한 평가 지표
- Precision과 Recall이 어느 한쪽으로 치우치지 않는 수치를 나타낼 때 상대적으로 높은 값을 가짐

![image.png](attachment:23f4a342-33cf-43e0-ab95-09eeabf37d99:image.png)

## ROC 곡선과 AUC

- ROC 곡선은 FPR이 변할 때 TPR이 어떻게 변하는지 나타내는 곡선
- 분류 성능지표로 사용되는 것은 ROC 곡선 면적에 기반한 AUC 값으로 결정
- AUC가 1에 가까울수록 좋은 수치
- roc_curve(), roc_auc_score()를 통해 ROC와 AUC 스코어를 알 수 있음
