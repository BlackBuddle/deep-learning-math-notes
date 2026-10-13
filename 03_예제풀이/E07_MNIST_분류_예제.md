---
title: "EX_07 예제 풀이: 손글씨 숫자 분류 (MNIST)"
type: note
tags: [deep-learning-math, dl-example, keras, mnist]
example: "01_ML_DL_Basic/EX_07_MNIST_Classification.ipynb"
concept: "D06_이미지분류와_MNIST"
created: 2026-10-13
updated: 2026-10-13
---

# EX\_07 예제 풀이 — 손글씨 숫자 분류 (MNIST)

[목차](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%88%98%ED%95%99_MOC.md) · 개념: [D06_이미지분류와_MNIST](../01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D06_%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%B6%84%EB%A5%98%EC%99%80_MNIST.md) · 이전: [E06_혈압_소프트맥스분류_예제](E06_%ED%98%88%EC%95%95_%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4%EB%B6%84%EB%A5%98_%EC%98%88%EC%A0%9C.md) · 다음: [E08_CNN_1Conv_MNIST_예제](E08_CNN_1Conv_MNIST_%EC%98%88%EC%A0%9C.md)

> [!NOTE]
> **3줄 요약**
> - 28×28 손글씨 그림 6만 장으로 0~9 숫자를 맞히는 모델을 만든다. [E06_혈압_소프트맥스분류_예제](E06_%ED%98%88%EC%95%95_%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4%EB%B6%84%EB%A5%98_%EC%98%88%EC%A0%9C.md)의 3등급 분류가 **10등급**이 된 것이다.
> - 그림을 한 줄(784칸)로 펴서(`Flatten`) `Dense(10, softmax)`에 넣는다. 입력이 2개에서 784개로 늘었을 뿐 구조는 같다.
> - 결과: 5번 학습 후 처음 보는 시험 그림 1만 장에서 정확도 **92.57%**.

## 0. 한눈에 보기

| 항목 | 내용 |
|---|---|
| 데이터 | MNIST: 학습용 60000장, 시험용 10000장, 각 28×28 흑백 |
| 정답 | 0~9 숫자 (원-핫이 아니라 숫자 그대로) |
| 전처리 | 픽셀 값 0~255를 0~1로 (÷ 255) |
| 모델 | `Flatten` → `Dense(10, activation='softmax')`, 가중치·편향 7850개 |
| 비용 | `sparse_categorical_crossentropy` |
| 학습 | Adam, 5번 |
| 결과 | 시험 데이터 loss 0.267, 정확도 0.9257 |

### EX\_06과 비교

| | EX\_06 | EX\_07 |
|---|---|---|
| 입력 | 숫자 2개 | 그림 28×28 = 784개 |
| 등급 수 | 3 | 10 |
| 정답 모양 | 원-핫 `[0,1,0]` | 숫자 그대로 `7` |
| 비용 | `categorical_crossentropy` | `sparse_categorical_crossentropy` |
| 최적화 | SGD | Adam |
| 확인 방법 | 학습 데이터로 확인 | **처음 보는 시험 데이터**로 확인 |

## 1. 셀별 풀이

### 1-1. 모듈 불러오기

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist     # Keras에 들어 있는 MNIST 데이터 불러오기 도구
```

### 1-2. MNIST 데이터 불러오기

```python
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
# 학습용(그림, 정답)과 시험용(그림, 정답) 네 덩어리를 한 번에 받음 (처음 실행 땐 인터넷에서 내려받음)

print("Type : {}".format(type(train_data)))      # 자료형
print("Shape : {}".format(train_data.shape))     # 모양
print("Type : {}".format(type(test_data)))
print("Shape : {}".format(test_data.shape))
print("Type : {}".format(type(train_labels)))
print("Shape : {}".format(train_labels.shape))
print("Type : {}".format(type(test_labels)))
print("Shape : {}".format(test_labels.shape))
```

**실행 결과**:

```
Type : <class 'numpy.ndarray'>
Shape : (60000, 28, 28)
Type : <class 'numpy.ndarray'>
Shape : (10000, 28, 28)
Type : <class 'numpy.ndarray'>
Shape : (60000,)
Type : <class 'numpy.ndarray'>
Shape : (10000,)
```

| 이름 | 모양 | 뜻 |
|---|---|---|
| train\_data | (60000, 28, 28) | 그림 6만 장, 각각 가로 28 × 세로 28칸 |
| test\_data | (10000, 28, 28) | 시험용 그림 1만 장 |
| train\_labels | (60000,) | 그림마다 정답 숫자 하나 (예: 5) |
| test\_labels | (10000,) | 시험용 정답 |

- 그림 한 장 = 28×28 숫자 표. 각 칸은 0(배경, 빈 곳) ~ 255(가장 진한 획)의 값이다. C의 `unsigned char img[28][28]`과 같다.
- **학습용과 시험용을 나누는 이유**: 학습에 쓴 문제로 시험을 보면 외워서 맞힐 수 있다. 처음 보는 그림을 맞혀야 진짜 실력이다.

### 1-3. 그림 10장 보기

```python
idx_show = np.arange(0, train_data.shape[0])     # [0, 1, 2, ..., 59999]
np.random.shuffle(idx_show)                      # 순서를 무작위로 섞음
idx_show = idx_show[0:10]                        # 앞에서 10개만 → 무작위 그림 번호 10개
fig_train, ax_train = plt.subplots(2,5,figsize=(16, 7))   # 2행 5열짜리 그림판
fig_train.suptitle("Train Data")                 # 전체 제목
for idx in range(10):
    ax_train[idx//5, idx%5].imshow(train_data[idx_show[idx]],cmap="binary")   # idx//5 = 행, idx%5 = 열
    ax_train[idx//5, idx%5].set_title("train_data[{}] (label : {})".format(idx_show[idx], train_labels[idx_show[idx]]))
```

![E07_cell6_1.png](../attachments/E07_cell6_1.png)

- `idx//5`, `idx%5`: idx = 7이면 7//5 = 1행, 7%5 = 2열. **몫과 나머지로 행·열을 구하는 방법**이다(파이썬 강의노트의 `//`, `%`).
- `cmap="binary"`: 0은 흰색, 큰 값일수록 검게 그리기. 그래서 흰 바탕에 검은 글씨로 보인다.

### 1-4. 0~1로 바꾸기

```python
train_data, test_data = train_data / 255.0, test_data / 255.0   # 픽셀 값 0~255를 0~1로
```

- 0 → 0.0, 128 → 0.502, 255 → 1.0
- [E04_다변수_다출력_회귀_예제](E04_%EB%8B%A4%EB%B3%80%EC%88%98_%EB%8B%A4%EC%B6%9C%EB%A0%A5_%ED%9A%8C%EA%B7%80_%EC%98%88%EC%A0%9C.md)의 Min-Max 정규화와 같은 목적이다. 픽셀의 최솟값이 0, 최댓값이 255라서 255로 나누기만 하면 된다.

### 1-5. 모델 정의

```python
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28)),                     # 입력: 28×28 그림 한 장
    tf.keras.layers.Flatten(),                         # 28×28 표를 한 줄 784칸으로 펴기
    # tf.keras.layers.Dense(256, activation='sigmoid'),   ← 중간층(주석 처리됨): 켜면 층이 깊어짐
    # tf.keras.layers.Dense(256, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')    # 출력 10개 = 숫자 0~9의 확률
])

model.summary()
```

**실행 결과**:

```
│ flatten (Flatten)               │ (None, 784)            │             0 │
│ dense (Dense)                   │ (None, 10)             │         7,850 │
 Total params: 7,850 (30.66 KB)
```

- **Flatten**: 2차원 표를 1차원 줄로 편다. 28 × 28 = 784. 가중치가 없어서 Param 0이다. C로 치면 `img[r][c]`를 `line[r*28 + c]`로 옮기는 것과 같다.
- **Dense(10)**: 가중치 784 × 10 = 7840 + 편향 10 = **7850**.
- 784개 픽셀 하나하나에 숫자별 가중치를 곱해 더하는 것이다. "이 칸이 까마면 7일 가능성이 올라간다" 같은 것을 배운다.

> [!NOTE]
> **주석 처리된 두 줄**
> `Dense(256, activation='sigmoid')`를 중간에 넣으면 784 → 256 → 256 → 10으로 층이 깊어진다(심층 신경망). 강의에서 켜고 끄며 비교해 보라는 의도로 보인다.

### 1-6. 컴파일

```python
model.compile(optimizer='adam',                          # Adam: 학습률을 알아서 조절하는 경사하강법
              loss='sparse_categorical_crossentropy',    # 정답이 원-핫이 아니라 숫자(7)일 때 쓰는 버전
              metrics=['accuracy'])
```

- **Adam**: 경사하강법의 개선판. 가중치마다 보폭을 자동으로 조절해서 학습률을 직접 맞추는 수고가 줄어든다.
- **sparse\_categorical\_crossentropy**: [E06_혈압_소프트맥스분류_예제](E06_%ED%98%88%EC%95%95_%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4%EB%B6%84%EB%A5%98_%EC%98%88%EC%A0%9C.md)의 `categorical_crossentropy`와 계산은 같다. 정답을 `[0,0,0,0,0,0,0,1,0,0]` 대신 `7`로 줘도 되게 한 것이다.

### 1-7. 학습

```python
history = model.fit(train_data, train_labels, epochs=5)   # 6만 장을 5바퀴
```

| Epoch | accuracy | loss |
|---|---|---|
| 1 | 0.8150 | 0.7134 |
| 2 | 0.9145 | 0.3099 |
| 3 | 0.9221 | 0.2837 |
| 4 | 0.9249 | 0.2718 |
| 5 | 0.9272 | 0.2578 |

- 한 바퀴(epoch)에 6만 장을 32장씩 묶어 60000 ÷ 32 = **1875번** 가중치를 고친다. 그래서 5바퀴만으로도 충분히 배운다.

### 1-8. 시험 데이터로 평가

```python
%%time
model.evaluate(test_data,  test_labels)    # 처음 보는 1만 장으로 시험 → [loss, accuracy]
```

**실행 결과**:

```
[0.2670372426509857, 0.9257000088691711]
```

- 시험 정확도 **92.57%**: 1만 장 중 약 9257장을 맞혔다.
- 학습 정확도(92.72%)와 비슷하다. 외운 게 아니라 일반적인 규칙을 배웠다는 뜻이다.
- 이 단순한 모델(층 1개)의 한계가 여기다. 다음 CNN 예제들이 이 숫자를 어디까지 올리는지 비교해 본다.

## 2. 이 예제에서 새로 나온 것

| 이름 | 뜻 | 관련 노트 |
|---|---|---|
| MNIST | 손글씨 숫자 그림 데이터 | |
| 학습/시험 데이터 | 배우는 데이터와 확인용 데이터를 나눔 | |
| shape (60000, 28, 28) | 그림 수 × 세로 × 가로 | [M05_벡터와_행렬](../02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M05_%EB%B2%A1%ED%84%B0%EC%99%80_%ED%96%89%EB%A0%AC.md), [D06_이미지분류와_MNIST](../01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D06_%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%B6%84%EB%A5%98%EC%99%80_MNIST.md) |
| ÷ 255 | 픽셀 값을 0~1로 | [E04_다변수_다출력_회귀_예제](E04_%EB%8B%A4%EB%B3%80%EC%88%98_%EB%8B%A4%EC%B6%9C%EB%A0%A5_%ED%9A%8C%EA%B7%80_%EC%98%88%EC%A0%9C.md) |
| `Flatten` | 표를 한 줄로 펴기 | |
| `optimizer='adam'` | 보폭을 자동 조절하는 경사하강법 | [D02_경사하강법으로_직선찾기](../01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D02_%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95%EC%9C%BC%EB%A1%9C_%EC%A7%81%EC%84%A0%EC%B0%BE%EA%B8%B0.md) |
| `sparse_categorical_crossentropy` | 정답이 숫자일 때의 분류 비용 | |
| `model.evaluate` | 시험 데이터로 [loss, accuracy] 계산 | |
| `plt.subplots(행, 열)` | 여러 그림 칸 만들기 | |

## 3. 스스로 확인

<details><summary>1. 그림이 32×32라면 Flatten 후 칸 수와 Dense(10)의 Param #는?</summary>

32 × 32 = **1024**칸, 1024 × 10 + 10 = **10250**

</details>

<details><summary>2. idx = 9일 때 `ax_train[idx//5, idx%5]`는 몇 행 몇 열인가?</summary>

9//5 = **1행**, 9%5 = **4열** (0부터 셈)

</details>

<details><summary>3. 왜 학습 데이터가 아니라 시험 데이터로 평가하나?</summary>

학습에 쓴 그림은 외워서 맞힐 수 있다. **처음 보는 그림**을 맞혀야 실제로 쓸 만한 모델이다.

</details>

