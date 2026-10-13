---
title: Keras 예제 모음
type: note
tags: [deep-learning-math, keras]
created: 2026-10-13
updated: 2026-10-13
---

# Keras 예제 모음

[목차로 돌아가기](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%88%98%ED%95%99_MOC.md) · 개념: [K01_Keras_기본개념](K01_Keras_%EA%B8%B0%EB%B3%B8%EA%B0%9C%EB%85%90.md)

> [!NOTE]
> **3줄 요약**
> - 1절은 예제 노트북에서 뽑은 **문제 유형별 최소 코드**다(실행 기록 있음). 새 문제를 풀 때 가장 비슷한 것을 복사해서 시작하면 된다.
> - 2~5절은 예제를 **조금씩 바꿔 보는 연습**이다. 가중치 개수는 계산해서 확인했다.
> - 이 PC에는 TensorFlow가 없어서 2~5절 코드는 아직 실행하지 않았다. Colab에서 돌려 보고 결과를 아래 "내 결과" 칸에 적어 두자.

> [!NOTE]
> **실행 여부 표시**
> - ✅ **실행 기록 있음**: 원래 노트북에서 실행된 코드와 출력
> - 🧪 **바꿔 보기**: 노트북 코드를 고친 것. 아직 실행하지 않음. Param # 같은 계산값만 확인함

## 1. 문제 유형별 최소 코드 ✅

### 1.1 회귀: 숫자 맞히기 (EX\_04)

```python
x = (x - x_min) / (x_max - x_min)                       # 입력 정규화
model = tf.keras.models.Sequential([
  tf.keras.Input(shape=(2,)),                           # 입력 2개
  tf.keras.layers.Dense(2)                              # 출력 2개, 활성화 없음
])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='mean_squared_error')
history = model.fit(x, labels, epochs=1000)
```

결과: loss 11027.59 → 5.36 ([E04_다변수_다출력_회귀_예제](../03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E04_%EB%8B%A4%EB%B3%80%EC%88%98_%EB%8B%A4%EC%B6%9C%EB%A0%A5_%ED%9A%8C%EA%B7%80_%EC%98%88%EC%A0%9C.md))

### 1.2 이진 분류 (EX\_05)

```python
model = tf.keras.models.Sequential([
  tf.keras.Input(shape=(2,)),
  tf.keras.layers.Dense(1, activation='sigmoid')        # 1일 확률 하나
])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x, labels, epochs=1000)             # labels: [[0],[1],...]
```

결과: 정확도 0.5556 → 0.7778 ([E05_혈압_로지스틱회귀_예제](../03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E05_%ED%98%88%EC%95%95_%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1%ED%9A%8C%EA%B7%80_%EC%98%88%EC%A0%9C.md))

### 1.3 다중 분류, 원-핫 정답 (EX\_06)

```python
model = tf.keras.models.Sequential([
  tf.keras.Input(shape=(2,)),
  tf.keras.layers.Dense(3, activation='softmax')        # 3등급 확률
])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x, labels, epochs=1000)             # labels: [[1,0,0],[0,1,0],...]
```

결과: 정확도 0.3333 → 0.7222 ([E06_혈압_소프트맥스분류_예제](../03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E06_%ED%98%88%EC%95%95_%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4%EB%B6%84%EB%A5%98_%EC%98%88%EC%A0%9C.md))

### 1.4 이미지 분류, 번호 정답 (EX\_07)

```python
train_data, test_data = train_data / 255.0, test_data / 255.0
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),                          # 784칸으로 펴기
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=5)
model.evaluate(test_data, test_labels)
```

결과: 시험 정확도 0.9257 ([E07_MNIST_분류_예제](../03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E07_MNIST_%EB%B6%84%EB%A5%98_%EC%98%88%EC%A0%9C.md))

### 1.5 CNN (CNN EX\_03)

```python
train_data = np.expand_dims(train_data.astype(np.float32), axis=-1) / 255.0   # (60000,28,28,1)
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(28, 28, 1)))
for _ in range(3):                                      # 노트북은 같은 세 줄을 3번 썼다. 반복문으로 줄인 것
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.4))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, shuffle=True, epochs=10,
                    validation_data=(test_data, test_labels))
```

결과: 시험 정확도 0.9915 ([E10_CNN_ReLU_Dropout_예제](../03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E10_CNN_ReLU_Dropout_%EC%98%88%EC%A0%9C.md)).
첫 줄(데이터 세 줄을 한 줄로)과 for문은 노트북 코드를 줄여 쓴 것이라 이 모양 그대로 실행하지는 않았다. 노트북과 같은 계산·같은 층 순서이므로 Param #도 80,266으로 같다.

## 2. 🧪 EX\_07에 중간층 켜기

노트북에 주석으로 남아 있던 두 줄을 켠다.

```python
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='sigmoid'),   # 주석 해제
    tf.keras.layers.Dense(256, activation='sigmoid'),   # 주석 해제
    tf.keras.layers.Dense(10, activation='softmax')
])
```

**예상 summary** (계산):

| 층 | 출력 모양 | Param # | 계산 |
|---|---|---|---|
| flatten | (None, 784) | 0 | |
| dense | (None, 256) | 200,960 | 784 × 256 + 256 |
| dense\_1 | (None, 256) | 65,792 | 256 × 256 + 256 |
| dense\_2 | (None, 10) | 2,570 | 256 × 10 + 10 |
| 합계 | | **269,322** | 원래 7,850의 약 34배 |

**볼 것**: 시험 정확도가 0.9257보다 오르는가? `'sigmoid'`를 `'relu'`로 바꾸면 어떻게 되는가([D08_ReLU와_드롭아웃](../01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D08_ReLU%EC%99%80_%EB%93%9C%EB%A1%AD%EC%95%84%EC%9B%83.md))?

> [!NOTE]
> **내 결과**: 직접 실행해 보고 여기에 적어 보자.

## 3. 🧪 EX\_05를 Functional API로

```python
inputs = tf.keras.Input(shape=(2,))                                    # 입력 정의
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(inputs)       # 층을 함수처럼 입력에 적용
model = tf.keras.Model(inputs=inputs, outputs=outputs)                 # 입력과 출력을 묶어 모델로
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='binary_crossentropy', metrics=['accuracy'])
```

- 1.2절의 Sequential 모델과 **같은 모델**이다(Param # 3).
- `층(...)(입력)`처럼 층을 만든 뒤 바로 괄호로 입력을 넣는 모양이 Functional API의 특징이다.

> [!NOTE]
> **내 결과**: 직접 실행해 보고 여기에 적어 보자.

## 4. 🧪 EX\_06을 번호 정답으로

원-핫 대신 등급 번호를 쓰고, loss에 `sparse_`를 붙인다.

```python
labels = tf.constant([0,1,2,0,1,0,0,0,1,0,1,2,1,2,2,1,2,2], dtype=tf.int32)     # 원-핫의 argmax와 같은 번호 (정수)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

- 계산 내용은 같다(정답 칸 확률의 −ln). 정답을 쓰는 방법만 다르다([D05_소프트맥스와_다중분류](../01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D05_%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4%EC%99%80_%EB%8B%A4%EC%A4%91%EB%B6%84%EB%A5%98.md) 4.4절).
- 이 번호 18개는 노트북 원-핫 정답을 `np.argmax(..., axis=1)`로 바꾼 것과 같다.

> [!NOTE]
> **내 결과**: 직접 실행해 보고 여기에 적어 보자.

## 5. 🧪 과적합이 시작되면 멈추기 (EarlyStopping)

```python
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',           # 검증 손실을 지켜봄
                                         patience=2,                   # 2바퀴 연속 안 좋아지면
                                         restore_best_weights=True)    # 가장 좋았던 가중치로 되돌리고 멈춤
history = model.fit(train_data, train_labels, epochs=30,
                    validation_data=(test_data, test_labels),
                    callbacks=[early])                                 # fit에 넘김
```

- CNN EX\_02([E09_CNN_3Conv_MNIST_예제](../03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E09_CNN_3Conv_MNIST_%EC%98%88%EC%A0%9C.md))처럼 검증 손실이 다시 오르기 시작하면 알아서 멈춘다.
- **콜백**(callback): fit이 바퀴마다 불러 주는 함수. 멈추기, 저장하기 등에 쓴다.

> [!NOTE]
> **내 결과**: 직접 실행해 보고 여기에 적어 보자.

## 6. 🧪 학습한 모델 저장하고 불러오기

```python
model.save('mnist_cnn.keras')                                  # 파일로 저장 (확장자 .keras)
loaded = tf.keras.models.load_model('mnist_cnn.keras')         # 다시 불러오기
loaded.evaluate(test_data, test_labels)                        # 저장 전과 같은 결과가 나와야 함
```

- 매번 10분씩 다시 학습하지 않고 저장해 둔 가중치를 쓸 수 있다.

> [!NOTE]
> **내 결과**: 직접 실행해 보고 여기에 적어 보자.

## 7. 스스로 확인

<details><summary>1. 2절 모델에서 첫 Dense의 Param #가 200,960인 이유는?</summary>

입력 784 × 출력 256 = 200,704, 편향 256 → **200,960**

</details>

<details><summary>2. Functional API로 만든 모델과 Sequential 모델은 학습 결과가 다른가?</summary>

같은 층을 같은 순서로 연결했다면 **같은 모델**이다. 만드는 문법만 다르다.

</details>

