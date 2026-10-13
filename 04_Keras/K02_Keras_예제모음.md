---
title: Keras 예제 모음
type: note
tags: [deep-learning-math, keras]
created: 2026-10-13
updated: 2026-10-13
cssclasses: [dl-note]
---

# Keras 예제 모음

[목차로 돌아가기](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%88%98%ED%95%99_MOC.md) · 개념: [K01_Keras_기본개념](../04_Keras/K01_Keras_%EA%B8%B0%EB%B3%B8%EA%B0%9C%EB%85%90.md)

> [!NOTE]
> **3줄 요약**
> - 1절은 예제 노트북에서 뽑은 **문제 유형별 최소 코드**다(실행 기록 있음). 새 문제를 풀 때 가장 비슷한 것을 복사해서 시작하면 된다.
> - 2~6절은 예제를 **조금씩 바꿔 보는 연습**이다. [파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수([가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)+[편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5))는 계산해서 확인했다.
> - 이 PC에는 TensorFlow가 없어서 2~6절 코드는 아직 실행하지 않았다. Colab에서 돌려 보고 결과를 아래 "내 결과" 칸에 적어 두자.

> [!NOTE]
> **실행 여부 표시**
> - ✅ **실행 기록 있음**: 원래 노트북에서 실행된 코드와 출력
> - 🧪 **바꿔 보기**: 노트북 코드를 고친 것. 아직 실행하지 않음. [파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수 같은 계산값만 확인함
> - 1절 안에서도 1.5절은 노트북을 줄여 다시 쓴 코드라 🧪에 해당한다.

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

결과: [손실](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98) 11027.59 → 5.36 ([E04_다변수_다출력_회귀_예제](../03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E04_%EB%8B%A4%EB%B3%80%EC%88%98_%EB%8B%A4%EC%B6%9C%EB%A0%A5_%ED%9A%8C%EA%B7%80_%EC%98%88%EC%A0%9C.md))

`epochs=1000`은 [에폭](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#epoch-%EC%97%90%ED%8F%AD)(epoch, 전체 데이터를 한 바퀴 다 보는 것) 수, 즉 1000에폭 학습한다는 뜻이다.

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

결과: [정확도](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#accuracy-%EC%A0%95%ED%99%95%EB%8F%84) 0.5556 → 0.7778 ([E05_혈압_로지스틱회귀_예제](../03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E05_%ED%98%88%EC%95%95_%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1%ED%9A%8C%EA%B7%80_%EC%98%88%EC%A0%9C.md))

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

결과: [정확도](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#accuracy-%EC%A0%95%ED%99%95%EB%8F%84) 0.3333 → 0.7222 ([E06_혈압_소프트맥스분류_예제](../03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E06_%ED%98%88%EC%95%95_%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4%EB%B6%84%EB%A5%98_%EC%98%88%EC%A0%9C.md))

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

결과: 시험 [정확도](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#accuracy-%EC%A0%95%ED%99%95%EB%8F%84) 0.9257 ([E07_MNIST_분류_예제](../03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E07_MNIST_%EB%B6%84%EB%A5%98_%EC%98%88%EC%A0%9C.md))

### 1.5 CNN (CNN EX\_03)

```python
train_data = np.expand_dims(train_data.astype(np.float32), axis=-1) / 255.0   # 맨 끝에 채널 칸 추가 → (60000,28,28,1), 255로 나눠 0~1로
test_data = np.expand_dims(test_data.astype(np.float32), axis=-1) / 255.0     # 같은 처리 → (10000,28,28,1)
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

결과: 시험 [정확도](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#accuracy-%EC%A0%95%ED%99%95%EB%8F%84) 0.9915 ([E10_CNN_ReLU_Dropout_예제](../03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E10_CNN_ReLU_Dropout_%EC%98%88%EC%A0%9C.md)).

🧪 이 절의 코드는 노트북을 줄여 다시 쓴 것이라, 이 모양 그대로 실행하지는 않았다. 줄인 곳은 두 군데다.

- 데이터 준비 두 줄: 노트북에서 여러 줄이던 것을 학습용 한 줄, 시험용 한 줄로 줄였다. [시험 데이터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0)에도 같은 처리를 해야 `validation_data`의 모양이 모델 입력 (28, 28, 1)과 맞는다.
- for문: 노트북에서 같은 세 줄을 3번 쓴 것을 반복문으로 줄였다.

노트북과 같은 계산·같은 [층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) 순서이므로 [파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수도 80,266으로 같다.

## 2. 🧪 중간층을 켠 EX\_07

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

**예상 [summary](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#summary-%EB%AA%A8%EB%8D%B8-%EC%9A%94%EC%95%BD)** (계산):

| [층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) | 출력 모양 | [파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수 | 계산 |
|---|---|---|---|
| flatten | (None, 784) | 0 | |
| dense | (None, 256) | 200,960 | 784 × 256 + 256 |
| dense\_1 | (None, 256) | 65,792 | 256 × 256 + 256 |
| dense\_2 | (None, 10) | 2,570 | 256 × 10 + 10 |
| 합계 | | **269,322** | 원래 7,850의 약 34배 |

**볼 것**: 시험 [정확도](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#accuracy-%EC%A0%95%ED%99%95%EB%8F%84)가 0.9257보다 오르는가? `'sigmoid'`를 `'relu'`로 바꾸면 어떻게 되는가([D08_ReLU와_드롭아웃](../01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D08_ReLU%EC%99%80_%EB%93%9C%EB%A1%AD%EC%95%84%EC%9B%83.md))?

<details><summary>내 결과</summary>



</details>

## 3. 🧪 Functional API로 바꾼 EX\_05

```python
inputs = tf.keras.Input(shape=(2,))                                    # 입력 정의
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(inputs)       # 층을 함수처럼 입력에 적용
model = tf.keras.Model(inputs=inputs, outputs=outputs)                 # 입력과 출력을 묶어 모델로
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='binary_crossentropy', metrics=['accuracy'])
```

- 1.2절의 [Sequential](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#sequential-%EC%88%9C%EC%B0%A8-%EB%AA%A8%EB%8D%B8) 모델과 **같은 모델**이다([파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수 3).
- `층(...)(입력)`처럼 [층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)을 만든 뒤 바로 괄호로 입력을 넣는 모양이 [Functional API](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#functional-api-%ED%95%A8%EC%88%98%ED%98%95-api)의 특징이다.

<details><summary>내 결과</summary>



</details>

## 4. 🧪 번호 정답으로 바꾼 EX\_06

[원-핫](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#one-hot-encoding-%EC%9B%90-%ED%95%AB-%EC%9D%B8%EC%BD%94%EB%94%A9) 대신 등급 번호를 쓰고, [손실 함수](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)에 `sparse_`를 붙인다. 이미 학습한 모델에 이어 쓰지 말고, 1.3절처럼 새 모델을 만든 뒤 아래 두 줄을 적용한다.

```python
# 새 모델을 만든 뒤 (1.3절의 model 정의를 다시 실행)
labels = tf.constant([0,1,2,0,1,0,0,0,1,0,1,2,1,2,2,1,2,2], dtype=tf.int32)     # 원-핫의 argmax와 같은 번호 (정수)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

- 계산 내용은 같다([정답](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5) 칸 [확률](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)의 −ln). [정답](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5)을 쓰는 방법만 다르다([D05_소프트맥스와_다중분류](../01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D05_%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4%EC%99%80_%EB%8B%A4%EC%A4%91%EB%B6%84%EB%A5%98.md)의 '범주형 [교차 엔트로피](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cross-entropy-%EA%B5%90%EC%B0%A8-%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC)').
- 이 번호 18개는 노트북 [원-핫](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#one-hot-encoding-%EC%9B%90-%ED%95%AB-%EC%9D%B8%EC%BD%94%EB%94%A9) [정답](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5)을 `np.argmax(..., axis=1)`로 바꾼 것과 같다.

<details><summary>내 결과</summary>



</details>

## 5. 🧪 과적합 방지용 조기 종료 (EarlyStopping)

[과적합](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#overfitting-%EA%B3%BC%EC%A0%81%ED%95%A9)은 [학습 데이터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0)에는 잘 맞는데 새 데이터에는 틀리는 상태다. [검증](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#validation-data-%EA%B2%80%EC%A6%9D-%EB%8D%B0%EC%9D%B4%ED%84%B0) [손실](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 더 이상 줄지 않고 오르기 시작하면 [과적합](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#overfitting-%EA%B3%BC%EC%A0%81%ED%95%A9)이 시작된 것이므로, 그 지점에서 [학습](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5)을 멈춘다.

```python
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',           # 검증 손실을 지켜봄
                                         patience=2,                   # 최고 기록을 2에폭 동안 못 넘기면 멈춘다
                                         restore_best_weights=True)    # 가장 좋았던 가중치로 되돌리고 멈춤
history = model.fit(train_data, train_labels, epochs=30,
                    validation_data=(test_data, test_labels),
                    callbacks=[early])                                 # fit에 넘김
```

- [CNN](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cnn-convolutional-neural-network-%ED%95%A9%EC%84%B1%EA%B3%B1-%EC%8B%A0%EA%B2%BD%EB%A7%9D) EX\_02([E09_CNN_3Conv_MNIST_예제](../03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E09_CNN_3Conv_MNIST_%EC%98%88%EC%A0%9C.md))처럼 [검증](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#validation-data-%EA%B2%80%EC%A6%9D-%EB%8D%B0%EC%9D%B4%ED%84%B0) [손실](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 최고 기록을 2에폭 동안 못 넘기면 알아서 멈춘다.
- **[콜백](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#callback-%EC%BD%9C%EB%B0%B1)**(callback): [fit](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#fit-%ED%95%99%EC%8A%B5-%EC%8B%A4%ED%96%89)이 [에폭](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#epoch-%EC%97%90%ED%8F%AD)마다 불러 주는 함수. 멈추기, 저장하기 등에 쓴다.

> [!WARNING]
> **⚠ 주의**
> 위 코드는 [시험 데이터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0)를 `validation_data`로 넣어 "가장 좋은 [에폭](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#epoch-%EC%97%90%ED%8F%AD)"을 골랐다. 그러면 그 뒤에 같은 [시험 데이터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0)로 잰 점수는 실제보다 조금 낙관적이다. 실제로는 `validation_split=0.1`을 쓰거나 검증용 데이터를 따로 떼어 두고, [시험 데이터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0)는 마지막에 한 번만 쓴다.

<details><summary>내 결과</summary>



</details>

## 6. 🧪 학습한 모델의 저장과 불러오기

```python
model.save('mnist_cnn.keras')                                  # 파일로 저장 (확장자 .keras)
loaded = tf.keras.models.load_model('mnist_cnn.keras')         # 다시 불러오기
loaded.evaluate(test_data, test_labels)                        # 저장 전과 같은 결과가 나와야 함
```

- 매번 다시 학습하지 않고 저장해 둔 모델(구조와 [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98))을 쓸 수 있다.

<details><summary>내 결과</summary>



</details>

## 7. 개념 재확인

<details><summary>1. 2절 모델에서 첫 덴스 층의 파라미터 수가 200,960인 이유는?</summary>

| 단계 | 하는 일 | 코드 | 계산 | 결과 |
|---|---|---|---|---|
| 1 | 입력 칸 수 | `Flatten()` | 28 × 28 | n = 784 |
| 2 | [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 개수 | `Dense(256, ...)` | 784 × 256 | 200,704 |
| 3 | [편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5) 개수 (출력 칸마다 하나) | `Dense(256, ...)` | 출력 칸 수 256 | 256 |
| 4 | 둘을 더하기 | `model.summary()` | 200,704 + 256 | [파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수 = 200,960 |

**① 쓰는 코드**
```python
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28)),                      # ① 입력 한 장의 모양 (28 × 28 표)
    tf.keras.layers.Flatten(),                          # ② 한 줄로 펴서 28 × 28 = 784칸
    tf.keras.layers.Dense(256, activation='sigmoid'),   # ③ 가중치 784 × 256개, ④ 편향 256개
    tf.keras.layers.Dense(256, activation='sigmoid'),   # ① 이 문제는 첫 덴스 층만 봄
    tf.keras.layers.Dense(10, activation='softmax')     # ① 이 문제는 첫 덴스 층만 봄
])
model.summary()                                         # ⑤ 첫 덴스 층의 Param #가 200,960으로 나옴
```

**② 입력 칸 수 구하기** (`Flatten()`)
- 28 × 28 = 784
- 입력 칸 수 n = 784

**③ [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 개수 구하기** (`Dense(256, ...)`)
- 입력 칸 784개가 출력 칸 256개 모두에 하나씩 연결됨
- 784 × 256 = 200,704

**④ [편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5) 개수 구하기** (`Dense(256, ...)`)
- [편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5)은 출력 칸마다 하나
- 출력 칸 수 256 = 256개

**⑤ 둘을 더하기** (`model.summary()`)
- 200,704 + 256 = 200,960
- 식으로는 입력 수 × 출력 수 + 출력 수 = 784 × 256 + 256

**답**: [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 200,704개 + [편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5) 256개 = **200,960**이다.

</details>

<details><summary>2. Functional API로 만든 모델과 Sequential 모델은 학습 결과가 다른가?</summary>

| 단계 | 하는 일 | 코드 | 계산 | 결과 |
|---|---|---|---|---|
| 1 | [Sequential](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#sequential-%EC%88%9C%EC%B0%A8-%EB%AA%A8%EB%8D%B8)로 만들기 | `Sequential([Input(shape=(2,)), Dense(1, ...)])` | 입력 2칸 → [덴스 층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#dense-layer-%EB%8D%B4%EC%8A%A4-%EC%B8%B5) 1개 | [파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수 = 2 × 1 + 1 = 3 |
| 2 | Functional로 만들기 | `Dense(1, ...)(inputs)` | 입력 2칸 → [덴스 층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#dense-layer-%EB%8D%B4%EC%8A%A4-%EC%B8%B5) 1개 | [파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수 = 2 × 1 + 1 = 3 |
| 3 | [층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) 잇는 문법 비교 | `Model(inputs=..., outputs=...)` | 리스트 순서 대 `층(입력)` | 문법만 다름, 구조는 같음 |
| 4 | [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 시작값 | - | 무작위로 시작 | 같은 [Sequential](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#sequential-%EC%88%9C%EC%B0%A8-%EB%AA%A8%EB%8D%B8)을 두 번 돌려도 결과 조금 다름 |

**① 쓰는 코드**
```python
seq = tf.keras.models.Sequential([                          # ① Sequential: 리스트로 층을 나열
  tf.keras.Input(shape=(2,)),                               # ② 입력 2칸
  tf.keras.layers.Dense(1, activation='sigmoid')            # ② 덴스 층 1개 → 파라미터 2 × 1 + 1 = 3
])
inputs = tf.keras.Input(shape=(2,))                         # ③ Functional: 입력 2칸
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(inputs)   # ③ 덴스 층 1개 → 파라미터 3
func = tf.keras.Model(inputs=inputs, outputs=outputs)       # ④ 입력과 출력을 묶음 (문법만 다름)
# ⑤ 두 모델 모두 가중치는 무작위 값으로 시작함
```

**② [Sequential](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#sequential-%EC%88%9C%EC%B0%A8-%EB%AA%A8%EB%8D%B8)로 만들기** (`Sequential([Input(shape=(2,)), Dense(1, ...)])`)
- 입력 2칸 → [덴스 층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#dense-layer-%EB%8D%B4%EC%8A%A4-%EC%B8%B5) 1개
- [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98): 2 × 1 = 2
- [편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5): 출력 칸마다 하나 → 1
- [파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수: 2 + 1 = 3

**③ Functional로 만들기** (`Dense(1, ...)(inputs)`)
- 입력 2칸 → [덴스 층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#dense-layer-%EB%8D%B4%EC%8A%A4-%EC%B8%B5) 1개
- [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98): 2 × 1 = 2
- [편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5): 1
- [파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수: 2 + 1 = 3 → ②와 같음

**④ [층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) 잇는 문법 비교** (`Model(inputs=..., outputs=...)`)
- [Sequential](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#sequential-%EC%88%9C%EC%B0%A8-%EB%AA%A8%EB%8D%B8)은 리스트 순서대로 이어 붙임
- Functional은 `층(입력)`처럼 함수로 이어 붙이고 `Model(inputs, outputs)`로 묶음
- 문법만 다르고 구조는 같음

**⑤ [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 시작값**
- 시작 [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)는 무작위
- 그래서 같은 [Sequential](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#sequential-%EC%88%9C%EC%B0%A8-%EB%AA%A8%EB%8D%B8)을 두 번 돌려도 결과가 조금 다름
- 결과 차이는 모델 종류가 아니라 이 시작값에서 생김

[층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)의 종류, 순서, 모양이 같으면 계산도 같다 ([K01_Keras_기본개념](../04_Keras/K01_Keras_%EA%B8%B0%EB%B3%B8%EA%B0%9C%EB%85%90.md)의 3절 "모델 구성의 세 가지 방법").

**답**: 같은 [층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)을 같은 순서로 연결했다면 **같은 모델**이다. 문법만 다르고, 결과 차이는 무작위 초기값에서만 생긴다.

</details>

<details><summary>내 이해</summary>



</details>
