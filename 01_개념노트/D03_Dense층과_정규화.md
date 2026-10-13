---
title: 덴스 층과 정규화
type: note
tags: [deep-learning-math, dl-concept]
example: "01_ML_DL_Basic/EX_04_N-variable_M-output_LR_Keras.ipynb"
created: 2026-10-13
updated: 2026-10-13
cssclasses: [dl-note]
---

# 덴스 층과 정규화 (EX\_04)

[목차로 돌아가기](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%88%98%ED%95%99_MOC.md) · 예제: `EX_04_N-variable_M-output_LR_Keras.ipynb` · 코드 풀이: [E04_다변수_다출력_회귀_예제](../03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E04_%EB%8B%A4%EB%B3%80%EC%88%98_%EB%8B%A4%EC%B6%9C%EB%A0%A5_%ED%9A%8C%EA%B7%80_%EC%98%88%EC%A0%9C.md)

> [!NOTE]
> **3줄 요약**
> - `Dense` [층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)은 **[퍼셉트론](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#perceptron-%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0)([가중합](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weighted-sum-%EA%B0%80%EC%A4%91%ED%95%A9)) 여러 개를 나란히 둔 것**이고, 계산은 [행렬 곱](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#matrix-multiplication-%ED%96%89%EB%A0%AC-%EA%B3%B1) $XW + b$ 한 줄이다.
> - [파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수([가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)+[편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5))는 "입력 수 × 출력 수 + 출력 수"다. EX\_04는 2 × 2 + 2 = 6.
> - Min-Max [정규화](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#normalization-%EC%A0%95%EA%B7%9C%ED%99%94)로 입력을 0~1로 맞추면 [기울기](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 크기가 고르게 되어 큰 [학습률](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)로 빨리 배운다(EX\_03: 0.0005·20만 번 → EX\_04: 0.1·1000번).

## 1. 개념 설명

**[덴스 층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#dense-layer-%EB%8D%B4%EC%8A%A4-%EC%B8%B5)**

> 비유: 심사위원 2명(출력)이 지원자의 나이·BMI(입력)를 보고 각자 점수를 매긴다. 심사위원마다 항목별 반영 비율([가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98))과 기본 점수([편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5))가 다르다.

정확히 말하면 [덴스 층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#dense-layer-%EB%8D%B4%EC%8A%A4-%EC%B8%B5)은 입력 여러 개를 받아 출력 여러 개를 내는 [층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)이다. 출력 하나하나가 [D01_퍼셉트론](../01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D01_%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0.md)처럼 "입력마다 [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)를 곱해 더하고 [편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5)을 더한" 값이다. 모든 입력이 모든 출력에 연결돼 있어서 "빽빽한(dense)" [층](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)이라고 부른다.

**Min-Max [정규화](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#normalization-%EC%A0%95%EA%B7%9C%ED%99%94)**: 단위와 범위가 다른 입력(나이 25~73, BMI 22~30)을 모두 0~1 범위로 맞춘다.

## 2. 필요한 이유

[D01_퍼셉트론](../01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D01_%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0.md)은 출력이 하나뿐이고, [D02_경사하강법으로_직선찾기](../01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D02_%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95%EC%9C%BC%EB%A1%9C_%EC%A7%81%EC%84%A0%EC%B0%BE%EA%B8%B0.md)의 직선은 입력이 x 하나뿐이었다. 나이·BMI처럼 입력이 여러 개이고 수축기·이완기처럼 맞힐 값도 여러 개이면, [퍼셉트론](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#perceptron-%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0)을 하나씩 따로 계산하는 대신 [행렬 곱](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#matrix-multiplication-%ED%96%89%EB%A0%AC-%EA%B3%B1) 한 줄로 묶어야 한다. 또 나이(25~73)처럼 값이 큰 입력을 그대로 쓰면 D02의 마지막 경고처럼 [학습률](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)을 아주 작게 해야 해서 [학습](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5)이 느려진다. [정규화](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#normalization-%EC%A0%95%EA%B7%9C%ED%99%94)는 이 문제를 줄인다.

## 3. 필요한 수학 개념

| 수학 개념 | 이 개념에서 쓰이는 곳 | 자세히 |
|---|---|---|
| 벡터와 행렬 | [행렬 곱](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#matrix-multiplication-%ED%96%89%EB%A0%AC-%EA%B3%B1), shape, [파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수([가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)+[편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5)) | [M05_벡터와_행렬](../02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M05_%EB%B2%A1%ED%84%B0%EC%99%80_%ED%96%89%EB%A0%AC.md) |
| 시그마와 평균 | 출력이 여러 개일 때의 [MSE](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#mse-mean-squared-error-%ED%8F%89%EA%B7%A0-%EC%A0%9C%EA%B3%B1-%EC%98%A4%EC%B0%A8) 평균 | [M02_시그마와_평균](../02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M02_%EC%8B%9C%EA%B7%B8%EB%A7%88%EC%99%80_%ED%8F%89%EA%B7%A0.md) |
| 일차함수 | [정규화](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#normalization-%EC%A0%95%EA%B7%9C%ED%99%94)는 일차함수다 | [M01_함수와_일차함수](../02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M01_%ED%95%A8%EC%88%98%EC%99%80_%EC%9D%BC%EC%B0%A8%ED%95%A8%EC%88%98.md) |

<details><summary>미니 복습: 행렬 곱</summary>

왼쪽 한 줄과 오른쪽 한 열을 같은 자리끼리 곱해 더한 값이 결과 한 칸. (a × **k**) @ (**k** × c) → (a × c). 가운데 숫자 k가 같아야 곱할 수 있다. `@`는 파이썬에서 [행렬 곱](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#matrix-multiplication-%ED%96%89%EB%A0%AC-%EA%B3%B1)을 하는 기호다. 자세히: [M05_벡터와_행렬](../02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M05_%EB%B2%A1%ED%84%B0%EC%99%80_%ED%96%89%EB%A0%AC.md)

</details>

<details><summary>미니 복습: 평균</summary>

전부 더해서 개수로 나눈다. 출력이 2개 × 18명이면 [오차](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) 36개의 평균. 자세히: [M02_시그마와_평균](../02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M02_%EC%8B%9C%EA%B7%B8%EB%A7%88%EC%99%80_%ED%8F%89%EA%B7%A0.md)

</details>

## 4. 계산 과정 (숫자 → 식 → 코드)

### 숫자

#### 덴스 층 손계산

[가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)를 임의로 정해서 Dense(2)가 하는 계산을 직접 해 본다([M05_벡터와_행렬](../02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M05_%EB%B2%A1%ED%84%B0%EC%99%80_%ED%96%89%EB%A0%AC.md) 4절 숫자). [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) W는 [`[3, 1]`, `[2, 4]`](입력 2개 × 출력 2개), [편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5) b는 `[1, 1]`(출력마다 하나)이다. 출력 하나는 "그 출력의 [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 열과 입력을 같은 자리끼리 곱해 더하고, [편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5)을 더한 값"이다.

첫 사람의 입력 `[1, 2]`로 계산하면:

- 출력 1: 1×3 + 2×2 + 1 = 8 (W의 첫 열 `[3, 2]`와 곱해 더하고, b의 첫 값 1을 더함)
- 출력 2: 1×1 + 2×4 + 1 = 10 (W의 둘째 열 `[1, 4]`와 곱해 더하고, b의 둘째 값 1을 더함)
- 그래서 첫 줄은 `[8, 10]`이다.

#### Min-Max 정규화 계산

나이는 25~73(폭 48), BMI는 22~30(폭 8)이다. 계산 방법은 "최솟값을 빼고, 폭(최댓값 − 최솟값)으로 나눈다"이다.

| 원래 값 | 계산 | [정규화](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#normalization-%EC%A0%95%EA%B7%9C%ED%99%94) |
|---|---|---|
| 나이 25 (최소) | (25 − 25) ÷ 48 | 0 |
| 나이 50 | (50 − 25) ÷ 48 | 0.5208 |
| 나이 73 (최대) | (73 − 25) ÷ 48 | 1 |
| BMI 25 | (25 − 22) ÷ 8 | 0.375 |

### 식

#### 덴스 층의 식

한 사람의 입력 $x = [x_1, x_2]$ (정규화한 나이, BMI), 출력 $[\hat{y}_1, \hat{y}_2]$ (수축기, 이완기)는 출력마다 "입력×[가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)의 합 + [편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5)"이다.

$$\hat{y}_1 = w_{11}x_1 + w_{21}x_2 + b_1 \qquad \hat{y}_2 = w_{12}x_1 + w_{22}x_2 + b_2$$

이 두 줄을 행렬로 묶으면:

$$\hat{y} = x\,W + b, \qquad W = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix}$$

18명을 한 번에 계산하면 X가 (18 × 2) 행렬이 되고:

$$\underbrace{\hat{Y}}_{18 \times 2} = \underbrace{X}_{18 \times 2}\;\underbrace{W}_{2 \times 2} + \underbrace{b}_{2}$$

<details><summary>기호 풀이</summary>

| 기호 | 읽는 법 | 뜻 | 예제 값 |
| --- | --- | --- | --- |
| $x_1,\ x_2$ | 엑스 일, 엑스 이 | 한 사람의 입력(정규화한 나이, BMI) | 숫자 절의 `[1, 2]` |
| $\hat{y}_1,\ \hat{y}_2$ | 와이 햇 일, 와이 햇 이 | 한 사람의 예측 출력(수축기, 이완기) | 8, 10 |
| $w_{ij}$ | 더블유 아이제이 | i번째 입력이 j번째 출력에 주는 [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) | $w_{11}$ = 3, $w_{21}$ = 2 |
| $X$ | 엑스(대문자) | 입력 행렬 (사람 × 항목) | 18 × 2 |
| $W$ | 더블유(대문자) | [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 행렬 (입력 × 출력). **열 하나 = 출력 하나** | 2 × 2 = 4개 |
| $b$ | 비 | [편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5) (출력마다 하나) | 2개 |
| $\hat{Y}$ | 와이 햇(대문자) | 예측 (사람 × 출력) | 18 × 2 |

</details>

b는 18줄 모두에 똑같이 더해진다(브로드캐스팅: 모양이 다른 b를 numpy가 알아서 각 줄에 같이 더해 주는 규칙).

[파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수 = [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 4 + [편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5) 2 = **6** → `model.summary()`의 6과 같다.

#### 출력이 2개일 때의 MSE

출력이 2개라서, 모든 사람과 모든 출력의 [오차](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)를 한꺼번에 제곱해 평균 낸다.

$$\text{MSE} = \frac{1}{18 \times 2}\sum_{\text{사람}}\sum_{\text{출력}}(\hat{y} - y)^2$$

[오차](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) 36개(18명 × 2)를 제곱해 평균 낸다.

[학습](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5) 시작 때는 예측이 0 근처라, [오차](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)가 거의 [정답](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5) 그 자체다(수축기 약 126, 이완기 약 79). [MSE](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#mse-mean-squared-error-%ED%8F%89%EA%B7%A0-%EC%A0%9C%EA%B3%B1-%EC%98%A4%EC%B0%A8)는 [오차](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) 제곱의 평균이므로 (126² + 79²) ÷ 2 ≈ 11,000이고, 실제로 [손실](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 약 11000에서 출발했다.

#### Min-Max 정규화의 식

각 값에서 최솟값을 빼고 폭으로 나눈다.

$$x' = \frac{x - \min}{\max - \min}$$

<details><summary>기호 풀이</summary>

| 기호 | 읽는 법 | 뜻 | 예제 값 |
| --- | --- | --- | --- |
| $x'$ | 엑스 프라임 | 정규화한 값 | 나이 50 → 0.5208 |
| $x$ | 엑스 | 원래 값 | 50 |
| $\min$ | 민 | [학습 데이터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0)의 최솟값 | 나이 25 |
| $\max$ | 맥스 | [학습 데이터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0)의 최댓값 | 나이 73 |

</details>

분자를 나눠서 쓰면 $x' = \frac{1}{\max - \min}x - \frac{\min}{\max - \min}$, 즉 **일차함수**다([M01_함수와_일차함수](../02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M01_%ED%95%A8%EC%88%98%EC%99%80_%EC%9D%BC%EC%B0%A8%ED%95%A8%EC%88%98.md)). 그래서 값의 순서와 간격 비율은 그대로 두고 범위만 0~1로 옮긴다.

**[학습](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5)이 빨라지는 이유**: [M04_편미분](../02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M04_%ED%8E%B8%EB%AF%B8%EB%B6%84.md)에서 w 방향 [기울기](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)에는 입력 x가 곱해졌다. 나이처럼 x가 크면(평균 50) 그 [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)의 [기울기](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)만 매우 커진다.

그 결과 [학습률](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)을 크게 하면 값이 튀고(폭발), 작게 하면 다른 [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)가 거의 안 움직인다. 입력을 0~1로 맞추면 모든 [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)의 [기울기](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 크기가 비슷해져서 큰 [학습률](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)을 쓸 수 있다.

| | EX\_03 ([정규화](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#normalization-%EC%A0%95%EA%B7%9C%ED%99%94) 없음) | EX\_04 ([정규화](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#normalization-%EC%A0%95%EA%B7%9C%ED%99%94)) |
|---|---|---|
| [학습률](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) | 0.0005 | 0.1 (200배) |
| 반복 | 200,000번 | 1,000번 |

### 코드

numpy로 Dense(2)가 하는 계산을 흉내 낸다. 숫자 절에서 손으로 계산한 첫 줄이 맞는지 확인할 수 있다.

```python
import numpy as np
W = np.array([[3, 1],
              [2, 4]])      # 입력 2 × 출력 2
b = np.array([1, 1])        # 출력마다 편향
X = np.array([[1, 2],
              [0, 1],
              [2, 0]])      # 3명 × 입력 2
print(X @ W + b)            # 3명 × 출력 2  ← Dense(2)가 하는 일
```

**실행 결과**:

```text
[[ 8 10]
 [ 3  5]
 [ 7  3]]
```

첫 줄 `[8, 10]`이 손으로 계산한 값과 같다.

Keras의 `Dense(2)`는 W는 무작위, b는 0으로 시작해서(Keras 기본값), `fit`([학습](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5)을 실행하는 함수)이 [경사하강법](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-descent-%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95)([D02_경사하강법으로_직선찾기](../01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D02_%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95%EC%9C%BC%EB%A1%9C_%EC%A7%81%EC%84%A0%EC%B0%BE%EA%B8%B0.md))으로 고쳐 나간다.

> [!WARNING]
> **⚠ 예측할 때도 같은 min, max로**
> 새 입력도 **[학습 데이터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0)의** min(25, 22), max(73, 30)으로 바꿔서 넣어야 한다. 새 데이터의 min, max를 따로 구하면 기준이 달라져 엉뚱한 답이 나온다.

## 5. 개념 재확인

<details><summary>1. 입력 4개, 출력 3개인 덴스 층의 W 모양과 파라미터 수는?</summary>

| 단계 | 하는 일 | 코드 | 계산 | 결과 |
|---|---|---|---|---|
| 1 | W 모양 (입력 수 × 출력 수) | `Dense(3)` | 4 × 3 | W는 (4, 3) |
| 2 | [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 개수 | - | 4 × 3 | 12 |
| 3 | [편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5) 개수 (출력마다 하나) | - | 출력 수 3 | 3 |
| 4 | 둘을 더하기 | `model.summary()` | 12 + 3 | [파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수 = 15 |

**① 쓰는 코드**
```python
W = np.zeros((4, 3))        # ② 가중치 행렬: 입력 4 × 출력 3 (본문은 2 × 2)
b = np.zeros(3)             # ④ 편향: 출력마다 하나 → 3개 (본문은 2개)
print(W.shape)              # ② W 모양 → (4, 3)
print(W.size + b.size)      # ③ 가중치 12개 + ④ 편향 3개 → ⑤ 15
```
값은 0으로 채웠다. 모양과 개수만 보는 코드라 값은 상관없다.

**② W 모양 정하기** (`Dense(3)`)
- 행 수 = 입력 수 = 4
- 열 수 = 출력 수 = 3 (W의 열 하나가 출력 하나)
- 모양은 (4, 3)

**③ [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 개수 세기**
- 4 × 3 = 12

**④ [편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5) 개수 세기**
- [편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5)은 출력마다 하나 → 출력 수 3 = 3개

**⑤ 둘을 더하기** (`model.summary()`의 [파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수)
- 12 + 3 = 15
- 식으로는 입력 수 × 출력 수 + 출력 수 = 4 × 3 + 3 = 15

**답**: W는 **(4 × 3)**, [파라미터](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 수 = **15**

</details>

<details><summary>2. BMI 30을 정규화하면? (min 22, max 30)</summary>

| 단계 | 하는 일 | 코드 | 계산 | 결과 |
|---|---|---|---|---|
| 1 | 폭 구하기 (분모) | `max - min` | 30 − 22 | 8 |
| 2 | 최솟값 빼기 (분자) | `x - min` | 30 − 22 | 8 |
| 3 | 나누기 | `(x - min) / (max - min)` | 8 ÷ 8 | x' = 1 |

**① 쓰는 식**

각 값에서 최솟값을 빼고 폭(최댓값 − 최솟값)으로 나눈다. 본문 식은 $x' = \frac{x - \min}{\max - \min}$이다.
넣을 값은 x = 30, min = 22, max = 30이다.

**② 폭 구하기** (`max - min`)
- 30 − 22 = 8

**③ 최솟값 빼기** (`x - min`)
- 30 − 22 = 8

**④ 나누기** (`(x - min) / (max - min)`)
- 8 ÷ 8 = 1 → x' = 1
- 최댓값은 언제나 (max − min) ÷ (max − min) = 1이 됨

**답**: **1** (BMI 30이 최댓값이라 1)

</details>

<details><summary>3. 덴스 층에 활성화 함수가 없으면 출력은 어떤 함수인가?</summary>

| 단계 | 하는 일 | 코드 | 계산 | 결과 |
|---|---|---|---|---|
| 1 | x₁ = 1일 때 (x₂ = 2, W 첫 열 `[3, 2]`, b = 1) | `Dense(...)` | 3 × 1 + 2 × 2 + 1 | 8 |
| 2 | x₁ = 2일 때 | `Dense(...)` | 3 × 2 + 2 × 2 + 1 | 11 |
| 3 | x₁ = 3일 때 | `Dense(...)` | 3 × 3 + 2 × 2 + 1 | 14 |
| 4 | 늘어난 양 | - | 11 − 8, 14 − 11 | 항상 3 (일정) |

**① 쓰는 코드**
```python
W = np.array([[3, 1],
              [2, 4]])      # ① 가중치: 입력 2 × 출력 2
b = np.array([1, 1])        # ① 출력마다 편향
X = np.array([[1, 2],       # ② x₁ = 1인 입력 (x₂ = 2는 그대로)
              [2, 2],       # ③ x₁ = 2인 입력
              [3, 2]])      # ④ x₁ = 3인 입력
print(X @ W + b)            # ⑤ 활성화 함수 없이 곱하고 더하기만 함 → 출력 1 열이 8, 11, 14
```

**② x₁ = 1일 때** (출력 1, W 첫 열 `[3, 2]`)
- 3 × 1 = 3
- 2 × 2 = 4
- 3 + 4 + 1([편향](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5)) = 8

**③ x₁ = 2일 때**
- 3 × 2 = 6
- 2 × 2 = 4
- 6 + 4 + 1 = 11

**④ x₁ = 3일 때**
- 3 × 3 = 9
- 2 × 2 = 4
- 9 + 4 + 1 = 14

**⑤ 늘어난 양 비교**
- 11 − 8 = 3
- 14 − 11 = 3
- 입력을 1 올릴 때마다 늘어나는 양이 3으로 같고, 이 3이 곧 [가중치](../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) w₁₁
- 늘어나는 양이 일정한 것은 일차함수(직선)의 성질

입력이 1 늘 때 출력이 늘어나는 양이 언제나 같은 것은 직선(입력이 2개면 평면)의 성질이다 ([M01_함수와_일차함수](../02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M01_%ED%95%A8%EC%88%98%EC%99%80_%EC%9D%BC%EC%B0%A8%ED%95%A8%EC%88%98.md)).

**답**: 입력에 대한 **일차함수**(직선·평면)다. EX\_04가 선형 회귀인 이유다.

</details>

<details><summary>내 이해</summary>



</details>
