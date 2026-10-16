---
tags: [MIT딥러닝, 예제]
cssclasses: [dl-note]
---

# 예제: 가우스 KL과 재매개변수화

개념 노트: [MM57_가우스_KL과_재매개변수화](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/11_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8/MM57_%EA%B0%80%EC%9A%B0%EC%8A%A4_KL%EA%B3%BC_%EC%9E%AC%EB%A7%A4%EA%B0%9C%EB%B3%80%EC%88%98%ED%99%94.md) · [목차](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md)

풀이는 접혀 있으니 먼저 풀어 보고 펼쳐 확인한다. 쓰는 식은 셋이다. 단위 가우스와의 KL $\tfrac{1}{2}(\mu^2 + \sigma^2 - \ln\sigma^2 - 1)$ (여러 차원이면 차원마다 계산해 더한다), 그 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $\partial\mathrm{KL}/\partial\mu = \mu$와 $\partial\mathrm{KL}/\partial\sigma = \sigma - 1/\sigma$, 재매개변수화 $z = \mu + \sigma\epsilon$이다. $\ln a$는 $e$(약 2.718)를 몇 제곱해야 $a$가 되는지를 뜻한다.

## 기본 (직접 대입)

### 문제 1

인코더가 $\mu = 2$, $\sigma = 1$을 냈다. 단위 가우스와의 KL을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mu = 2$ (인코더 평균)
- $\sigma = 1$ (인코더 표준편차)

**풀이**
1. 식을 그대로 쓴다: $\mathrm{KL} = \tfrac{1}{2}\big(\mu^2 + \sigma^2 - \ln\sigma^2 - 1\big)$
2. 값을 대입한다: $\mathrm{KL} = \tfrac{1}{2}\big(2^2 + 1^2 - \ln 1^2 - 1\big)$
3. 제곱한다: $2^2 = 4$, $1^2 = 1$
4. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 계산한다: $\ln 1 = 0$ ($e^0 = 1$ 이므로)
5. 괄호 안을 계산한다: $4 + 1 = 5$, $5 - 0 = 5$, $5 - 1 = 4$
6. 반으로: $4 / 2 = 2$

**답**: KL = **2**. 표준편차가 이미 1이라 평균이 원점에서 떨어진 몫($\tfrac{1}{2}\mu^2 = 2$)만 남는다

</details>

### 문제 2

인코더가 $\mu = 0$, $\sigma = 2$를 냈다. 단위 가우스와의 KL을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mu = 0$, $\sigma = 2$ (인코더 출력)

**풀이**
1. 식을 그대로 쓴다: $\mathrm{KL} = \tfrac{1}{2}\big(\mu^2 + \sigma^2 - \ln\sigma^2 - 1\big)$
2. 값을 대입한다: $\mathrm{KL} = \tfrac{1}{2}\big(0^2 + 2^2 - \ln 2^2 - 1\big)$
3. 제곱한다: $0^2 = 0$, $2^2 = 4$
4. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 계산한다: $\ln 4 = 1.3863$
5. 괄호 안을 계산한다: $0 + 4 = 4$, $4 - 1.3863 = 2.6137$, $2.6137 - 1 = 1.6137$
6. 반으로: $1.6137 / 2 = 0.8069$

**답**: KL = **0.8069**. 평균은 원점인데도 퍼짐이 1보다 커서 벌을 받는다

</details>

### 문제 3

2차원 [잠재 공간](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#latent-space-%EC%9E%A0%EC%9E%AC-%EA%B3%B5%EA%B0%84)에서 인코더가 평균 $(1, 0)$, 표준편차 $(1, 0.5)$를 냈다. 단위 가우스와의 KL을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mu_1 = 1$, $\mu_2 = 0$ (평균)
- $\sigma_1 = 1$, $\sigma_2 = 0.5$ (표준편차)
- $d = 2$

**풀이**
1. 식을 그대로 쓴다: $\mathrm{KL} = \dfrac{1}{2}\sum_{k=1}^{d}\big(\mu_k^2 + \sigma_k^2 - \ln\sigma_k^2 - 1\big)$
2. 값을 대입한다: $\mathrm{KL} = \dfrac{1}{2}\sum_{k=1}^{2}\big(\mu_k^2 + \sigma_k^2 - \ln\sigma_k^2 - 1\big)$
3. Σ를 풀어 쓴다: $\dfrac{1}{2}\big(1^2 + 1^2 - \ln 1^2 - 1\big) + \dfrac{1}{2}\big(0^2 + 0.5^2 - \ln 0.5^2 - 1\big)$
4. 첫 차원: $1 + 1 = 2$, $\ln 1 = 0$ 이므로 $2 - 0 - 1 = 1$, 반으로 $0.5$
5. 둘째 차원의 제곱: $0^2 = 0$, $0.5^2 = 0.25$
6. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 계산한다: $\ln 0.25 = -1.3863$
7. 둘째 차원 괄호 안: $0 + 0.25 = 0.25$, $0.25 - (-1.3863) = 1.6363$, $1.6363 - 1 = 0.6363$
8. 반으로: $0.6363 / 2 = 0.3181$ (반올림 전 값)
9. 두 차원을 더한다: $0.5 + 0.3181 = 0.8181$

**답**: KL = **0.8181**. 첫 차원은 평균 몫, 둘째 차원은 퍼짐이 작은 몫만 벌을 받는다

</details>

### 문제 4

인코더가 $\mu = 0.5$, $\sigma = 2$를 냈고 단위 가우스에서 $\epsilon = 1.5$를 뽑았다. 재매개변수화로 표본 $z$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mu = 0.5$ (평균), $\sigma = 2$ (표준편차)
- $\epsilon = 1.5$ (단위 가우스에서 뽑은 잡음)

**풀이**
1. 식을 그대로 쓴다: $z = \mu + \sigma\epsilon$
2. 값을 대입한다: $z = 0.5 + 2 \times 1.5$
3. 곱셈을 먼저 한다: $2 \times 1.5 = 3$ 이므로 $z = 0.5 + 3$
4. 더한다: $0.5 + 3 = 3.5$

**답**: $z = $ **3.5**. 무작위성은 $\epsilon$에만 있고 $z$는 $\mu, \sigma$의 평범한 함수다

</details>

## 응용

### 문제 5

인코더가 $\mu = 2$, $\sigma = 0.5$를 냈다. KL의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $\partial\mathrm{KL}/\partial\mu$, $\partial\mathrm{KL}/\partial\sigma$를 구하고, [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $\eta = 0.1$로 경사하강 한 번 한 뒤의 $(\mu, \sigma)$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mu = 2$, $\sigma = 0.5$ (지금 값)
- $\eta = 0.1$ ([학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))

**풀이**
1. [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 식을 쓴다: $\dfrac{\partial\mathrm{KL}}{\partial\mu} = \mu$, $\dfrac{\partial\mathrm{KL}}{\partial\sigma} = \sigma - \dfrac{1}{\sigma}$
2. 값을 대입한다: $\dfrac{\partial\mathrm{KL}}{\partial\mu} = 2$, $\dfrac{\partial\mathrm{KL}}{\partial\sigma} = 0.5 - \dfrac{1}{0.5}$
3. 나눗셈을 한다: $1 / 0.5 = 2$ 이므로 $0.5 - 2 = -1.5$
4. 갱신 식을 쓴다: $\theta \leftarrow \theta - \eta\,\nabla\mathrm{KL}$
5. 값을 대입한다: $\mu \leftarrow 2 - 0.1 \times 2$, $\sigma \leftarrow 0.5 - 0.1 \times (-1.5)$
6. [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)을 곱한다: $0.1 \times 2 = 0.2$, $0.1 \times (-1.5) = -0.15$
7. 뺀다: $2 - 0.2 = 1.8$, $0.5 - (-0.15) = 0.65$

**답**: [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $(2, -1.5)$, 갱신 뒤 $(\mu, \sigma) = $ **(1.8, 0.65)**. $\mu$는 0으로, $\sigma$는 1로 다가간다

</details>

### 문제 6

$x = 1$, 인코더 $\mu = 0.3$, $\sigma = 0.4$, 뽑은 잡음 $\epsilon = -0.5$, [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98) $L = (x - 2z)^2/2$이다. 재매개변수화와 [연쇄법칙](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#chain-rule-%EC%97%B0%EC%87%84%EB%B2%95%EC%B9%99)으로 $\partial L/\partial\mu$와 $\partial L/\partial\sigma$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $x = 1$ (입력), $\mu = 0.3$, $\sigma = 0.4$ (인코더 출력)
- $\epsilon = -0.5$ (잡음)
- $L = (x - 2z)^2/2$ (재구성 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98))

**풀이**
1. 식을 그대로 쓴다: $z = \mu + \sigma\epsilon$, $\dfrac{\partial L}{\partial\mu} = \dfrac{\partial L}{\partial z}\cdot 1$, $\dfrac{\partial L}{\partial\sigma} = \dfrac{\partial L}{\partial z}\cdot\epsilon$
2. 값을 대입한다: $z = 0.3 + 0.4 \times (-0.5)$
3. 곱셈을 먼저 한다: $0.4 \times (-0.5) = -0.2$ 이므로 $z = 0.3 + (-0.2) = 0.1$
4. $L$을 $z$로 미분하면 $(x - 2z) \times (-2)$이다. 대입한다: $(1 - 2 \times 0.1) \times (-2)$
5. 곱셈과 뺄셈: $2 \times 0.1 = 0.2$, $1 - 0.2 = 0.8$
6. $-2$를 곱한다: $0.8 \times (-2) = -1.6$
7. $\mu$ [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0): $-1.6 \times 1 = -1.6$
8. $\sigma$ [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0): $-1.6 \times (-0.5) = 0.8$

**답**: $\partial L/\partial\mu = $ **-1.6**, $\partial L/\partial\sigma = $ **0.8**. $\mu$를 키우고 $\sigma$를 줄이면 $z$가 $x = 1$을 되살리는 값 0.5 쪽으로 간다

</details>

### 문제 7

$\mu = 0$으로 같을 때 $\sigma = 0.5$와 $\sigma = 1.5$의 KL을 각각 구해 비교하라. (둘 다 $\sigma = 1$에서 0.5만큼 떨어져 있다.)

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mu = 0$ (평균)
- $\sigma = 0.5$ 와 $\sigma = 1.5$ (두 후보)

**풀이**
1. 식을 그대로 쓴다: $\mathrm{KL} = \tfrac{1}{2}\big(\mu^2 + \sigma^2 - \ln\sigma^2 - 1\big)$
2. 값을 대입한다 ($\sigma = 0.5$): $\mathrm{KL} = \tfrac{1}{2}\big(0 + 0.5^2 - \ln 0.5^2 - 1\big)$
3. 제곱과 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8): $0.5^2 = 0.25$, $\ln 0.25 = -1.3863$
4. 괄호 안: $0 + 0.25 = 0.25$, $0.25 - (-1.3863) = 1.6363$, $1.6363 - 1 = 0.6363$
5. 반으로: $0.6363 / 2 = 0.3181$ (반올림 전 값)
6. 값을 대입한다 ($\sigma = 1.5$): $\mathrm{KL} = \tfrac{1}{2}\big(0 + 1.5^2 - \ln 1.5^2 - 1\big)$
7. 제곱과 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8): $1.5^2 = 2.25$, $\ln 2.25 = 0.8109$
8. 괄호 안: $0 + 2.25 = 2.25$, $2.25 - 0.8109 = 1.4391$, $1.4391 - 1 = 0.4391$
9. 반으로: $0.4391 / 2 = 0.2195$ (반올림 전 값)
10. 비교한다: $0.3181 > 0.2195$

**답**: $\sigma = 0.5$는 **0.3181**, $\sigma = 1.5$는 **0.2195**. 표준편차가 1보다 작을 때의 벌이 더 크다. 구름이 점으로 쪼그라드는 것을 더 세게 막는다

</details>

### 문제 8

신경망이 $\sigma$ 대신 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0) $\ln\sigma^2 = -1$을 냈다. 평균은 $\mu = 0$이다. 이때 $\sigma^2$, $\sigma$와 KL을 구하라. ($e^{-1} = 0.3679$)

<details><summary>풀이와 답</summary>

**주어진 값**
- $\ln\sigma^2 = -1$ (신경망 출력, [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0))
- $\mu = 0$
- $e^{-1} = 0.3679$ ($e = 2.718$을 $-1$제곱, 곧 $1/e$)

**풀이** ([로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0)은 어떤 실수든 될 수 있어서 $\sigma > 0$를 늘 지킨다)
1. [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0)을 구한다: $\sigma^2 = e^{\ln\sigma^2} = e^{-1} = 0.3679$
2. 표준편차를 구한다: $\sigma = \sqrt{0.3679} = 0.6065$
3. KL 식을 쓴다: $\mathrm{KL} = \tfrac{1}{2}\big(\mu^2 + \sigma^2 - \ln\sigma^2 - 1\big)$
4. 값을 대입한다: $\mathrm{KL} = \tfrac{1}{2}\big(0^2 + 0.3679 - (-1) - 1\big)$
5. 괄호 안: $0 + 0.3679 = 0.3679$, $0.3679 - (-1) = 1.3679$, $1.3679 - 1 = 0.3679$
6. 반으로: $0.3679 / 2 = 0.1839$ (반올림 전 값)

**답**: $\sigma^2 = $ **0.3679**, $\sigma = $ **0.6065**, KL = **0.1839**

</details>

## 강의와 연결

### 문제 9

([Lec15 10.1](../../Lec15_%EB%B3%80%EB%B6%84_%EC%98%A4%ED%86%A0%EC%9D%B8%EC%BD%94%EB%8D%94.md#101-%EB%91%90-%ED%95%AD%EC%9D%98-%EB%9C%BB)) 3차원 [잠재 공간](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#latent-space-%EC%9E%A0%EC%9E%AC-%EA%B3%B5%EA%B0%84)에서 인코더가 평균 $(0.5, -0.5, 0)$, 표준편차 $(1, 1, 1)$을 냈다. 재구성 항이 $-1.2$일 때 [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\boldsymbol{\mu} = (0.5, -0.5, 0)$, $\boldsymbol{\sigma} = (1, 1, 1)$ (인코더 출력)
- 재구성 항 $= -1.2$

**풀이** ([ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C) = 재구성 항 − KL 항)
1. KL 식을 쓴다: $\mathrm{KL} = \dfrac{1}{2}\sum_{k=1}^{3}\big(\mu_k^2 + \sigma_k^2 - \ln\sigma_k^2 - 1\big)$
2. 표준편차가 모두 1이므로 $\sigma_k^2 - \ln\sigma_k^2 - 1 = 1 - 0 - 1 = 0$ ($\ln 1 = 0$). 식이 $\dfrac{1}{2}\sum_{k}\mu_k^2$로 줄어든다
3. Σ를 풀어 쓴다: $\dfrac{1}{2}\big(0.5^2 + (-0.5)^2 + 0^2\big)$
4. 제곱한다: $0.25,\ 0.25,\ 0$
5. 더한다: $0.25 + 0.25 + 0 = 0.5$
6. 반으로: $0.5 / 2 = 0.25$
7. [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C) 식을 쓰고 대입한다: $\text{ELBO} = -1.2 - 0.25$
8. 뺀다: $-1.2 - 0.25 = -1.45$

**답**: KL 항 **0.25**, [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C) **-1.45**. 평균이 원점에서 멀수록 KL 항이 [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)를 깎는다

</details>

### 문제 10

([Lec15 10.2](../../Lec15_%EB%B3%80%EB%B6%84_%EC%98%A4%ED%86%A0%EC%9D%B8%EC%BD%94%EB%8D%94.md#102-%EC%A7%93%EB%88%84%EB%A5%B4%EA%B8%B0%EC%99%80-%EB%B3%B5%EC%9B%90%EC%9D%98-%EC%A4%84%EB%8B%A4%EB%A6%AC%EA%B8%B0)) 인코더 A가 모든 입력에 같은 $\mu = 0.1$, $\sigma = 0.9$를 내고, 인코더 B는 입력을 잘 담아 $\mu = 2$, $\sigma = 0.5$를 낸다. 각 KL을 구하라. 어느 쪽이 사후 붕괴에 가까운가?

<details><summary>풀이와 답</summary>

**주어진 값**
- 인코더 A: $\mu = 0.1$, $\sigma = 0.9$ (입력과 상관없이 늘 같은 값)
- 인코더 B: $\mu = 2$, $\sigma = 0.5$

**풀이**
1. 식을 그대로 쓴다: $\mathrm{KL} = \tfrac{1}{2}\big(\mu^2 + \sigma^2 - \ln\sigma^2 - 1\big)$
2. 값을 대입한다 (A): $\mathrm{KL}_A = \tfrac{1}{2}\big(0.1^2 + 0.9^2 - \ln 0.9^2 - 1\big)$
3. 제곱과 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8): $0.1^2 = 0.01$, $0.9^2 = 0.81$, $\ln 0.81 = -0.2107$
4. 괄호 안: $0.01 + 0.81 = 0.82$, $0.82 - (-0.2107) = 1.0307$, $1.0307 - 1 = 0.0307$
5. 반으로: $0.0307 / 2 = 0.0154$ (반올림 전 값)
6. 값을 대입한다 (B): $\mathrm{KL}_B = \tfrac{1}{2}\big(2^2 + 0.5^2 - \ln 0.5^2 - 1\big)$
7. 제곱과 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8): $2^2 = 4$, $0.5^2 = 0.25$, $\ln 0.25 = -1.3863$
8. 괄호 안: $4 + 0.25 = 4.25$, $4.25 - (-1.3863) = 5.6363$, $5.6363 - 1 = 4.6363$
9. 반으로: $4.6363 / 2 = 2.3181$ (반올림 전 값)

**답**: A는 **0.0154**, B는 **2.3181**. A는 KL이 거의 0이라 단위 가우스와 같아졌다. $z$가 입력 정보를 담지 않는 사후 붕괴에 가깝다. B는 입력 정보를 담는 대가로 KL 벌을 크게 낸다

</details>
