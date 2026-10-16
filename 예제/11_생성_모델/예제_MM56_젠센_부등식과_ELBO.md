---
tags: [MIT딥러닝, 예제]
cssclasses: [dl-note]
---

# 예제: 젠센 부등식과 ELBO

개념 노트: [MM56_젠센_부등식과_ELBO](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/11_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8/MM56_%EC%A0%A0%EC%84%BC_%EB%B6%80%EB%93%B1%EC%8B%9D%EA%B3%BC_ELBO.md) · [목차](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md)

풀이는 접혀 있으니 먼저 풀어 보고 펼쳐 확인한다. 쓰는 식은 둘이다. [젠센 부등식](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#jensen-inequality-%EC%A0%A0%EC%84%BC-%EB%B6%80%EB%93%B1%EC%8B%9D) $\ln\mathbb{E}[Y] \ge \mathbb{E}[\ln Y]$와 $\ln p(x) = \text{ELBO} + \mathrm{KL}(q(z)\,\|\,p(z\mid x))$이다. 모델은 $z \sim \mathcal{N}(0, 1)$, $x\mid z \sim \mathcal{N}(2z, 1)$, $x = 1$이고 참값은 $\ln p(1) = \ln 0.1614 = -1.8237$이다. 인코더가 $\mathcal{N}(\mu, \sigma^2)$를 낼 때 [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)는 (재구성 항) − (KL 항)이고, 재구성 항은 $-\tfrac{1}{2}\ln 2\pi - \tfrac{(1 - 2\mu)^2 + 4\sigma^2}{2}$, KL 항은 $\tfrac{1}{2}(\mu^2 + \sigma^2 - \ln\sigma^2 - 1)$이다.

## 기본 (직접 대입)

### 문제 1

$Y$가 1 또는 9를 같은 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)로 갖는다. $\ln\mathbb{E}[Y]$와 $\mathbb{E}[\ln Y]$를 구해 비교하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $Y = 1$ 또는 $9$ (각각 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $\tfrac{1}{2}$)

**풀이**
1. 식을 그대로 쓴다: $\ln\mathbb{E}[Y] \ \ge\ \mathbb{E}[\ln Y]$
2. 값을 대입한다: $\ln\Big(\dfrac{1 + 9}{2}\Big) \ \ge\ \dfrac{\ln 1 + \ln 9}{2}$
3. 평균을 먼저 낸다: $(1 + 9) / 2 = 5$
4. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 씌운다 ($\ln a$는 $e$를 몇 제곱해야 $a$가 되는지, $e$는 약 2.718): $\ln 5 = 1.6094$
5. 각 값의 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8): $\ln 1 = 0$ ($e^0 = 1$), $\ln 9 = 2.1972$
6. 더하고 2로 나눈다: $0 + 2.1972 = 2.1972$, $2.1972 / 2 = 1.0986$
7. 비교한다: $1.6094 \ge 1.0986$

**답**: $\ln\mathbb{E}[Y] = $ **1.6094**, $\mathbb{E}[\ln Y] = $ **1.0986**. 평균의 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)가 더 크다

</details>

### 문제 2

$Y$가 항상 2이다(같은 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)로 2 또는 2). $\ln\mathbb{E}[Y]$와 $\mathbb{E}[\ln Y]$를 구하라. [젠센 부등식](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#jensen-inequality-%EC%A0%A0%EC%84%BC-%EB%B6%80%EB%93%B1%EC%8B%9D)의 등호는 언제 성립하는가?

<details><summary>풀이와 답</summary>

**주어진 값**
- $Y = 2$ 또는 $2$ (각각 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $\tfrac{1}{2}$, 늘 같은 값)

**풀이**
1. 식을 그대로 쓴다: $\ln\mathbb{E}[Y] \ \ge\ \mathbb{E}[\ln Y]$
2. 값을 대입한다: $\ln\Big(\dfrac{2 + 2}{2}\Big) \ \ge\ \dfrac{\ln 2 + \ln 2}{2}$
3. 왼쪽을 계산한다: $(2 + 2) / 2 = 2$, $\ln 2 = 0.6931$
4. 오른쪽을 계산한다: $(0.6931 + 0.6931) / 2 = 1.3863 / 2 = 0.6931$
5. 비교한다: $0.6931 = 0.6931$

**답**: 둘 다 **0.6931**로 같다. $Y$가 늘 같은 값일 때만 등호가 성립하고, 값이 흩어질수록 간극이 커진다

</details>

### 문제 3

어떤 인코더의 재구성 항이 $-1.2$, KL 항이 $0.5$이다. [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)를 구하라. 간극 $\mathrm{KL}(q\,\|\,p(z\mid x))$가 0.2이면 $\ln p(x)$는 얼마인가?

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mathbb{E}_q[\ln p(x\mid z)] = -1.2$ (재구성 항)
- $\mathrm{KL}(q\,\|\,p(z)) = 0.5$ (KL 항)
- $\mathrm{KL}(q\,\|\,p(z\mid x)) = 0.2$ (간극)

**풀이**
1. 식을 그대로 쓴다: $\text{ELBO} = \mathbb{E}_q[\ln p(x\mid z)] - \mathrm{KL}(q\,\|\,p(z))$
2. 값을 대입한다: $\text{ELBO} = -1.2 - 0.5$
3. 뺀다: $-1.2 - 0.5 = -1.7$
4. 둘째 식을 쓴다: $\ln p(x) = \text{ELBO} + \mathrm{KL}(q\,\|\,p(z\mid x))$
5. 값을 대입한다: $\ln p(x) = -1.7 + 0.2$
6. 더한다: $-1.7 + 0.2 = -1.5$

**답**: [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C) **-1.7**, $\ln p(x) = $ **-1.5**. [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)는 참값보다 간극(0.2)만큼 아래에 있다

</details>

### 문제 4

어떤 인코더의 [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)가 $-2.1$이다. 참값 $\ln p(1) = -1.8237$과의 간극을 구하고, 이 간극이 KL이 될 수 있는 값(0 이상)인지 확인하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\ln p(1) = -1.8237$ (참값)
- $\text{ELBO} = -2.1$

**풀이**
1. 식을 그대로 쓴다: $\text{간극} = \ln p(x) - \text{ELBO}$
2. 값을 대입한다: $\text{간극} = -1.8237 - (-2.1)$
3. 부호를 정리한다: 빼는 수가 음수이므로 더한다. $-1.8237 + 2.1$
4. 계산한다: $2.1 - 1.8237 = 0.2763$
5. 부호를 확인한다: $0.2763 \ge 0$

**답**: 간극 **0.2763**. KL은 0 이상이라 음수가 나왔다면 계산이 틀린 것이다

</details>

## 응용

### 문제 5

인코더가 $\mu = 0.2$, $\sigma^2 = 0.5$를 냈다. 재구성 항, KL 항, [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mu = 0.2$, $\sigma^2 = 0.5$ (인코더 출력)
- $x = 1$, 디코더 평균 $2z$, [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0) 1

**풀이**
1. 식을 그대로 쓴다: $\text{ELBO} = \Big(-\tfrac{1}{2}\ln 2\pi - \tfrac{(1 - 2\mu)^2 + 4\sigma^2}{2}\Big) - \tfrac{1}{2}\big(\mu^2 + \sigma^2 - \ln\sigma^2 - 1\big)$
2. 값을 대입한다: $\text{ELBO} = \Big(-\tfrac{1}{2}\ln 2\pi - \tfrac{(1 - 0.4)^2 + 4 \times 0.5}{2}\Big) - \tfrac{1}{2}\big(0.2^2 + 0.5 - \ln 0.5 - 1\big)$
3. 재구성 항의 제곱: $(1 - 0.4)^2 = 0.6^2 = 0.36$
4. 곱셈: $4 \times 0.5 = 2$
5. 더하고 반으로: $0.36 + 2 = 2.36$, $2.36 / 2 = 1.18$
6. 상수를 빼서 재구성 항: $\tfrac{1}{2}\ln 2\pi = 0.9189$, $-0.9189 - 1.18 = -2.0989$
7. KL 항의 제곱: $0.2^2 = 0.04$
8. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8): $\ln 0.5 = -0.6931$
9. 괄호 안: $0.04 + 0.5 = 0.54$, $0.54 - (-0.6931) = 1.2331$, $1.2331 - 1 = 0.2331$
10. 반으로: $0.2331 / 2 = 0.1166$
11. 뺀다: $-2.0989 - 0.1166 = -2.2155$

**답**: 재구성 항 **-2.0989**, KL 항 **0.1166**, [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C) **-2.2155**

</details>

### 문제 6

문제 5의 인코더의 간극 $\ln p(1) - \text{ELBO}$를 구하라. 이 간극은 무엇을 뜻하는가?

<details><summary>풀이와 답</summary>

**주어진 값**
- $\ln p(1) = -1.8237$ (참값)
- $\text{ELBO} = -2.2155$ (문제 5의 결과)

**풀이**
1. 식을 그대로 쓴다: $\text{간극} = \ln p(x) - \text{ELBO} = \mathrm{KL}(q(z)\,\|\,p(z\mid x))$
2. 값을 대입한다: $\text{간극} = -1.8237 - (-2.2155)$
3. 부호를 정리한다: $-1.8237 + 2.2155$
4. 계산한다: $2.2155 - 1.8237 = 0.3918$ (반올림 전 값으로는 0.3919)

**답**: 간극 **0.3919**(약 0.39). 이 인코더가 진짜 [사후 분포](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#prior-posterior-%EC%82%AC%EC%A0%84-%EB%B6%84%ED%8F%AC%EC%99%80-%EC%82%AC%ED%9B%84-%EB%B6%84%ED%8F%AC) $p(z\mid x) = \mathcal{N}(0.4, 0.2)$에서 떨어진 정도다. 개념 노트의 인코더 $\mathcal{N}(0.5, 0.36)$의 간극(0.1311)보다 더 멀다

</details>

### 문제 7

인코더가 [사전 분포](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#prior-posterior-%EC%82%AC%EC%A0%84-%EB%B6%84%ED%8F%AC%EC%99%80-%EC%82%AC%ED%9B%84-%EB%B6%84%ED%8F%AC)와 같은 $\mathcal{N}(0, 1)$($\mu = 0$, $\sigma^2 = 1$)을 냈다. [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)를 구하고 [사후 분포](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#prior-posterior-%EC%82%AC%EC%A0%84-%EB%B6%84%ED%8F%AC%EC%99%80-%EC%82%AC%ED%9B%84-%EB%B6%84%ED%8F%AC)를 맞힌 인코더의 [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)($-1.8237$)와 비교하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mu = 0$, $\sigma^2 = 1$ (입력을 무시하고 [사전 분포](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#prior-posterior-%EC%82%AC%EC%A0%84-%EB%B6%84%ED%8F%AC%EC%99%80-%EC%82%AC%ED%9B%84-%EB%B6%84%ED%8F%AC)를 그대로 낸 인코더)
- 비교할 값: [사후 분포](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#prior-posterior-%EC%82%AC%EC%A0%84-%EB%B6%84%ED%8F%AC%EC%99%80-%EC%82%AC%ED%9B%84-%EB%B6%84%ED%8F%AC) 인코더의 [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C) $= -1.8237$

**풀이**
1. 식을 그대로 쓴다: $\text{ELBO} = \Big(-\tfrac{1}{2}\ln 2\pi - \tfrac{(1 - 2\mu)^2 + 4\sigma^2}{2}\Big) - \tfrac{1}{2}\big(\mu^2 + \sigma^2 - \ln\sigma^2 - 1\big)$
2. 값을 대입한다: $\text{ELBO} = \Big(-\tfrac{1}{2}\ln 2\pi - \tfrac{(1 - 0)^2 + 4 \times 1}{2}\Big) - \tfrac{1}{2}\big(0 + 1 - \ln 1 - 1\big)$
3. 재구성 항: $(1 - 0)^2 = 1$, $4 \times 1 = 4$, $1 + 4 = 5$, $5 / 2 = 2.5$, $-0.9189 - 2.5 = -3.4189$
4. KL 항: $\ln 1 = 0$ 이므로 $0 + 1 - 0 - 1 = 0$, 반으로 $0$
5. 뺀다: $-3.4189 - 0 = -3.4189$
6. 비교한다: $-3.4189 < -1.8237$

**답**: [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C) **-3.4189**. KL 항은 0으로 가장 작지만 재구성 항이 크게 나빠져서 [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)가 훨씬 낮다. 입력 정보를 담아야 [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)가 오른다

</details>

### 문제 8

[젠센 부등식](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#jensen-inequality-%EC%A0%A0%EC%84%BC-%EB%B6%80%EB%93%B1%EC%8B%9D)의 방향은 함수 모양에 따라 달라진다. $Y$가 1 또는 3을 같은 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)로 가질 때 아래로 볼록한 함수 $\phi(y) = y^2$에 대해 $\phi(\mathbb{E}[Y])$와 $\mathbb{E}[\phi(Y)]$를 구해 비교하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $Y = 1$ 또는 $3$ (각각 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $\tfrac{1}{2}$)
- $\phi(y) = y^2$ (아래로 볼록한 함수)

**풀이**
1. 식을 쓴다: $\phi(\mathbb{E}[Y]) = (\mathbb{E}[Y])^2$, $\mathbb{E}[\phi(Y)] = \mathbb{E}[Y^2]$
2. 값을 대입한다: $\Big(\dfrac{1 + 3}{2}\Big)^2$ 와 $\dfrac{1^2 + 3^2}{2}$
3. 왼쪽: $(1 + 3) / 2 = 2$, $2^2 = 4$
4. 오른쪽: $1^2 = 1$, $3^2 = 9$, $(1 + 9) / 2 = 5$
5. 비교한다: $4 \le 5$

**답**: $\phi(\mathbb{E}[Y]) = $ **4**, $\mathbb{E}[\phi(Y)] = $ **5**. $\ln$(오목)과 반대로 평균의 함수값이 더 작다

</details>

## 강의와 연결

### 문제 9

([Lec15 9](../../Lec15_%EB%B3%80%EB%B6%84_%EC%98%A4%ED%86%A0%EC%9D%B8%EC%BD%94%EB%8D%94.md#9-%EB%AA%A8%EB%91%90-%ED%95%A9%EC%B9%98%EA%B8%B0-elbo)) [중요도 샘플링](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#importance-sampling-%EC%A4%91%EC%9A%94%EB%8F%84-%EC%83%98%ED%94%8C%EB%A7%81)의 무게 $w = \dfrac{p(z)\,p(x\mid z)}{q(z)}$가 표본에 따라 0.1 또는 0.9이고 같은 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)이다. $\ln\mathbb{E}_q[w]$와 $\mathbb{E}_q[\ln w]$를 구하라. 어느 쪽이 [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)의 모양인가?

<details><summary>풀이와 답</summary>

**주어진 값**
- $w = 0.1$ 또는 $0.9$ (각각 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $\tfrac{1}{2}$)

**풀이** ([ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)는 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 [기댓값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#expectation-%EA%B8%B0%EB%8C%93%EA%B0%92) 안으로 넣은 $\mathbb{E}_q[\ln w]$ 쪽이다)
1. 식을 그대로 쓴다: $\ln\mathbb{E}_q[w] \ \ge\ \mathbb{E}_q[\ln w]$
2. 값을 대입한다: $\ln\Big(\dfrac{0.1 + 0.9}{2}\Big) \ \ge\ \dfrac{\ln 0.1 + \ln 0.9}{2}$
3. 왼쪽: $(0.1 + 0.9) / 2 = 0.5$, $\ln 0.5 = -0.6931$
4. 오른쪽의 각 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8): $\ln 0.1 = -2.3026$, $\ln 0.9 = -0.1054$
5. 더하고 2로 나눈다: $-2.3026 + (-0.1054) = -2.4080$, $-2.4080 / 2 = -1.2040$
6. 비교한다: $-0.6931 \ge -1.2040$

**답**: $\ln\mathbb{E}_q[w] = $ **-0.6931**, $\mathbb{E}_q[\ln w] = $ **-1.2040**. $\ln p(x)$는 왼쪽, [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)는 오른쪽 모양이고, 둘의 차이 0.5109가 하한의 느슨함이다

</details>

### 문제 10

([Lec15 8](../../Lec15_%EB%B3%80%EB%B6%84_%EC%98%A4%ED%86%A0%EC%9D%B8%EC%BD%94%EB%8D%94.md#8-%EC%9A%94%EB%A0%B9-3-%EC%9D%B8%EC%BD%94%EB%8D%94%EB%A1%9C-%EC%B5%9C%EC%A0%81%EC%9D%98-%ED%91%9C%EB%B3%B8-%EB%B6%84%ED%8F%AC-%EC%98%88%EC%B8%A1%ED%95%98%EA%B8%B0)) 인코더가 $q = \mathcal{N}(0.4, 0.36)$을 냈다(평균은 [사후 분포](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#prior-posterior-%EC%82%AC%EC%A0%84-%EB%B6%84%ED%8F%AC%EC%99%80-%EC%82%AC%ED%9B%84-%EB%B6%84%ED%8F%AC)와 같고 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0)만 0.36). [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)를 구하고, 간극이 $\mathrm{KL}\big(\mathcal{N}(0.4, 0.36)\,\|\,\mathcal{N}(0.4, 0.2)\big)$와 같은지 확인하라. 두 가우스의 KL은 $\tfrac{1}{2}\ln\frac{v_2}{v_1} + \frac{v_1 + (m_1 - m_2)^2}{2 v_2} - \tfrac{1}{2}$이다.

<details><summary>풀이와 답</summary>

**주어진 값**
- $q = \mathcal{N}(0.4, 0.36)$, 즉 $\mu = 0.4$, $\sigma^2 = 0.36$
- [사후 분포](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#prior-posterior-%EC%82%AC%EC%A0%84-%EB%B6%84%ED%8F%AC%EC%99%80-%EC%82%AC%ED%9B%84-%EB%B6%84%ED%8F%AC) $p(z\mid x) = \mathcal{N}(0.4, 0.2)$
- $\ln p(1) = -1.8237$

**풀이**
1. 재구성 항의 식을 쓰고 대입한다: $-\tfrac{1}{2}\ln 2\pi - \tfrac{(1 - 2 \times 0.4)^2 + 4 \times 0.36}{2}$
2. 제곱과 곱셈: $(1 - 0.8)^2 = 0.04$, $4 \times 0.36 = 1.44$
3. 더하고 반으로: $0.04 + 1.44 = 1.48$, $1.48 / 2 = 0.74$
4. 상수를 빼서 재구성 항: $-0.9189 - 0.74 = -1.6589$
5. KL 항의 식을 쓰고 대입한다: $\tfrac{1}{2}\big(0.4^2 + 0.36 - \ln 0.36 - 1\big)$
6. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8): $\ln 0.36 = -1.0217$
7. 괄호 안: $0.16 + 0.36 = 0.52$, $0.52 - (-1.0217) = 1.5417$, $1.5417 - 1 = 0.5417$
8. 반으로: $0.5417 / 2 = 0.2708$ (반올림 전 값)
9. [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C)를 구한다: $-1.6589 - 0.2708 = -1.9297$ (반올림 전 값으로는 $-1.9298$)
10. 간극을 구한다: $-1.8237 - (-1.9298) = 0.1061$
11. 가우스 KL 식에 대입한다: $\tfrac{1}{2}\ln\frac{0.2}{0.36} + \frac{0.36 + (0.4 - 0.4)^2}{2 \times 0.2} - \tfrac{1}{2}$
12. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8): $0.2 / 0.36 = 0.5556$, $\ln 0.5556 = -0.5878$, 반으로 $-0.2939$
13. 분수: $(0.4 - 0.4)^2 = 0$, $0.36 + 0 = 0.36$, $2 \times 0.2 = 0.4$, $0.36 / 0.4 = 0.9$
14. 합친다: $-0.2939 + 0.9 - 0.5 = 0.1061$

**답**: [ELBO](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#elbo-evidence-lower-bound-%EC%A6%9D%EA%B1%B0-%ED%95%98%ED%95%9C) **-1.9298**(약), 간극 **0.1061**이고 KL($q\,\|\,$[사후 분포](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#prior-posterior-%EC%82%AC%EC%A0%84-%EB%B6%84%ED%8F%AC%EC%99%80-%EC%82%AC%ED%9B%84-%EB%B6%84%ED%8F%AC)) **0.1061**과 같다. 평균을 맞혀도 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0)이 달라서 간극이 남는다

</details>
