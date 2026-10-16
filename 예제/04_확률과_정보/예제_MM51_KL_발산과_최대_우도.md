---
tags: [MIT딥러닝, 예제]
cssclasses: [dl-note]
---

# 예제: KL 발산과 최대 우도

개념 노트: [MM51_KL_발산과_최대_우도](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/04_%ED%99%95%EB%A5%A0%EA%B3%BC_%EC%A0%95%EB%B3%B4/MM51_KL_%EB%B0%9C%EC%82%B0%EA%B3%BC_%EC%B5%9C%EB%8C%80_%EC%9A%B0%EB%8F%84.md) · [목차](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md)

풀이는 접혀 있으니 먼저 풀어 보고 펼쳐 확인한다. $\log$는 자연로그 $\ln$이다($\ln a$는 "$e$(약 2.718)를 몇 제곱하면 $a$가 되는가"). 표준편차 1인 [정규분포](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#normal-distribution-%EC%A0%95%EA%B7%9C%EB%B6%84%ED%8F%AC)의 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) 밀도는 $\log \mathcal{N}(x; \mu, 1) = -0.9189 - \tfrac{1}{2}(x - \mu)^2$이다(0.9189 $= \tfrac{1}{2}\ln 2\pi$).

## 기본 (직접 대입)

### 문제 1

진짜 분포 $p = (0.5, 0.5)$, 모델 $q = (0.25, 0.75)$일 때 $D_{KL}(p \,\|\, q)$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $p = (0.5, 0.5)$ (진짜 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))
- $q = (0.25, 0.75)$ (모델이 믿는 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))

**풀이**
1. 식을 그대로 쓴다: $D_{KL}(p \,\|\, q) = \sum_{x} p(x) \log \dfrac{p(x)}{q(x)}$
2. 값을 대입한다 (Σ를 두 항으로 풀어 썼다): $D_{KL} = 0.5 \log \dfrac{0.5}{0.25} + 0.5 \log \dfrac{0.5}{0.75}$
3. 첫 항의 나눗셈을 한다: $0.5 / 0.25 = 2$
4. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 구한다: $\ln 2 = 0.6931$
5. 둘째 항의 나눗셈을 한다: $0.5 / 0.75 = 0.6667$
6. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 구한다. 1보다 작은 수의 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)는 음수다: $\ln 0.6667 = -0.4055$
7. 각 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)에 $p$를 곱한다: $0.5 \times 0.6931 = 0.3466$, $0.5 \times (-0.4055) = -0.2027$
8. 더한다: $0.3466 + (-0.2027) = 0.1438$ (반올림 전 값으로 계산)

**답**: $D_{KL}(p \,\|\, q) = 0.1438$

</details>

### 문제 2

$p = q = (0.3, 0.7)$일 때 $D_{KL}(p \,\|\, q)$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $p = (0.3, 0.7)$ (진짜 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))
- $q = (0.3, 0.7)$ (모델 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0), $p$와 같다)

**풀이**
1. 식을 그대로 쓴다: $D_{KL}(p \,\|\, q) = \sum_{x} p(x) \log \dfrac{p(x)}{q(x)}$
2. 값을 대입한다: $D_{KL} = 0.3 \log \dfrac{0.3}{0.3} + 0.7 \log \dfrac{0.7}{0.7}$
3. 나눗셈을 한다: $0.3 / 0.3 = 1$, $0.7 / 0.7 = 1$
4. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 구한다. $e$의 0제곱이 1이므로: $\ln 1 = 0$
5. 곱한다: $0.3 \times 0 = 0$, $0.7 \times 0 = 0$
6. 더한다: $0 + 0 = 0$

**답**: $D_{KL} = 0$ ($p = q$일 때만 0이다)

</details>

### 문제 3

문제 1의 두 분포에서 순서를 바꾼 $D_{KL}(q \,\|\, p)$를 구하고 문제 1의 값과 비교하라. $p = (0.5, 0.5)$, $q = (0.25, 0.75)$.

<details><summary>풀이와 답</summary>

**주어진 값**
- $q = (0.25, 0.75)$ (이번에는 기준 분포)
- $p = (0.5, 0.5)$ (비교할 분포)

**풀이**
1. 식을 그대로 쓴다: $D_{KL}(q \,\|\, p) = \sum_{x} q(x) \log \dfrac{q(x)}{p(x)}$
2. 값을 대입한다: $D_{KL} = 0.25 \log \dfrac{0.25}{0.5} + 0.75 \log \dfrac{0.75}{0.5}$
3. 첫 항의 나눗셈을 한다: $0.25 / 0.5 = 0.5$
4. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 구한다: $\ln 0.5 = -0.6931$
5. 둘째 항의 나눗셈을 한다: $0.75 / 0.5 = 1.5$
6. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 구한다: $\ln 1.5 = 0.4055$
7. 각 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)에 곱한다: $0.25 \times (-0.6931) = -0.1733$, $0.75 \times 0.4055 = 0.3041$
8. 더한다: $-0.1733 + 0.3041 = 0.1308$

**답**: $D_{KL}(q \,\|\, p) = 0.1308$. 문제 1의 $0.1438$과 다르다. [KL 발산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#kl-divergence-kl-%EB%B0%9C%EC%82%B0)은 순서를 바꾸면 값이 달라진다(대칭이 아니다).

</details>

### 문제 4

모델 $\mathcal{N}(\mu, 1)$에서 $\mu = 1$이고 데이터가 $x = 3$이다. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) 밀도 $\log \mathcal{N}(3; 1, 1)$을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $x = 3$ (데이터)
- $\mu = 1$ (모델 평균), $\sigma = 1$
- $0.9189 = \tfrac{1}{2}\ln 2\pi$ (상수)

**풀이**
1. 식을 그대로 쓴다: $\log \mathcal{N}(x; \mu, 1) = -0.9189 - \tfrac{1}{2}(x - \mu)^2$
2. 값을 대입한다: $\log \mathcal{N}(3; 1, 1) = -0.9189 - \tfrac{1}{2}(3 - 1)^2$
3. 괄호 안을 뺀다: $3 - 1 = 2$
4. 제곱한다: $2^2 = 4$
5. 절반을 취한다: $0.5 \times 4 = 2$
6. 뺀다: $-0.9189 - 2 = -2.9189$

**답**: $-2.9189$ (데이터가 평균에서 멀수록 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) 밀도가 작아진다)

</details>

## 응용

### 문제 5

[학습 데이터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0)가 $0, 2, 4$이다. 모델 $\mathcal{N}(\mu, 1)$에서 $\mu = 2$일 때 평균 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) [우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#likelihood-%EC%9A%B0%EB%8F%84)를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $x^{(1)} = 0$, $x^{(2)} = 2$, $x^{(3)} = 4$ ([학습 데이터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0))
- $N = 3$ (데이터 수)
- $\mu = 2$, $\sigma = 1$

**풀이**
1. 식을 그대로 쓴다: $\dfrac{1}{N}\sum_{i=1}^{N} \log p_\theta(x^{(i)})$
2. 값을 대입한다 (Σ를 세 항으로 풀어 썼다): $\dfrac{1}{3}\big(\log \mathcal{N}(0; 2, 1) + \log \mathcal{N}(2; 2, 1) + \log \mathcal{N}(4; 2, 1)\big)$
3. 첫 항: $(0 - 2)^2 = 4$ 이므로 $-0.9189 - 0.5 \times 4 = -2.9189$
4. 둘째 항: $(2 - 2)^2 = 0$ 이므로 $-0.9189 - 0 = -0.9189$
5. 셋째 항: $(4 - 2)^2 = 4$ 이므로 $-0.9189 - 0.5 \times 4 = -2.9189$
6. 세 항을 더한다: $-2.9189 + (-0.9189) + (-2.9189) = -6.7568$
7. 3으로 나눈다: $-6.7568 / 3 = -2.2523$

**답**: 평균 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) [우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#likelihood-%EC%9A%B0%EB%8F%84) $-2.2523$

</details>

### 문제 6

문제 5와 같은 [학습 데이터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0) $0, 2, 4$에서 $\mu = 3$일 때 평균 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) [우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#likelihood-%EC%9A%B0%EB%8F%84)를 구하고, $\mu = 2$와 $\mu = 3$ 중 어느 쪽이 [최대 우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#maximum-likelihood-%EC%B5%9C%EB%8C%80-%EC%9A%B0%EB%8F%84)에 가까운지 말하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $x^{(1)} = 0$, $x^{(2)} = 2$, $x^{(3)} = 4$ ([학습 데이터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0))
- $N = 3$
- $\mu = 3$ (새로 시험할 평균), 비교 대상은 문제 5의 $-2.2523$

**풀이**
1. 식을 그대로 쓴다: $\dfrac{1}{N}\sum_{i=1}^{N} \log p_\theta(x^{(i)})$
2. 값을 대입한다: $\dfrac{1}{3}\big(\log \mathcal{N}(0; 3, 1) + \log \mathcal{N}(2; 3, 1) + \log \mathcal{N}(4; 3, 1)\big)$
3. 첫 항: $(0 - 3)^2 = 9$ 이므로 $-0.9189 - 0.5 \times 9 = -5.4189$
4. 둘째 항: $(2 - 3)^2 = 1$ 이므로 $-0.9189 - 0.5 = -1.4189$
5. 셋째 항: $(4 - 3)^2 = 1$ 이므로 $-0.9189 - 0.5 = -1.4189$
6. 세 항을 더한다: $-5.4189 + (-1.4189) + (-1.4189) = -8.2567$
7. 3으로 나눈다: $-8.2567 / 3 = -2.7522$ (반올림 전 값으로 계산하면 $-2.7523$)
8. 비교한다: $-2.2523 > -2.7523$

**답**: $\mu = 3$은 $-2.7523$. $\mu = 2$($-2.2523$)가 더 크므로 [최대 우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#maximum-likelihood-%EC%B5%9C%EB%8C%80-%EC%9A%B0%EB%8F%84)에 더 가깝다. 데이터의 평균 2에서 가장 크다.

</details>

### 문제 7

[학습 데이터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0)가 $1, 2, 6$이다. $\mu = 2, 3, 4$ 중 평균 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) [우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#likelihood-%EC%9A%B0%EB%8F%84)가 가장 큰 것을 찾아라. (지름길: 평균 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) [우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#likelihood-%EC%9A%B0%EB%8F%84) $= -0.9189 - \tfrac{1}{2} \times$ (거리 제곱의 평균))

<details><summary>풀이와 답</summary>

**주어진 값**
- $x^{(1)} = 1$, $x^{(2)} = 2$, $x^{(3)} = 6$ ([학습 데이터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0))
- $N = 3$
- 후보 $\mu = 2, 3, 4$

**풀이**
1. 식을 그대로 쓴다: 평균 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) [우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#likelihood-%EC%9A%B0%EB%8F%84) $= -0.9189 - \tfrac{1}{2}\cdot\dfrac{1}{N}\sum_{i=1}^{N}(x^{(i)} - \mu)^2$
2. 값을 대입한다 ($\mu = 2$): $-0.9189 - \tfrac{1}{2}\cdot\dfrac{1}{3}\big((1-2)^2 + (2-2)^2 + (6-2)^2\big)$
3. 거리 제곱을 더한다: $1 + 0 + 16 = 17$
4. 평균을 낸다: $17 / 3 = 5.6667$, 절반은 $2.8333$ 이므로 $-0.9189 - 2.8333 = -3.7523$ (반올림 전 값으로 계산)
5. 값을 대입한다 ($\mu = 3$): $-0.9189 - \tfrac{1}{2}\cdot\dfrac{1}{3}\big((1-3)^2 + (2-3)^2 + (6-3)^2\big)$
6. 거리 제곱을 더한다: $4 + 1 + 9 = 14$
7. 평균을 낸다: $14 / 3 = 4.6667$, 절반은 $2.3333$ 이므로 $-0.9189 - 2.3333 = -3.2523$
8. 값을 대입한다 ($\mu = 4$): $-0.9189 - \tfrac{1}{2}\cdot\dfrac{1}{3}\big((1-4)^2 + (2-4)^2 + (6-4)^2\big)$
9. 거리 제곱을 더한다: $9 + 4 + 4 = 17$
10. 평균을 내고 뺀다: $17 / 3 = 5.6667$, 절반 $2.8333$ 이므로 $-0.9189 - 2.8333 = -3.7523$
11. 비교한다: $-3.2523$이 가장 크다

**답**: $\mu = 3$일 때 $-3.2523$으로 가장 크다. 데이터의 평균 $(1 + 2 + 6)/3 = 3$이다. [정규분포](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#normal-distribution-%EC%A0%95%EA%B7%9C%EB%B6%84%ED%8F%AC)에서 [최대 우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#maximum-likelihood-%EC%B5%9C%EB%8C%80-%EC%9A%B0%EB%8F%84) 해는 데이터의 평균이다.

</details>

### 문제 8

[학습 데이터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0)가 $0$ 하나뿐이다. 모델이 이 점에 좁은 [정규분포](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#normal-distribution-%EC%A0%95%EA%B7%9C%EB%B6%84%ED%8F%AC) $\mathcal{N}(0, 0.1^2)$를 맞췄다(외우기). [학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5) 점 $x = 0$과 새 점 $x = 0.5$의 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) 밀도를 각각 구하라. (정규 밀도에 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 씌운 식: $\log \mathcal{N}(x; \mu, \sigma^2) = -\tfrac{1}{2}\ln(2\pi\sigma^2) - \dfrac{(x - \mu)^2}{2\sigma^2}$)

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mu = 0$, $\sigma = 0.1$ (모델), $\sigma^2 = 0.01$
- $x = 0$ ([학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5) 점), $x = 0.5$ (새 점)

**풀이**
1. 식을 그대로 쓴다: $\log \mathcal{N}(x; \mu, \sigma^2) = -\tfrac{1}{2}\ln(2\pi\sigma^2) - \dfrac{(x - \mu)^2}{2\sigma^2}$
2. 값을 대입한다 ($x = 0$): $-\tfrac{1}{2}\ln(2\pi \times 0.01) - \dfrac{(0 - 0)^2}{2 \times 0.01}$
3. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) 안을 곱한다: $2 \times 3.1416 \times 0.01 = 0.0628$
4. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 구한다: $\ln 0.0628 = -2.7674$
5. 앞 항을 계산한다: $-\tfrac{1}{2} \times (-2.7674) = 1.3837$ (반올림 전 값으로 $1.3836$)
6. 뒤 항을 계산한다: $(0 - 0)^2 = 0$ 이므로 $0 / 0.02 = 0$
7. [학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5) 점의 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) 밀도: $1.3836 - 0 = 1.3836$
8. 값을 대입한다 ($x = 0.5$): 앞 항은 그대로 $1.3836$, 뒤 항은 $\dfrac{(0.5 - 0)^2}{2 \times 0.01}$
9. 제곱한다: $0.5^2 = 0.25$
10. 분모를 곱한다: $2 \times 0.01 = 0.02$
11. 나눈다: $0.25 / 0.02 = 12.5$
12. 뺀다: $1.3836 - 12.5 = -11.1164$

**답**: [학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5) 점 $+1.3836$, 새 점 $-11.1164$. [학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5) 점에서는 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) [우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#likelihood-%EC%9A%B0%EB%8F%84)가 높지만(좁은 종 꼭대기) 조금만 벗어난 새 점에서는 크게 나쁘다. [학습 데이터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-data-test-data-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80-%EC%8B%9C%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0)를 외운 모델의 [과적합](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#overfitting-%EA%B3%BC%EC%A0%81%ED%95%A9)이다.

</details>

## 강의와 연결

### 문제 9

[Lec14 5.1](../../Lec14_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8_%EA%B8%B0%EC%B4%88.md#51-kl-%EB%B0%9C%EC%82%B0%EC%97%90%EC%84%9C-%EC%B5%9C%EB%8C%80-%EC%9A%B0%EB%8F%84%EB%A1%9C)은 $D_{KL}(p \,\|\, q) = \mathbb{E}_p[\log p] - \mathbb{E}_p[\log q]$로 풀어 첫 항이 모델과 무관한 상수임을 보인다. $p = (0.5, 0.5)$, $q = (0.25, 0.75)$로 두 항을 각각 구해 뺀 값이 문제 1의 $0.1438$과 같은지 확인하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $p = (0.5, 0.5)$ (진짜 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))
- $q = (0.25, 0.75)$ (모델 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))

**풀이**
1. 식을 그대로 쓴다: $D_{KL}(p \,\|\, q) = \mathbb{E}_{x \sim p}[\log p(x)] - \mathbb{E}_{x \sim p}[\log q(x)]$
2. 값을 대입한다: $D_{KL} = \big(0.5 \log 0.5 + 0.5 \log 0.5\big) - \big(0.5 \log 0.25 + 0.5 \log 0.75\big)$
3. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 구한다: $\ln 0.5 = -0.6931$, $\ln 0.25 = -1.3863$, $\ln 0.75 = -0.2877$
4. 첫 괄호를 계산한다: $0.5 \times (-0.6931) + 0.5 \times (-0.6931) = -0.3466 - 0.3466 = -0.6931$ (이 값은 $q$가 바뀌어도 그대로다)
5. 둘째 괄호를 계산한다: $0.5 \times (-1.3863) + 0.5 \times (-0.2877) = -0.6931 - 0.1438 = -0.8370$
6. 뺀다: $-0.6931 - (-0.8370) = 0.1439$ (반올림 전 값으로 계산하면 $0.1438$)

**답**: $0.1438$로 문제 1과 같다. 첫 항은 모델과 상관없는 상수라서 KL을 줄이는 일은 $\mathbb{E}_p[\log q]$(평균 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) [우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#likelihood-%EC%9A%B0%EB%8F%84))를 키우는 일, 곧 [최대 우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#maximum-likelihood-%EC%B5%9C%EB%8C%80-%EC%9A%B0%EB%8F%84)와 같다.

</details>

### 문제 10

[Lec14 7](../../Lec14_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8_%EA%B8%B0%EC%B4%88.md#7-%EC%9E%90%EA%B8%B0%ED%9A%8C%EA%B7%80-%EB%AA%A8%EB%8D%B8)의 [자기회귀 모델](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#autoregressive-model-%EC%9E%90%EA%B8%B0%ED%9A%8C%EA%B7%80-%EB%AA%A8%EB%8D%B8)은 다음 [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0)의 [교차 엔트로피](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cross-entropy-%EA%B5%90%EC%B0%A8-%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC) $-\ln p_\theta(\text{정답 토큰})$을 줄인다. [정답](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5) [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0) 세 개에 모델이 준 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)이 $0.5, 0.25, 0.8$일 때 토큰당 평균 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- [정답](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5) [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0) [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $p_1 = 0.5$, $p_2 = 0.25$, $p_3 = 0.8$ (모델이 [정답](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5)에 준 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))
- $N = 3$ ([토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0) 수)

**풀이**
1. 식을 그대로 쓴다: 평균 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98) $= \dfrac{1}{N}\sum_{i=1}^{N}\big(-\ln p_i\big)$
2. 값을 대입한다 (Σ를 세 항으로 풀어 썼다): 평균 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98) $= \dfrac{1}{3}\big(-\ln 0.5 - \ln 0.25 - \ln 0.8\big)$
3. 첫 [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0): $\ln 0.5 = -0.6931$ 이므로 $-\ln 0.5 = 0.6931$
4. 둘째 [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0): $\ln 0.25 = -1.3863$ 이므로 $-\ln 0.25 = 1.3863$
5. 셋째 [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0): $\ln 0.8 = -0.2231$ 이므로 $-\ln 0.8 = 0.2231$
6. 세 값을 더한다: $0.6931 + 1.3863 + 0.2231 = 2.3026$
7. 3으로 나눈다: $2.3026 / 3 = 0.7675$

**답**: 토큰당 평균 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98) $0.7675$. [정답](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5)에 큰 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)을 줄수록 $-\ln p$가 작아진다. 세 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)의 곱 $0.5 \times 0.25 \times 0.8 = 0.1$의 $-\ln$이 합 $2.3026$이다.

</details>
