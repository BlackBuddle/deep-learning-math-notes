---
tags: [MIT딥러닝, 예제]
cssclasses: [dl-note]
---

# 예제: 삼중항 손실과 InfoNCE

개념 노트: [MM45_삼중항_손실과_InfoNCE](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/10_%ED%91%9C%ED%98%84_%ED%95%99%EC%8A%B5/MM45_%EC%82%BC%EC%A4%91%ED%95%AD_%EC%86%90%EC%8B%A4%EA%B3%BC_InfoNCE.md) · [목차](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md)

풀이는 접혀 있으니 먼저 풀어 보고 펼쳐 확인한다. 쓰는 규칙은 둘이다: [삼중항 손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#triplet-loss-%EC%82%BC%EC%A4%91%ED%95%AD-%EC%86%90%EC%8B%A4)은 $\mathcal{L} = \max(0,\ d_{ap}^2 - d_{an}^2 + m)$이고 ($d_{ap}^2$는 앵커-양성, $d_{an}^2$는 앵커-음성 거리 제곱, $m$은 여유), $\max(0, \cdot)$는 0과 괄호 안 값 중 큰 것이다. [InfoNCE](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#infonce-infonce-%EC%86%90%EC%8B%A4)는 유사도 $s$를 온도 $\tau$로 나누고, $e$의 거듭제곱([지수](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#e-exponent-e%EC%99%80-%EC%A7%80%EC%88%98))을 취해 [소프트맥스](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#softmax-%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4) [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)을 만든 뒤 양성 칸 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)에 $-\ln$을 붙인 값이다. $e$는 약 2.718인 수이고, $\ln y$는 $e$를 몇 제곱해야 $y$가 되는지다.

## 기본 (직접 대입)

### 문제 1

$d_{ap}^2 = 0.5$, $d_{an}^2 = 1.2$, 여유 $m = 0.3$일 때 [삼중항 손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#triplet-loss-%EC%82%BC%EC%A4%91%ED%95%AD-%EC%86%90%EC%8B%A4)을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $d_{ap}^2 = 0.5$ (앵커-양성 거리 제곱)
- $d_{an}^2 = 1.2$ (앵커-음성 거리 제곱)
- $m = 0.3$ (여유)

**풀이**
1. 식을 그대로 쓴다: $\mathcal{L} = \max(0,\ d_{ap}^2 - d_{an}^2 + m)$
2. 값을 대입한다: $\mathcal{L} = \max(0,\ 0.5 - 1.2 + 0.3)$
3. 괄호 안을 계산한다: 0.5 − 1.2 = −0.7, −0.7 + 0.3 = −0.4 이므로 $\mathcal{L} = \max(0,\ -0.4)$
4. 둘 중 큰 값을 고른다: $\max(0, -0.4) = 0$

**답**: $\mathcal{L} = 0$. 음성이 이미 충분히 멀어서 배울 것이 없다 ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)도 0).

</details>

### 문제 2

$d_{ap}^2 = 0.5$, $d_{an}^2 = 0.2$, $m = 0.3$일 때 [삼중항 손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#triplet-loss-%EC%82%BC%EC%A4%91%ED%95%AD-%EC%86%90%EC%8B%A4)을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $d_{ap}^2 = 0.5$, $d_{an}^2 = 0.2$, $m = 0.3$

**풀이**
1. 식을 그대로 쓴다: $\mathcal{L} = \max(0,\ d_{ap}^2 - d_{an}^2 + m)$
2. 값을 대입한다: $\mathcal{L} = \max(0,\ 0.5 - 0.2 + 0.3)$
3. 괄호 안을 계산한다: 0.5 − 0.2 = 0.3, 0.3 + 0.3 = 0.6 이므로 $\mathcal{L} = \max(0,\ 0.6)$
4. 둘 중 큰 값을 고른다: $\max(0, 0.6) = 0.6$

**답**: $\mathcal{L} = 0.6$. 음성이 양성보다 가까워 크게 벌을 받는다 (어려운 음성).

</details>

### 문제 3

앵커-양성 유사도 $s^+ = 1.0$, 음성 두 개의 유사도 $s_1^- = 0$, $s_2^- = 0$, 온도 $\tau = 1$일 때 [InfoNCE 손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#infonce-infonce-%EC%86%90%EC%8B%A4)을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $s^+ = 1.0$ (양성과의 유사도)
- $s_1^- = 0$, $s_2^- = 0$ (음성 두 개와의 유사도), $N = 2$
- $\tau = 1$ (온도)

**풀이**
1. 식을 그대로 쓴다: $\mathcal{L} = -\ln \dfrac{e^{s^+/\tau}}{e^{s^+/\tau} + \sum_{i=1}^{N} e^{s_i^-/\tau}}$
2. 값을 대입한다: $\mathcal{L} = -\ln \dfrac{e^{1.0/1}}{e^{1.0/1} + e^{0/1} + e^{0/1}}$
3. 온도로 나눈다: 1.0 / 1 = 1, 0 / 1 = 0
4. [지수](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#e-exponent-e%EC%99%80-%EC%A7%80%EC%88%98)를 계산한다 ($e^{a}$는 $e$의 $a$제곱, 0제곱은 1): $e^{1} = 2.7183$, $e^{0} = 1$
5. 분모를 더한다: $2.7183 + 1 + 1 = 4.7183$
6. 분자를 분모로 나눈다: $2.7183 / 4.7183 = 0.5761$
7. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 계산한다: $\ln 0.5761 = -0.5514$
8. 부호를 바꾼다: $\mathcal{L} = 0.5514$

**답**: $\mathcal{L} = 0.5514$ (양성 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) 0.5761)

</details>

### 문제 4

양성 유사도 0.5, 음성 하나의 유사도 0.1, 온도 $\tau = 0.5$일 때 양성 칸의 [소프트맥스](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#softmax-%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4) [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)과 [InfoNCE 손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#infonce-infonce-%EC%86%90%EC%8B%A4)을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $s^+ = 0.5$, $s^- = 0.1$ (유사도), $N = 1$
- $\tau = 0.5$ (온도)

**풀이**
1. 식을 그대로 쓴다: $p^+ = \dfrac{e^{s^+/\tau}}{e^{s^+/\tau} + e^{s^-/\tau}}$, $\mathcal{L} = -\ln p^+$
2. 값을 대입한다: $p^+ = \dfrac{e^{0.5/0.5}}{e^{0.5/0.5} + e^{0.1/0.5}}$
3. 온도로 나눈다: 0.5 / 0.5 = 1, 0.1 / 0.5 = 0.2
4. [지수](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#e-exponent-e%EC%99%80-%EC%A7%80%EC%88%98)를 계산한다: $e^{1} = 2.7183$, $e^{0.2} = 1.2214$
5. 분모를 더한다: $2.7183 + 1.2214 = 3.9397$
6. 나눈다: $p^+ = 2.7183 / 3.9397 = 0.6900$
7. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 계산한다: $\ln 0.69 = -0.3711$
8. 부호를 바꾼다: $\mathcal{L} = 0.3711$

**답**: $p^+ = 0.69$, $\mathcal{L} = 0.3711$

</details>

## 응용

### 문제 5

단위 원 위의 앵커 $\mathbf{a} = (1, 0)$, 양성 $\mathbf{p} = (0.8, 0.6)$, 음성 $\mathbf{n} = (0.6, 0.8)$, 여유 $m = 0.6$일 때 [삼중항 손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#triplet-loss-%EC%82%BC%EC%A4%91%ED%95%AD-%EC%86%90%EC%8B%A4)을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mathbf{a} = (1, 0)$ (앵커), $\mathbf{p} = (0.8, 0.6)$ (양성), $\mathbf{n} = (0.6, 0.8)$ (음성)
- $m = 0.6$ (여유)

**풀이**
1. 식을 그대로 쓴다: $\mathcal{L} = \max(0,\ \|\mathbf{a} - \mathbf{p}\|^2 - \|\mathbf{a} - \mathbf{n}\|^2 + m)$
2. 앵커-양성 거리 제곱: $\|(1,0) - (0.8,0.6)\|^2$
3. 칸별로 뺀다: $(0.2, -0.6)$
4. 제곱하고 더한다: $0.2^2 + (-0.6)^2 = 0.04 + 0.36 = 0.4$
5. 앵커-음성 거리 제곱: $\|(1,0) - (0.6,0.8)\|^2$
6. 칸별로 뺀다: $(0.4, -0.8)$
7. 제곱하고 더한다: $0.4^2 + (-0.8)^2 = 0.16 + 0.64 = 0.8$
8. 값을 대입한다: $\mathcal{L} = \max(0,\ 0.4 - 0.8 + 0.6)$
9. 괄호 안을 계산한다: 0.4 − 0.8 = −0.4, −0.4 + 0.6 = 0.2 이므로 $\mathcal{L} = \max(0,\ 0.2)$
10. 둘 중 큰 값을 고른다: $\max(0, 0.2) = 0.2$

**답**: $\mathcal{L} = 0.2$. 음성이 양성보다 멀지만(0.8 > 0.4) 여유 0.6을 채우지 못했다.

</details>

### 문제 6

$d_{ap}^2 = 0.6$, $m = 0.5$이고 음성 세 개의 거리 제곱이 $d_{an}^2 = 2.0,\ 0.9,\ 0.3$이다. 음성마다 [삼중항 손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#triplet-loss-%EC%82%BC%EC%A4%91%ED%95%AD-%EC%86%90%EC%8B%A4)을 구하고, 배울 것이 있는 ([손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 0보다 큰) 음성을 고르라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $d_{ap}^2 = 0.6$, $m = 0.5$
- $d_{an}^2 = 2.0,\ 0.9,\ 0.3$ (세 음성)

**풀이**
1. 식을 그대로 쓴다: $\mathcal{L} = \max(0,\ d_{ap}^2 - d_{an}^2 + m)$
2. 첫째 음성에 값을 대입한다: $\max(0,\ 0.6 - 2.0 + 0.5)$
3. 괄호 안을 계산한다: 0.6 − 2.0 = −1.4, −1.4 + 0.5 = −0.9 이므로 $\max(0, -0.9) = 0$
4. 둘째 음성에 값을 대입한다: $\max(0,\ 0.6 - 0.9 + 0.5)$
5. 괄호 안을 계산한다: 0.6 − 0.9 = −0.3, −0.3 + 0.5 = 0.2 이므로 $\max(0, 0.2) = 0.2$
6. 셋째 음성에 값을 대입한다: $\max(0,\ 0.6 - 0.3 + 0.5)$
7. 괄호 안을 계산한다: 0.6 − 0.3 = 0.3, 0.3 + 0.5 = 0.8 이므로 $\max(0, 0.8) = 0.8$

**답**: [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)은 **0, 0.2, 0.8**. 둘째와 셋째 음성만 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 0보다 커서 배울 것이 있고, 첫째(쉬운 음성)는 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)도 0이다.

</details>

### 문제 7

양성 유사도 0.8, 음성 유사도 0.2와 −0.1이다. 온도 $\tau = 0.25$일 때 양성 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)과 [InfoNCE 손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#infonce-infonce-%EC%86%90%EC%8B%A4)을 구하고, $\tau = 0.5$의 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98) 0.3829보다 작은지 확인하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $s^+ = 0.8$, $s_1^- = 0.2$, $s_2^- = -0.1$ (유사도), $N = 2$
- $\tau = 0.25$ (온도)

**풀이**
1. 식을 그대로 쓴다: $\mathcal{L} = -\ln \dfrac{e^{s^+/\tau}}{e^{s^+/\tau} + e^{s_1^-/\tau} + e^{s_2^-/\tau}}$
2. 값을 대입한다: $\mathcal{L} = -\ln \dfrac{e^{0.8/0.25}}{e^{0.8/0.25} + e^{0.2/0.25} + e^{-0.1/0.25}}$
3. 온도로 나눈다: 0.8 / 0.25 = 3.2, 0.2 / 0.25 = 0.8, −0.1 / 0.25 = −0.4
4. [지수](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#e-exponent-e%EC%99%80-%EC%A7%80%EC%88%98)를 계산한다: $e^{3.2} = 24.5325$, $e^{0.8} = 2.2255$, $e^{-0.4} = 0.6703$
5. 분모를 더한다: $24.5325 + 2.2255 + 0.6703 = 27.4283$
6. 나눈다: $24.5325 / 27.4283 = 0.8944$
7. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 계산한다: $\ln 0.8944 = -0.1116$
8. 부호를 바꾼다: $\mathcal{L} = 0.1116$
9. 비교한다: $0.1116 < 0.3829$

**답**: 양성 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) **0.8944**, $\mathcal{L} = $ **0.1116**. 온도가 작을수록 유사도 차이가 부풀려져 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 작아진다.

</details>

### 문제 8

양성 유사도 1.0, 음성 유사도가 모두 0, 온도 $\tau = 1$이다. 음성이 1개일 때와 3개일 때 [InfoNCE 손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#infonce-infonce-%EC%86%90%EC%8B%A4)을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $s^+ = 1.0$, $s_i^- = 0$ (모든 음성)
- $\tau = 1$, $N = 1$ 또는 $3$

**풀이**
1. 식을 그대로 쓴다: $\mathcal{L} = -\ln \dfrac{e^{s^+/\tau}}{e^{s^+/\tau} + \sum_{i=1}^{N} e^{s_i^-/\tau}}$
2. 양성 항을 계산한다: $e^{1.0/1} = e^{1} = 2.7183$
3. 음성 항 하나를 계산한다: $e^{0/1} = e^{0} = 1$
4. $N = 1$에 대입한다: $\mathcal{L} = -\ln \dfrac{2.7183}{2.7183 + 1}$
5. 더하고 나눈다: 2.7183 + 1 = 3.7183, 2.7183 / 3.7183 = 0.7311
6. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 계산하고 부호를 바꾼다: $\ln 0.7311 = -0.3133$ 이므로 $\mathcal{L} = 0.3133$
7. $N = 3$에 대입한다: $\mathcal{L} = -\ln \dfrac{2.7183}{2.7183 + 1 + 1 + 1}$
8. Σ를 풀어 쓴다: 음성 항이 세 개라 1 + 1 + 1 = 3 이고 분모는 2.7183 + 3 = 5.7183
9. 나눈다: 2.7183 / 5.7183 = 0.4754
10. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 계산하고 부호를 바꾼다: $\ln 0.4754 = -0.7437$ 이므로 $\mathcal{L} = 0.7437$

**답**: 음성 1개 **0.3133**, 3개 **0.7437**. 같은 표현이라도 음성이 많으면 문제가 어려워져 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 커진다.

</details>

## 강의와 연결

### 문제 9

Lec12 5.2의 [하드 네거티브 마이닝](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#hard-negative-mining-%ED%95%98%EB%93%9C-%EB%84%A4%EA%B1%B0%ED%8B%B0%EB%B8%8C-%EB%A7%88%EC%9D%B4%EB%8B%9D)을 숫자로 보자. $d_{ap}^2 = 0.5$, $m = 0.3$이고 후보 음성 네 개의 거리 제곱이 $d_{an}^2 = 1.5,\ 1.0,\ 0.4,\ 0.2$이다. 네 음성의 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)을 구하고, 전체 평균과 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 0보다 큰 음성만의 평균을 비교하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $d_{ap}^2 = 0.5$, $m = 0.3$
- $d_{an}^2 = 1.5,\ 1.0,\ 0.4,\ 0.2$ (후보 음성 네 개)

**풀이**
1. 식을 그대로 쓴다: $\mathcal{L} = \max(0,\ d_{ap}^2 - d_{an}^2 + m)$
2. 첫째 음성: $\max(0,\ 0.5 - 1.5 + 0.3) = \max(0,\ -0.7) = 0$
3. 둘째 음성: $\max(0,\ 0.5 - 1.0 + 0.3) = \max(0,\ -0.2) = 0$
4. 셋째 음성: $\max(0,\ 0.5 - 0.4 + 0.3) = \max(0,\ 0.4) = 0.4$
5. 넷째 음성: $\max(0,\ 0.5 - 0.2 + 0.3) = \max(0,\ 0.6) = 0.6$
6. 전체 평균을 구한다: $(0 + 0 + 0.4 + 0.6) / 4 = 1.0 / 4 = 0.25$
7. [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 0보다 큰 음성만의 평균을 구한다: $(0.4 + 0.6) / 2 = 0.5$

**답**: [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)은 **0, 0, 0.4, 0.6**. 전체 평균은 **0.25**, 어려운 음성만의 평균은 **0.5**. 쉬운 음성(앞 두 개)은 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)도 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)도 0이라 배움에 도움이 되지 않으므로, 어려운 음성을 골라 쓴다.

</details>

### 문제 10

Lec12 6.1의 [InfoNCE](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#infonce-infonce-%EC%86%90%EC%8B%A4)는 "보기 여럿 중 양성을 고르는 [소프트맥스](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#softmax-%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4) [분류](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#regression-classification-%ED%9A%8C%EA%B7%80%EC%99%80-%EB%B6%84%EB%A5%98)의 [교차 엔트로피](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cross-entropy-%EA%B5%90%EC%B0%A8-%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC)"다. 유사도 $\mathbf{s} = (0.6,\ 0.3,\ 0.0)$ (첫째가 양성), 온도 $\tau = 0.3$일 때 세 칸의 [소프트맥스](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#softmax-%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4) [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)을 구하고, 양성 칸의 $-\ln$이 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 됨을 보이라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mathbf{s} = (0.6, 0.3, 0.0)$ (유사도, 첫째 칸이 양성)
- $\tau = 0.3$ (온도)

**풀이**
1. 식을 그대로 쓴다: $p_k = \dfrac{e^{s_k/\tau}}{\sum_j e^{s_j/\tau}}$, $\mathcal{L} = -\ln p_1$
2. 온도로 나눈다: 0.6 / 0.3 = 2, 0.3 / 0.3 = 1, 0.0 / 0.3 = 0
3. [지수](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#e-exponent-e%EC%99%80-%EC%A7%80%EC%88%98)를 계산한다: $e^{2} = 7.3891$, $e^{1} = 2.7183$, $e^{0} = 1$
4. 분모를 더한다: $7.3891 + 2.7183 + 1 = 11.1074$
5. 첫째 칸 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0): $7.3891 / 11.1074 = 0.6652$
6. 둘째 칸 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0): $2.7183 / 11.1074 = 0.2447$
7. 셋째 칸 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0): $1 / 11.1074 = 0.0900$
8. 합을 확인한다: $0.6652 + 0.2447 + 0.0900 = 0.9999$ (반올림 때문에 1에서 조금 벗어난다)
9. [정답](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5)이 첫째 칸이므로 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)에 대입한다: $\mathcal{L} = -\ln 0.6652$
10. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 계산하고 부호를 바꾼다: $\ln 0.6652 = -0.4076$ 이므로 $\mathcal{L} = 0.4076$

**답**: [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) **(0.6652, 0.2447, 0.0900)**, $\mathcal{L} = $ **0.4076**. [정답](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5) 칸 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)에 $-\ln$을 붙인 것이 [소프트맥스](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#softmax-%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4) [분류](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#regression-classification-%ED%9A%8C%EA%B7%80%EC%99%80-%EB%B6%84%EB%A5%98)의 [교차 엔트로피](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cross-entropy-%EA%B5%90%EC%B0%A8-%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC)이고, 그것이 [InfoNCE 손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#infonce-infonce-%EC%86%90%EC%8B%A4)이다.

</details>
