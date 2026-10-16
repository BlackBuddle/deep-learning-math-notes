---
tags: [MIT딥러닝, 예제]
cssclasses: [dl-note]
---

# 예제: Best-of-N과 적중 확률

개념 노트: [MM80_Best-of-N과_적중_확률](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/13_%EC%96%B8%EC%96%B4_%EB%AA%A8%EB%8D%B8%EA%B3%BC_%EC%B6%94%EB%A1%A0/MM80_Best-of-N%EA%B3%BC_%EC%A0%81%EC%A4%91_%ED%99%95%EB%A5%A0.md) · [목차](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md)

풀이는 접혀 있으니 먼저 풀어 보고 펼쳐 확인한다. 쓰는 식은 적중 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $P = 1 - (1-p)^N$, 필요한 횟수 $N \ge \ln(1-c)/\ln(1-p)$, 문장 점수 $s(y) = \sum_t \ln p(y_t \mid \dots)$이다.

## 기본 (직접 대입)

### 문제 1

한 번 뽑아 맞힐 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)이 $p = 0.25$인 문제를 $N = 2$번 뽑고 [검증기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#verifier-%EA%B2%80%EC%A6%9D%EA%B8%B0)로 확인한다. 적어도 한 번 맞힐 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)은?

<details><summary>풀이와 답</summary>

**주어진 값**
- $p = 0.25$ (한 번 맞힐 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))
- $N = 2$ (뽑는 횟수)

**풀이**
1. 식을 그대로 쓴다: $P = 1 - (1 - p)^N$
2. 값을 대입한다: $P = 1 - (1 - 0.25)^2$
3. 괄호 안을 뺀다: $1 - 0.25 = 0.75$ 이므로 $P = 1 - 0.75^2$
4. 제곱한다: $0.75^2 = 0.75 \times 0.75 = 0.5625$ 이므로 $P = 1 - 0.5625$
5. 뺀다: $1 - 0.5625 = 0.4375$

**답**: $0.4375$ (두 번 모두 틀릴 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) 0.5625를 1에서 뺀 값)

</details>

### 문제 2

한 번 맞힐 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)이 $p = 0.1$이고 $N = 5$번 뽑는다. 적어도 한 번 맞힐 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)은?

<details><summary>풀이와 답</summary>

**주어진 값**
- $p = 0.1$ (한 번 맞힐 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))
- $N = 5$ (뽑는 횟수)

**풀이**
1. 식을 그대로 쓴다: $P = 1 - (1 - p)^N$
2. 값을 대입한다: $P = 1 - (1 - 0.1)^5$
3. 괄호 안을 뺀다: $1 - 0.1 = 0.9$ 이므로 $P = 1 - 0.9^5$
4. 거듭제곱을 한다: $0.9 \times 0.9 = 0.81$, $0.81 \times 0.9 = 0.729$, $0.729 \times 0.9 = 0.6561$, $0.6561 \times 0.9 = 0.59049$ 이므로 $P = 1 - 0.59049$
5. 뺀다: $1 - 0.59049 = 0.40951$

**답**: 약 $0.4095$

</details>

### 문제 3

[토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0) 3개짜리 문장의 토큰별 [조건부 확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#conditional-probability-%EC%A1%B0%EA%B1%B4%EB%B6%80-%ED%99%95%EB%A5%A0)이 $0.5, 0.4, 0.9$이다. 문장 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)과 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) [우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#likelihood-%EC%9A%B0%EB%8F%84) 점수를 구하고, 두 값이 맞는지 확인하라. ($\ln 0.5 = -0.6931$, $\ln 0.4 = -0.9163$, $\ln 0.9 = -0.1054$는 계산기로 구한다.)

<details><summary>풀이와 답</summary>

**주어진 값**
- $p(y_1) = 0.5$, $p(y_2 \mid y_1) = 0.4$, $p(y_3 \mid y_1, y_2) = 0.9$ (토큰별 [조건부 확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#conditional-probability-%EC%A1%B0%EA%B1%B4%EB%B6%80-%ED%99%95%EB%A5%A0))

**풀이**
1. 식을 그대로 쓴다: 문장 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $= p(y_1)\,p(y_2 \mid y_1)\,p(y_3 \mid y_1, y_2)$, 점수 $s = \sum_{t=1}^{3} \ln p(y_t \mid \dots)$
2. [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)에 대입해 곱한다: $0.5 \times 0.4 \times 0.9$
3. 앞 곱셈을 한다: $0.5 \times 0.4 = 0.2$ 이므로 $0.2 \times 0.9$
4. 곱한다: $0.2 \times 0.9 = 0.18$
5. 점수에 대입한다 (Σ를 세 항으로 펼친다): $s = \ln 0.5 + \ln 0.4 + \ln 0.9$
6. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) 값을 넣는다: $s = -0.6931 + (-0.9163) + (-0.1054)$
7. 더한다: $-0.6931 - 0.9163 = -1.6094$, $-1.6094 - 0.1054 = -1.7148$
8. 확인한다: $\ln 0.18 = -1.7148$ (계산기)

**답**: 문장 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $0.18$, 점수 $-1.7148$. 점수는 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)의 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)라서 $\ln 0.18$과 같다 (곱이 합으로 바뀐다).

</details>

### 문제 4

한 번 맞힐 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)이 $p = 0.5$일 때 성공 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $c = 0.9$ 이상을 얻으려면 최소 몇 번 뽑아야 하는가?

<details><summary>풀이와 답</summary>

**주어진 값**
- $p = 0.5$ (한 번 맞힐 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))
- $c = 0.9$ (목표 성공 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))

**풀이**
1. 식을 그대로 쓴다: $N \ge \dfrac{\ln(1 - c)}{\ln(1 - p)}$
2. 값을 대입한다: $N \ge \dfrac{\ln(1 - 0.9)}{\ln(1 - 0.5)}$
3. 괄호 안을 뺀다: $1 - 0.9 = 0.1$, $1 - 0.5 = 0.5$ 이므로 $N \ge \dfrac{\ln 0.1}{\ln 0.5}$
4. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 구한다 ($\ln$은 자연로그, 계산기로 구한다): $\ln 0.1 = -2.3026$, $\ln 0.5 = -0.6931$ 이므로 $N \ge \dfrac{-2.3026}{-0.6931}$
5. 나눈다 (음수 ÷ 음수 = 양수): $2.3026 \div 0.6931 = 3.32$ 이므로 $N \ge 3.32$
6. 횟수는 정수이므로 올린다: $N = 4$

**답**: $4$번. 확인: $1 - 0.5^4 = 1 - 0.0625 = 0.9375 \ge 0.9$.

</details>

## 응용

### 문제 5

문장 세 개의 토큰별 [조건부 확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#conditional-probability-%EC%A1%B0%EA%B1%B4%EB%B6%80-%ED%99%95%EB%A5%A0)이 아래와 같다. 문장 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)이 가장 큰 것을 고르는 Best-of-3은 어느 문장을 고르는가? 첫 [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0)만 보고 욕심껏 고르는 탐욕 디코딩이라면 어느 문장 쪽으로 가는가?

| 문장 | 첫 [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0) | 둘째 [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0) |
|---|---|---|
| A | 0.7 | 0.6 |
| B | 0.9 | 0.4 |
| C | 0.5 | 0.9 |

<details><summary>풀이와 답</summary>

**주어진 값**
- A: $0.7$, $0.6$ / B: $0.9$, $0.4$ / C: $0.5$, $0.9$ (토큰별 [조건부 확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#conditional-probability-%EC%A1%B0%EA%B1%B4%EB%B6%80-%ED%99%95%EB%A5%A0))

**풀이**
1. 식을 그대로 쓴다: 문장 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $= p(y_1)\,p(y_2 \mid y_1)$
2. A를 계산한다: $0.7 \times 0.6 = 0.42$
3. B를 계산한다: $0.9 \times 0.4 = 0.36$
4. C를 계산한다: $0.5 \times 0.9 = 0.45$
5. 큰 쪽을 고른다 ([argmax](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#argmax-%EC%B5%9C%EB%8C%93%EA%B0%92-%EC%9C%84%EC%B9%98)): $0.45 > 0.42 > 0.36$ 이므로 C
6. [탐욕 디코딩](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#greedy-decoding-%ED%83%90%EC%9A%95-%EB%94%94%EC%BD%94%EB%94%A9)은 첫 [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0)이 가장 큰 쪽을 고른다: $0.9 > 0.7 > 0.5$ 이므로 B

**답**: Best-of-3은 문장 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $0.45$인 **C**를 고른다. [탐욕 디코딩](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#greedy-decoding-%ED%83%90%EC%9A%95-%EB%94%94%EC%BD%94%EB%94%A9)은 첫 [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0)이 가장 큰 B(문장 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) 0.36)로 가서 C에 닿지 못한다.

</details>

### 문제 6

한 번 맞힐 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $p = 0.3$일 때 $N = 4, 8, 16$의 적중 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)을 구하고, N을 두 배로 늘릴 때마다 얼마씩 오르는지 비교하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $p = 0.3$ (한 번 맞힐 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)), 한 번 틀릴 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $1 - p = 0.7$
- $N = 4, 8, 16$ (뽑는 횟수)

**풀이**
1. 식을 그대로 쓴다: $P = 1 - (1 - p)^N = 1 - 0.7^N$
2. $N = 4$를 계산한다: $0.7^4 = 0.49 \times 0.49 = 0.2401$ 이므로 $P = 1 - 0.2401 = 0.7599$
3. $N = 8$을 계산한다: $0.7^8 = 0.2401 \times 0.2401 = 0.0576$ 이므로 $P = 1 - 0.0576 = 0.9424$
4. $N = 16$을 계산한다: $0.7^{16} = 0.0576 \times 0.0576 = 0.0033$ 이므로 $P = 1 - 0.0033 = 0.9967$
5. 4 → 8의 오름: $0.9424 - 0.7599 = 0.1825$
6. 8 → 16의 오름: $0.9967 - 0.9424 = 0.0543$

**답**: $0.7599$, $0.9424$, $0.9967$. 오르는 폭이 약 $0.18$에서 약 $0.05$로 줄어든다. 계산을 두 배로 쓸 때마다 얻는 이득이 점점 작아진다.

</details>

### 문제 7

한 번 맞힐 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)이 $p = 0.4$이다. 온도를 0보다 크게 두고 $N = 10$번 [독립](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#independence-%EB%8F%85%EB%A6%BD)으로 뽑으면 적중 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)이 얼마인가? 온도를 0으로 두어 열 번 모두 똑같은 답이 나오면 얼마인가?

<details><summary>풀이와 답</summary>

**주어진 값**
- $p = 0.4$ (한 번 맞힐 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))
- $N = 10$ (뽑는 횟수)

**풀이**
1. [독립](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#independence-%EB%8F%85%EB%A6%BD)으로 뽑는 경우의 식을 쓴다: $P = 1 - (1 - p)^N$
2. 값을 대입한다: $P = 1 - (1 - 0.4)^{10}$
3. 괄호 안을 뺀다: $1 - 0.4 = 0.6$ 이므로 $P = 1 - 0.6^{10}$
4. 거듭제곱을 한다: $0.6^{10} = 0.0060$ (0.6을 열 번 곱한다) 이므로 $P = 1 - 0.0060$
5. 뺀다: $1 - 0.0060 = 0.9940$
6. 온도 0에서는 열 번이 모두 같은 답이다. 한 번 뽑은 것과 같으므로 $P = p$
7. 값을 쓴다: $P = 0.4$

**답**: [독립](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#independence-%EB%8F%85%EB%A6%BD)으로 뽑으면 약 $0.9940$, 온도 0이면 $0.4$. 표본이 서로 독립이어야 식이 맞는다.

</details>

### 문제 8

한 번 맞힐 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)이 $p = 0.2$일 때 성공 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $0.99$ 이상을 얻으려면 최소 몇 번 뽑아야 하는가? 얻은 $N$으로 확인하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $p = 0.2$ (한 번 맞힐 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))
- $c = 0.99$ (목표 성공 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))

**풀이**
1. 식을 그대로 쓴다: $N \ge \dfrac{\ln(1 - c)}{\ln(1 - p)}$
2. 값을 대입한다: $N \ge \dfrac{\ln(1 - 0.99)}{\ln(1 - 0.2)}$
3. 괄호 안을 뺀다: $1 - 0.99 = 0.01$, $1 - 0.2 = 0.8$ 이므로 $N \ge \dfrac{\ln 0.01}{\ln 0.8}$
4. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 구한다: $\ln 0.01 = -4.6052$, $\ln 0.8 = -0.2231$ 이므로 $N \ge \dfrac{-4.6052}{-0.2231}$
5. 나눈다: $4.6052 \div 0.2231 = 20.64$ 이므로 $N \ge 20.64$
6. 정수로 올린다: $N = 21$
7. 확인한다: $1 - 0.8^{21} = 1 - 0.0092 = 0.9908 \ge 0.99$ ($N = 20$이면 $1 - 0.8^{20} = 0.9885$로 모자란다)

**답**: $21$번

</details>

## 강의와 연결

### 문제 9

[Lec24 6.2](../../Lec24_%EC%B6%94%EB%A1%A0_%EB%B0%A9%EB%B2%95.md#62-%EA%B2%80%EC%A6%9D%EC%9D%B4-%EC%83%9D%EC%84%B1%EB%B3%B4%EB%8B%A4-%EC%89%BD%EB%8B%A4): 코드 문제 하나를 모델이 한 번에 맞힐 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)은 $p = 0.05$뿐이지만, 단위 테스트([검증기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#verifier-%EA%B2%80%EC%A6%9D%EA%B8%B0))로 맞는 코드를 알아볼 수 있다. $N = 50$번 뽑으면 적어도 하나가 통과할 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)은?

<details><summary>풀이와 답</summary>

**주어진 값**
- $p = 0.05$ (한 번에 맞는 코드를 낼 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))
- $N = 50$ (뽑는 횟수)
- [검증기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#verifier-%EA%B2%80%EC%A6%9D%EA%B8%B0)가 완벽하다고 가정한다

**풀이**
1. 식을 그대로 쓴다: $P = 1 - (1 - p)^N$
2. 값을 대입한다: $P = 1 - (1 - 0.05)^{50}$
3. 괄호 안을 뺀다: $1 - 0.05 = 0.95$ 이므로 $P = 1 - 0.95^{50}$
4. 거듭제곱을 한다: $0.95^{50} = 0.0769$ (0.95를 쉰 번 곱한다, 계산기) 이므로 $P = 1 - 0.0769$
5. 뺀다: $1 - 0.0769 = 0.9231$

**답**: 약 $0.9231$. 한 번에는 5%뿐이던 문제가 50번 뽑으면 92%로 풀린다. 이것이 "[검증](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#validation-data-%EA%B2%80%EC%A6%9D-%EB%8D%B0%EC%9D%B4%ED%84%B0)이 생성보다 쉬울 때" 표본을 더 뽑아 계산을 쓰는 이득이다.

</details>

### 문제 10

[Lec24 4.2](../../Lec24_%EC%B6%94%EB%A1%A0_%EB%B0%A9%EB%B2%95.md#42-%ED%83%90%EC%9A%95-%EC%83%98%ED%94%8C%EB%A7%81%EC%9D%80-%EC%9A%B0%EB%8F%84%EB%A5%BC-%EC%B5%9C%EB%8C%80%EB%A1%9C-%EB%A7%8C%EB%93%A4%EC%A7%80-%EC%95%8A%EB%8A%94%EB%8B%A4): 두 토큰짜리 문장에서 첫 [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0)이 $a$($0.6$)이면 둘째 [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0) [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)이 $0.5$, 첫 [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0)이 $b$($0.4$)이면 둘째 [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0) [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)이 $0.9$이다. [탐욕 디코딩](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#greedy-decoding-%ED%83%90%EC%9A%95-%EB%94%94%EC%BD%94%EB%94%A9)이 만드는 문장과 [우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#likelihood-%EC%9A%B0%EB%8F%84)가 가장 큰 문장은 같은가? [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) [우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#likelihood-%EC%9A%B0%EB%8F%84) 점수로도 확인하라. ($\ln 0.3 = -1.2040$, $\ln 0.36 = -1.0217$)

<details><summary>풀이와 답</summary>

**주어진 값**
- $p(a) = 0.6$, $p(\text{둘째} \mid a) = 0.5$ (문장 $a$ 쪽)
- $p(b) = 0.4$, $p(\text{둘째} \mid b) = 0.9$ (문장 $b$ 쪽)

**풀이**
1. [탐욕 디코딩](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#greedy-decoding-%ED%83%90%EC%9A%95-%EB%94%94%EC%BD%94%EB%94%A9)은 첫 [토큰](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#token-%ED%86%A0%ED%81%B0)이 가장 큰 쪽을 고른다: $0.6 > 0.4$ 이므로 $a$ 문장
2. 문장 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)의 식을 쓴다: $p(y_1)\,p(y_2 \mid y_1)$
3. $a$ 문장에 대입한다: $0.6 \times 0.5 = 0.3$
4. $b$ 문장에 대입한다: $0.4 \times 0.9 = 0.36$
5. 비교한다: $0.36 > 0.3$ 이므로 [우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#likelihood-%EC%9A%B0%EB%8F%84)가 큰 문장은 $b$
6. 점수로 확인한다: $a$ 문장 $\ln 0.3 = -1.2040$, $b$ 문장 $\ln 0.36 = -1.0217$
7. 큰 쪽을 고른다: $-1.0217 > -1.2040$ 이므로 $b$

**답**: [탐욕 디코딩](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#greedy-decoding-%ED%83%90%EC%9A%95-%EB%94%94%EC%BD%94%EB%94%A9)은 $a$ 문장([확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $0.3$)을 만들지만, [우도](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#likelihood-%EC%9A%B0%EB%8F%84)가 가장 큰 문장은 $b$([확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $0.36$)다. 두 문장을 모두 뽑아 점수를 비교하는 Best-of-2는 $b$를 고른다.

</details>
