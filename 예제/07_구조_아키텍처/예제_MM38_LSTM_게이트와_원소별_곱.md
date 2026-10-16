---
tags: [MIT딥러닝, 예제]
cssclasses: [dl-note]
---

# 예제: LSTM 게이트와 원소별 곱

개념 노트: [MM38_LSTM_게이트와_원소별_곱](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/07_%EA%B5%AC%EC%A1%B0_%EC%95%84%ED%82%A4%ED%85%8D%EC%B2%98/MM38_LSTM_%EA%B2%8C%EC%9D%B4%ED%8A%B8%EC%99%80_%EC%9B%90%EC%86%8C%EB%B3%84_%EA%B3%B1.md) · [목차](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md)

풀이는 접혀 있으니 먼저 풀어 보고 펼쳐 확인한다. 쓰는 규칙은 셋이다: [원소별 곱](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#hadamard-product-%EC%9B%90%EC%86%8C%EB%B3%84-%EA%B3%B1) $\odot$는 같은 자리끼리만 곱하고, [시그모이드](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#sigmoid-%EC%8B%9C%EA%B7%B8%EB%AA%A8%EC%9D%B4%EB%93%9C) $\sigma(z) = 1/(1 + e^{-z})$는 아무 수를 0~1로 눌러 주고, $\tanh$는 아무 수를 −1~1로 눌러 준다. $e$는 약 2.718인 수다.

## 기본 (직접 대입)

### 문제 1

[게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8) $\mathbf{g} = (0.9, 0.1, 0.5)$와 값 $\mathbf{v} = (10, -20, 4)$가 있다. $\mathbf{g} \odot \mathbf{v}$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mathbf{g} = (0.9, 0.1, 0.5)$ ([게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8), 칸마다 통과 비율)
- $\mathbf{v} = (10, -20, 4)$ ([게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)를 통과시킬 값)

**풀이**
1. 식을 그대로 쓴다: $\mathbf{g} \odot \mathbf{v} = (g_1 v_1,\ g_2 v_2,\ g_3 v_3)$
2. 값을 대입한다: $\mathbf{g} \odot \mathbf{v} = (0.9 \times 10,\ 0.1 \times (-20),\ 0.5 \times 4)$
3. 같은 자리끼리 곱한다: 0.9 × 10 = 9, 0.1 × (−20) = −2, 0.5 × 4 = 2 이므로 $\mathbf{g} \odot \mathbf{v} = (9, -2, 2)$

**답**: $(9, -2, 2)$. 첫 칸은 90%가 남고, 둘째 칸은 10%만 남는다.

</details>

### 문제 2

점수 $\mathbf{z} = (2, -2, 0)$으로 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8) $\mathbf{g} = \sigma(\mathbf{z})$를 만들어 $\mathbf{v} = (5, 5, 5)$에 곱하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mathbf{z} = (2, -2, 0)$ ([게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)를 만드는 점수)
- $\mathbf{v} = (5, 5, 5)$ (통과시킬 값)

**풀이**
1. 식을 그대로 쓴다: $\mathbf{g} \odot \mathbf{v} = (\sigma(z_1) v_1,\ \sigma(z_2) v_2,\ \sigma(z_3) v_3)$, $\sigma(z) = 1/(1 + e^{-z})$
2. 값을 대입한다: $\mathbf{g} \odot \mathbf{v} = (\sigma(2) \times 5,\ \sigma(-2) \times 5,\ \sigma(0) \times 5)$
3. 첫째 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8): $\sigma(2) = 1/(1 + e^{-2}) = 1/(1 + 0.1353) = 0.8808$
4. 둘째 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8): $\sigma(-2) = 1/(1 + e^{2}) = 1/(1 + 7.3891) = 0.1192$
5. 셋째 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8): $\sigma(0) = 1/(1 + e^{0}) = 1/(1 + 1) = 0.5$ ($e^0$은 1)
6. 같은 자리끼리 곱한다: 0.8808 × 5 = 4.404, 0.1192 × 5 = 0.596, 0.5 × 5 = 2.5 이므로 $\mathbf{g} \odot \mathbf{v} = (4.404, 0.596, 2.5)$

**답**: $(4.404,\ 0.596,\ 2.5)$

</details>

### 문제 3

한 칸짜리 셀에서 $C_{t-1} = 2$, $f_t = 0.9$, $i_t = 0.2$, $\tilde{C}_t = -0.5$이다. 새 [셀 상태](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cell-state-%EC%85%80-%EC%83%81%ED%83%9C) $C_t$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $C_{t-1} = 2$ (이전 [셀 상태](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cell-state-%EC%85%80-%EC%83%81%ED%83%9C))
- $f_t = 0.9$ (망각 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8), 이전 내용을 남길 비율)
- $i_t = 0.2$ (입력 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8), 새 내용을 쓸 비율)
- $\tilde{C}_t = -0.5$ (새로 쓸 후보 내용)

**풀이**
1. 식을 그대로 쓴다: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
2. 값을 대입한다: $C_t = 0.9 \times 2 + 0.2 \times (-0.5)$
3. 남긴 몫을 곱한다: 0.9 × 2 = 1.8 이므로 $C_t = 1.8 + 0.2 \times (-0.5)$
4. 새로 쓴 몫을 곱한다: 0.2 × (−0.5) = −0.1 이므로 $C_t = 1.8 + (-0.1)$
5. 더한다: 1.8 − 0.1 = 1.7 이므로 $C_t = 1.7$

**답**: $C_t = 1.7$

</details>

### 문제 4

[셀 상태](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cell-state-%EC%85%80-%EC%83%81%ED%83%9C) $C_t = 1$, 출력 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8) $o_t = 0.5$일 때 [은닉 상태](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#hidden-state-%EC%9D%80%EB%8B%89-%EC%83%81%ED%83%9C) $h_t = o_t \odot \tanh(C_t)$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $C_t = 1$ ([셀 상태](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cell-state-%EC%85%80-%EC%83%81%ED%83%9C))
- $o_t = 0.5$ (출력 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8), 내보낼 비율)

**풀이**
1. 식을 그대로 쓴다: $h_t = o_t \odot \tanh(C_t)$
2. 값을 대입한다: $h_t = 0.5 \times \tanh(1)$
3. $\tanh$는 아무 수를 −1~1로 눌러 주는 함수다. 값을 구한다: $\tanh(1) = 0.7616$ 이므로 $h_t = 0.5 \times 0.7616$
4. 곱한다: 0.5 × 0.7616 = 0.3808 이므로 $h_t = 0.3808$

**답**: $h_t = 0.3808$

</details>

## 응용

### 문제 5

칸이 둘인 셀을 한 단계 갱신하라. $C_{t-1} = (1, -2)$, 망각 점수 $(0, 2)$, 입력 점수 $(2, -2)$, 후보 점수 $(0.5, 1)$이다. [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)는 $\sigma$로, 후보는 $\tanh$로 만든다.

<details><summary>풀이와 답</summary>

**주어진 값**
- $C_{t-1} = (1, -2)$ (이전 [셀 상태](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cell-state-%EC%85%80-%EC%83%81%ED%83%9C))
- 망각 점수 $(0, 2)$ → $f_t = \sigma(0, 2)$
- 입력 점수 $(2, -2)$ → $i_t = \sigma(2, -2)$
- 후보 점수 $(0.5, 1)$ → $\tilde{C}_t = \tanh(0.5, 1)$

**풀이**
1. 식을 그대로 쓴다: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
2. 값을 대입한다: $C_t = \sigma(0, 2) \odot (1, -2) + \sigma(2, -2) \odot \tanh(0.5, 1)$
3. 망각 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8): $\sigma(0) = 0.5$, $\sigma(2) = 0.8808$ 이므로 $f_t = (0.5, 0.8808)$
4. 입력 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8): $\sigma(2) = 0.8808$, $\sigma(-2) = 0.1192$ 이므로 $i_t = (0.8808, 0.1192)$
5. 후보: $\tanh(0.5) = 0.4621$, $\tanh(1) = 0.7616$ 이므로 $\tilde{C}_t = (0.4621, 0.7616)$
6. 남긴 몫을 칸마다 곱한다: 0.5 × 1 = 0.5, 0.8808 × (−2) = −1.7616 이므로 $f_t \odot C_{t-1} = (0.5, -1.7616)$
7. 새로 쓴 몫을 칸마다 곱한다: 0.8808 × 0.4621 = 0.407, 0.1192 × 0.7616 = 0.0908 이므로 $i_t \odot \tilde{C}_t = (0.407, 0.0908)$
8. 칸끼리 더한다: 0.5 + 0.407 = 0.907, −1.7616 + 0.0908 = −1.6708 이므로 $C_t = (0.907, -1.6708)$

**답**: $C_t = (0.907,\ -1.6708)$

</details>

### 문제 6

[셀 상태](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cell-state-%EC%85%80-%EC%83%81%ED%83%9C)를 따라 거슬러 가는 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)는 $\frac{\partial C_T}{\partial C_0} \approx \prod_{t=1}^{T} f_t$다. 세 단계의 망각 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)가 $f_1 = 0.9$, $f_2 = 0.8$, $f_3 = 0.5$일 때 처음 셀 값이 몇 배로 남는지 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $f_1 = 0.9$, $f_2 = 0.8$, $f_3 = 0.5$ (단계별 망각 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8))
- $T = 3$ (지나가는 단계 수)

**풀이**
1. 식을 그대로 쓴다: $\frac{\partial C_T}{\partial C_0} \approx \prod_{t=1}^{T} f_t$
2. 값을 대입한다: $\frac{\partial C_3}{\partial C_0} \approx \prod_{t=1}^{3} f_t$
3. 파이(Π, 모두 곱하라는 기호)를 풀어 쓴다: $f_1 \times f_2 \times f_3 = 0.9 \times 0.8 \times 0.5$
4. 앞의 둘을 곱한다: 0.9 × 0.8 = 0.72 이므로 $0.72 \times 0.5$
5. 남은 하나를 곱한다: 0.72 × 0.5 = 0.36

**답**: **0.36배**. [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)가 하나라도 작으면 그 단계에서 많이 잃는다.

</details>

### 문제 7

세 칸 셀 $C_{t-1} = (3, -1, 0.5)$에서 (가) $f_t = (1, 1, 1)$, $i_t = (0, 0, 0)$이면 $C_t$는? (나) $f_t = (0, 0, 0)$, $i_t = (1, 1, 1)$, $\tilde{C}_t = (0.2, -0.4, 0.6)$이면 $C_t$는?

<details><summary>풀이와 답</summary>

**주어진 값**
- $C_{t-1} = (3, -1, 0.5)$ (이전 [셀 상태](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cell-state-%EC%85%80-%EC%83%81%ED%83%9C))
- (가) $f_t = (1, 1, 1)$, $i_t = (0, 0, 0)$ (완전히 남기고 쓰지 않음)
- (나) $f_t = (0, 0, 0)$, $i_t = (1, 1, 1)$, $\tilde{C}_t = (0.2, -0.4, 0.6)$ (완전히 지우고 새로 씀)

**풀이**
1. 식을 그대로 쓴다: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
2. (가) 값을 대입한다: $C_t = (1, 1, 1) \odot (3, -1, 0.5) + (0, 0, 0) \odot \tilde{C}_t$
3. (가) 칸마다 곱한다: 1 × 3 = 3, 1 × (−1) = −1, 1 × 0.5 = 0.5 이고, 둘째 항은 0 × (무엇이든) = 0 이므로 $C_t = (3, -1, 0.5) + (0, 0, 0)$
4. (가) 더한다: $C_t = (3, -1, 0.5)$
5. (나) 값을 대입한다: $C_t = (0, 0, 0) \odot (3, -1, 0.5) + (1, 1, 1) \odot (0.2, -0.4, 0.6)$
6. (나) 칸마다 곱한다: 첫째 항은 0 × (무엇이든) = 0, 둘째 항은 1 × 0.2 = 0.2, 1 × (−0.4) = −0.4, 1 × 0.6 = 0.6 이므로 $C_t = (0, 0, 0) + (0.2, -0.4, 0.6)$
7. (나) 더한다: $C_t = (0.2, -0.4, 0.6)$

**답**: (가) $C_t = (3, -1, 0.5)$, 이전 값 그대로(항등 함수). (나) $C_t = (0.2, -0.4, 0.6)$, 후보로 완전히 교체.

</details>

### 문제 8

망각 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)가 매 단계 $f = 0.95$일 때, 처음 기억이 절반(0.5) 아래로 떨어지는 첫 단계 수 $T$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $f = 0.95$ (매 단계 남길 비율)
- 기준 0.5 (처음 기억의 절반)

**풀이**
1. 식을 그대로 쓴다: $\prod_{t=1}^{T} f_t = f^{T}$
2. 값을 대입한다: $0.95^{T} < 0.5$를 만족하는 가장 작은 $T$를 찾는다
3. $T = 13$을 넣어 본다: $0.95^{13} = 0.5133$ (아직 절반 위)
4. $T = 14$를 넣어 본다: $0.95^{14} = 0.4877$ (처음으로 절반 아래)

**답**: **14단계**

</details>

## 강의와 연결

### 문제 9

Lec10 6의 [LSTM](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#lstm-long-short-term-memory-%EC%9E%A5%EB%8B%A8%EA%B8%B0-%EB%A9%94%EB%AA%A8%EB%A6%AC) 한 단계를 칸이 둘인 셀로 계산하라. $C_{t-1} = (2, -1)$, 망각 점수 $(2, -2)$, 입력 점수 $(-2, 2)$, 후보 점수 $(1, -1)$, 출력 점수 $(0, 2)$이다. 새 [셀 상태](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cell-state-%EC%85%80-%EC%83%81%ED%83%9C) $C_t$와 [은닉 상태](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#hidden-state-%EC%9D%80%EB%8B%89-%EC%83%81%ED%83%9C) $h_t = o_t \odot \tanh(C_t)$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $C_{t-1} = (2, -1)$ (이전 [셀 상태](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cell-state-%EC%85%80-%EC%83%81%ED%83%9C))
- 망각 점수 $(2, -2)$, 입력 점수 $(-2, 2)$, 출력 점수 $(0, 2)$ ([게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)를 만드는 점수)
- 후보 점수 $(1, -1)$ (후보를 만드는 점수)

**풀이**
1. 식을 그대로 쓴다: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$, $h_t = o_t \odot \tanh(C_t)$
2. 값을 대입한다: $C_t = \sigma(2, -2) \odot (2, -1) + \sigma(-2, 2) \odot \tanh(1, -1)$, $h_t = \sigma(0, 2) \odot \tanh(C_t)$
3. 망각 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8): $\sigma(2) = 0.8808$, $\sigma(-2) = 0.1192$ 이므로 $f_t = (0.8808, 0.1192)$
4. 입력 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8): $\sigma(-2) = 0.1192$, $\sigma(2) = 0.8808$ 이므로 $i_t = (0.1192, 0.8808)$
5. 후보: $\tanh(1) = 0.7616$, $\tanh(-1) = -0.7616$ 이므로 $\tilde{C}_t = (0.7616, -0.7616)$
6. 남긴 몫을 칸마다 곱한다: 0.8808 × 2 = 1.7616, 0.1192 × (−1) = −0.1192 이므로 $(1.7616, -0.1192)$
7. 새로 쓴 몫을 칸마다 곱한다: 0.1192 × 0.7616 = 0.0908, 0.8808 × (−0.7616) = −0.6708 이므로 $(0.0908, -0.6708)$
8. 칸끼리 더한다: 1.7616 + 0.0908 = 1.8524, −0.1192 + (−0.6708) = −0.79 이므로 $C_t = (1.8524, -0.79)$
9. [셀 상태](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cell-state-%EC%85%80-%EC%83%81%ED%83%9C)를 누른다: $\tanh(1.8524) = 0.952$, $\tanh(-0.79) = -0.6584$
10. 출력 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8): $\sigma(0) = 0.5$, $\sigma(2) = 0.8808$ 이므로 $o_t = (0.5, 0.8808)$
11. 칸마다 곱한다: 0.5 × 0.952 = 0.476, 0.8808 × (−0.6584) = −0.5799 이므로 $h_t = (0.476, -0.5799)$

**답**: $C_t = (1.8524,\ -0.79)$, $h_t = (0.476,\ -0.5799)$. 첫 칸은 옛 값 2를 거의 유지했고, 둘째 칸은 옛 값 −1을 지우고 새 후보로 바꿨다.

</details>

### 문제 10

Lec10에서 [RNN](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rnn-recurrent-neural-network-%EC%88%9C%ED%99%98-%EC%8B%A0%EA%B2%BD%EB%A7%9D)은 같은 $w$가 거듭 곱해져 기억을 잃는다고 했다. [RNN](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rnn-recurrent-neural-network-%EC%88%9C%ED%99%98-%EC%8B%A0%EA%B2%BD%EB%A7%9D)의 $w = 0.9$와 [LSTM](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#lstm-long-short-term-memory-%EC%9E%A5%EB%8B%A8%EA%B8%B0-%EB%A9%94%EB%AA%A8%EB%A6%AC)의 망각 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8) $f = 0.99$로 30단계 뒤에 처음 값이 몇 배로 남는지 비교하라. [LSTM](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#lstm-long-short-term-memory-%EC%9E%A5%EB%8B%A8%EA%B8%B0-%EB%A9%94%EB%AA%A8%EB%A6%AC)이 [RNN](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rnn-recurrent-neural-network-%EC%88%9C%ED%99%98-%EC%8B%A0%EA%B2%BD%EB%A7%9D)의 몇 배인가?

<details><summary>풀이와 답</summary>

**주어진 값**
- $w = 0.9$ ([RNN](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rnn-recurrent-neural-network-%EC%88%9C%ED%99%98-%EC%8B%A0%EA%B2%BD%EB%A7%9D) 순환 배율, 매 단계 곱해지는 값)
- $f = 0.99$ ([LSTM](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#lstm-long-short-term-memory-%EC%9E%A5%EB%8B%A8%EA%B8%B0-%EB%A9%94%EB%AA%A8%EB%A6%AC) 망각 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8), 매 단계 남길 비율)
- $T = 30$ (단계 수)

**풀이**
1. 식을 그대로 쓴다: 남는 비율 $= \prod_{t=1}^{T} (\text{매 단계 배율}) = (\text{배율})^{T}$
2. [RNN](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rnn-recurrent-neural-network-%EC%88%9C%ED%99%98-%EC%8B%A0%EA%B2%BD%EB%A7%9D) 값을 대입한다: $0.9^{30}$
3. [RNN](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rnn-recurrent-neural-network-%EC%88%9C%ED%99%98-%EC%8B%A0%EA%B2%BD%EB%A7%9D)을 계산한다: $0.9^{30} = 0.0424$
4. [LSTM](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#lstm-long-short-term-memory-%EC%9E%A5%EB%8B%A8%EA%B8%B0-%EB%A9%94%EB%AA%A8%EB%A6%AC) 값을 대입한다: $0.99^{30}$
5. [LSTM](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#lstm-long-short-term-memory-%EC%9E%A5%EB%8B%A8%EA%B8%B0-%EB%A9%94%EB%AA%A8%EB%A6%AC)을 계산한다: $0.99^{30} = 0.7397$
6. 비교하려고 나눈다: $0.7397 / 0.0424 = 17.4$ (소수 첫째 자리까지)

**답**: [RNN](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rnn-recurrent-neural-network-%EC%88%9C%ED%99%98-%EC%8B%A0%EA%B2%BD%EB%A7%9D)은 **0.0424**배, [LSTM](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#lstm-long-short-term-memory-%EC%9E%A5%EB%8B%A8%EA%B8%B0-%EB%A9%94%EB%AA%A8%EB%A6%AC)은 **0.7397**배가 남는다. [LSTM](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#lstm-long-short-term-memory-%EC%9E%A5%EB%8B%A8%EA%B8%B0-%EB%A9%94%EB%AA%A8%EB%A6%AC)이 약 **17.4배** 더 남는다.

</details>
