---
title: 예제 MM10 선형층과 ReLU의 역전파
type: note
tags: [MIT딥러닝, 예제]
created: 2026-10-15
cssclasses: [dl-note]
---

# 예제: 선형층과 ReLU의 역전파

개념 노트: [MM10_선형층과_ReLU의_역전파](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/05_%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%9D%98_%EA%B3%84%EC%82%B0/MM10_%EC%84%A0%ED%98%95%EC%B8%B5%EA%B3%BC_ReLU%EC%9D%98_%EC%97%AD%EC%A0%84%ED%8C%8C.md) · [목차](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md)

풀이는 접혀 있으니 먼저 직접 풀어 보고, 펼쳐서 확인한다.

## 기본 (직접 대입)

### 문제 1
$W = \begin{pmatrix} 2 & -1 \\ 1 & 1 \end{pmatrix}$, $x = (1, 3)$일 때 [선형층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#linear-layer-%EC%84%A0%ED%98%95%EC%B8%B5) $z = Wx$와 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 출력 $h = \mathrm{ReLU}(z)$를 구하라. ([ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)는 음수를 0으로 바꾼다)

<details><summary>풀이와 답</summary>

**주어진 값**
- $W = \begin{pmatrix} 2 & -1 \\ 1 & 1 \end{pmatrix}$ ([가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 행렬)
- $x = (1, 3)$ (입력)

**풀이**
1. 식을 그대로 쓴다: $z = Wx$, $h = \mathrm{ReLU}(z)$
2. 값을 대입한다: $z = \begin{pmatrix} 2 & -1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} 1 \\ 3 \end{pmatrix}$
3. 1번째 칸을 펼친다 (W의 1행과 x): $z_1 = 2 \times 1 + (-1) \times 3$
4. 계산한다: 2 − 3 = −1 이므로 $z_1 = -1$
5. 2번째 칸을 펼친다 (W의 2행과 x): $z_2 = 1 \times 1 + 1 \times 3$
6. 계산한다: 1 + 3 = 4 이므로 $z_2 = 4$
7. [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)를 적용한다: −1은 음수라 0, 4는 그대로 → $h = (0, 4)$

**답**: $z = (-1, 4)$, $h = (0, 4)$

</details>

### 문제 2
문제 1의 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 출력 $h$ 쪽에서 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $g_h = (5, 2)$가 내려왔다. [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8) $\mathbb{1}[z > 0]$와 $z$ 쪽 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $g_z = g_h \odot \mathbb{1}[z > 0]$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $z = (-1, 4)$ (문제 1에서 저장해 둔 [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C) 값)
- $g_h = (5, 2)$ (위에서 내려온 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))

**풀이**
1. 식을 그대로 쓴다: $g_z = g_h \odot \mathbb{1}[z > 0]$ (같은 자리끼리 곱, [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)는 z가 양수면 1, 아니면 0)
2. [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)를 만든다: −1은 양수가 아니라 0, 4는 양수라 1 이므로 $\mathbb{1}[z > 0] = (0, 1)$
3. 값을 대입한다: $g_z = (5, 2) \odot (0, 1)$
4. 같은 자리끼리 곱한다: 5 × 0 = 0, 2 × 1 = 2

**답**: [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8) $(0, 1)$, $g_z = (0, 2)$. [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C)에서 꺼졌던 첫 칸은 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 막힌다.

</details>

### 문제 3
문제 1의 [선형층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#linear-layer-%EC%84%A0%ED%98%95%EC%B8%B5)에서 $g_{out} = g_z = (0, 2)$일 때 입력 쪽 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $g_{in} = g_{out}W$와 [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $\partial J/\partial W = g_{out}^\top x^\top$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $g_{out} = (0, 2)$ ([선형층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#linear-layer-%EC%84%A0%ED%98%95%EC%B8%B5) 출력 z 쪽 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))
- $W = \begin{pmatrix} 2 & -1 \\ 1 & 1 \end{pmatrix}$, $x = (1, 3)$

**풀이**
1. 식을 그대로 쓴다: $g_{in} = g_{out}\,W, \qquad \frac{\partial J}{\partial W} = g_{out}^\top\, x^\top$
2. 값을 대입한다: $g_{in} = (0,\ 2)\begin{pmatrix} 2 & -1 \\ 1 & 1 \end{pmatrix}$
3. 1번째 칸을 펼친다 (W의 1열과 곱함): $0 \times 2 + 2 \times 1$
4. 계산한다: 0 + 2 = 2
5. 2번째 칸을 펼친다 (W의 2열과 곱함): $0 \times (-1) + 2 \times 1$
6. 계산한다: 0 + 2 = 2 이므로 $g_{in} = (2, 2)$
7. [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 대입한다: $\begin{pmatrix} 0 \\ 2 \end{pmatrix}(1,\ 3)$
8. 1행을 계산한다: 0 × 1 = 0, 0 × 3 = 0 이므로 1행 (0, 0)
9. 2행을 계산한다: 2 × 1 = 2, 2 × 3 = 6 이므로 2행 (2, 6)

**답**: $g_{in} = (2, 2)$, $\frac{\partial J}{\partial W} = \begin{pmatrix} 0 & 0 \\ 2 & 6 \end{pmatrix}$. 꺼진 유닛(1행)의 [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)는 0이다.

</details>

### 문제 4
$W = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \end{pmatrix}$, $x = (1, 2, 3)$, 출력 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $g_{out} = (2, -1)$인 [선형층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#linear-layer-%EC%84%A0%ED%98%95%EC%B8%B5) $y = Wx$에서 $y$, $g_{in}$, $\partial J/\partial W$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $W = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \end{pmatrix}$ (2×3), $x = (1, 2, 3)$ (입력, 3칸)
- $g_{out} = (2, -1)$ (출력 쪽 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0), 2칸)

**풀이**
1. 식을 그대로 쓴다: $y = Wx,\ g_{in} = g_{out}W,\ \frac{\partial J}{\partial W} = g_{out}^\top x^\top$
2. $y$ 1번째 칸을 펼친다: $1 \times 1 + 0 \times 2 + 2 \times 3 = 1 + 0 + 6 = 7$
3. $y$ 2번째 칸을 펼친다: $0 \times 1 + 1 \times 2 + 1 \times 3 = 0 + 2 + 3 = 5$
4. $g_{in}$ 값을 대입한다: $(2,\ -1)\begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \end{pmatrix}$
5. 1번째 칸: 2×1 + (−1)×0 = 2 + 0 = 2
6. 2번째 칸: 2×0 + (−1)×1 = 0 − 1 = −1
7. 3번째 칸: 2×2 + (−1)×1 = 4 − 1 = 3
8. $\partial J/\partial W$ 1행: $g_1 = 2$를 $(1, 2, 3)$에 곱한다: (2, 4, 6)
9. $\partial J/\partial W$ 2행: $g_2 = -1$을 $(1, 2, 3)$에 곱한다: (−1, −2, −3)

**답**: $y = (7, 5)$, $g_{in} = (2, -1, 3)$ (x와 같은 3칸), $\frac{\partial J}{\partial W} = \begin{pmatrix} 2 & 4 & 6 \\ -1 & -2 & -3 \end{pmatrix}$ (W와 같은 2×3 모양)

</details>

## 응용

### 문제 5
문제 4의 [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)로 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $\eta = 0.5$를 써서 W를 한 번 갱신하라. $W \leftarrow W - \eta\,\partial J/\partial W$

<details><summary>풀이와 답</summary>

**주어진 값**
- $W = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \end{pmatrix}$
- $\frac{\partial J}{\partial W} = \begin{pmatrix} 2 & 4 & 6 \\ -1 & -2 & -3 \end{pmatrix}$ (문제 4)
- $\eta = 0.5$ ([학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))

**풀이**
1. 식을 그대로 쓴다: $W \leftarrow W - \eta\,\frac{\partial J}{\partial W}$
2. 값을 대입한다: $W \leftarrow \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \end{pmatrix} - 0.5 \times \begin{pmatrix} 2 & 4 & 6 \\ -1 & -2 & -3 \end{pmatrix}$
3. [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)을 칸마다 곱한다: 1행 0.5×(2, 4, 6) = (1, 2, 3), 2행 0.5×(−1, −2, −3) = (−0.5, −1, −1.5)
4. 1행을 뺀다: 1 − 1 = 0, 0 − 2 = −2, 2 − 3 = −1
5. 2행을 뺀다: 0 − (−0.5) = 0.5, 1 − (−1) = 2, 1 − (−1.5) = 2.5

**답**: 새 $W = \begin{pmatrix} 0 & -2 & -1 \\ 0.5 & 2 & 2.5 \end{pmatrix}$

</details>

### 문제 6
2층 MLP의 [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C). $x = (1, 2)$, $W_1 = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$, $W_2 = (1,\ 2)$, [정답](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5) $t = 0$, [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98) $J = \frac{1}{2}(y - t)^2$일 때 $z = W_1x$, $h = \mathrm{ReLU}(z)$, $y = W_2h$, $J$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $x = (1, 2)$, $W_1 = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$, $W_2 = (1,\ 2)$
- $t = 0$ ([정답](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5))

**풀이**
1. 식을 그대로 쓴다: $z = W_1x,\ h = \mathrm{ReLU}(z),\ y = W_2h,\ J = \frac{1}{2}(y - t)^2$
2. $z$ 1번째 칸을 펼친다: $1 \times 1 + 1 \times 2 = 1 + 2 = 3$
3. $z$ 2번째 칸을 펼친다: $1 \times 1 + (-1) \times 2 = 1 - 2 = -1$ 이므로 $z = (3, -1)$
4. [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)를 적용한다: 3은 그대로, −1은 0 → $h = (3, 0)$
5. $y$를 대입한다: $y = 1 \times 3 + 2 \times 0$
6. 계산한다: 3 + 0 = 3 이므로 $y = 3$
7. $J$를 대입한다: $J = \frac{1}{2}(3 - 0)^2$
8. 제곱을 한다: $3^2 = 9$ 이므로 $J = \frac{1}{2} \times 9$
9. 계산한다: 9 ÷ 2 = 4.5

**답**: $z = (3, -1)$, $h = (3, 0)$, $y = 3$, $J = 4.5$

</details>

### 문제 7
문제 6의 2층 MLP를 역전파하여 $g_y$, $\partial J/\partial W_2$, $g_h$, $g_z$, $\partial J/\partial W_1$, $g_x$를 구하라.

<details><summary>풀이와 답</summary>

위에서 아래로 한 칸씩 내려간다. [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C) 값(z, h)이 필요하다.

**주어진 값**
- $x = (1, 2)$, $W_1 = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$, $W_2 = (1,\ 2)$, $t = 0$
- $z = (3, -1)$, $h = (3, 0)$, $y = 3$ (문제 6)

**풀이**
1. 식을 그대로 쓴다: $g_y = y - t$ ($J = \frac{1}{2}(y - t)^2$을 y로 [미분](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#derivative-%EB%AF%B8%EB%B6%84))
2. 값을 대입한다: $g_y = 3 - 0 = 3$
3. $W_2$ [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 식: $\frac{\partial J}{\partial W_2} = g_y \times h$
4. 값을 대입한다: $3 \times (3, 0) = (9, 0)$
5. $g_h$ 식: $g_h = g_y\,W_2$
6. 값을 대입한다: $3 \times (1, 2) = (3, 6)$
7. [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)를 만든다: $z = (3, -1)$이므로 $\mathbb{1}[z > 0] = (1, 0)$
8. $g_z$를 구한다: $(3, 6) \odot (1, 0) = (3 \times 1,\ 6 \times 0) = (3, 0)$
9. $W_1$ [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 식: $\frac{\partial J}{\partial W_1} = g_z^\top x^\top$
10. 값을 대입한다: $\begin{pmatrix} 3 \\ 0 \end{pmatrix}(1,\ 2)$
11. 1행을 계산한다: 3 × 1 = 3, 3 × 2 = 6 이므로 (3, 6)
12. 2행을 계산한다: 0 × 1 = 0, 0 × 2 = 0 이므로 (0, 0)
13. $g_x$ 식: $g_x = g_z\,W_1$
14. 값을 대입한다: $(3,\ 0)\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$
15. 1번째 칸: 3×1 + 0×1 = 3
16. 2번째 칸: 3×1 + 0×(−1) = 3

**답**: $g_y = 3$, $\frac{\partial J}{\partial W_2} = (9, 0)$, $g_h = (3, 6)$, $g_z = (3, 0)$, $\frac{\partial J}{\partial W_1} = \begin{pmatrix} 3 & 6 \\ 0 & 0 \end{pmatrix}$, $g_x = (3, 3)$. $z_2$가 음수라 $W_1$의 2행과 $W_2$의 둘째 칸 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)는 0이다.

</details>

### 문제 8
$x = 3$이 두 갈래 $u = 2x$, $v = x^2$로 쓰이고 $J = u \times v$일 때, [가지치기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pruning-%EA%B0%80%EC%A7%80%EC%B9%98%EA%B8%B0) 규칙(가지별 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 더한다)으로 $\partial J/\partial x$를 구하고 직접 [미분](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#derivative-%EB%AF%B8%EB%B6%84)과 비교하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $x = 3$ (입력)
- $u = 2x,\ v = x^2$ (두 갈래), $J = uv$
- $\frac{\partial J}{\partial u} = v,\ \frac{\partial J}{\partial v} = u$ (곱의 [미분](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#derivative-%EB%AF%B8%EB%B6%84))
- $\frac{\partial u}{\partial x} = 2,\ \frac{\partial v}{\partial x} = 2x$

**풀이**
1. 식을 그대로 쓴다: $\frac{\partial J}{\partial x} = \frac{\partial J}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial J}{\partial v}\frac{\partial v}{\partial x}$
2. 값을 구한다: $u = 2 \times 3 = 6$, $v = 3^2 = 9$
3. 값을 대입한다: $\frac{\partial J}{\partial x} = 9 \times 2 + 6 \times (2 \times 3)$
4. 가지 u를 계산한다: 9 × 2 = 18
5. 가지 v를 계산한다: 2 × 3 = 6, 6 × 6 = 36
6. 더한다: 18 + 36 = 54
7. 검산: $J = 2x \cdot x^2 = 2x^3$ 이므로 $\frac{dJ}{dx} = 6x^2 = 6 \times 3^2 = 6 \times 9 = 54$

**답**: $\partial J/\partial x = 54$ (직접 [미분](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#derivative-%EB%AF%B8%EB%B6%84)과 같다)

</details>

## 강의와 연결

### 문제 9
[Lec02 8.2](../../Lec02_%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%95%99%EC%8A%B5.md#82-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-%EA%B3%B5%EC%9C%A0)의 [파라미터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 공유. 같은 [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) $w = 2$를 두 번 써서 $y_1 = w x_1$, $y_2 = w x_2$이고 $x = (3, 4)$, $J = y_1 + y_2$일 때 $y$와 $\partial J/\partial w$를 구하라.

<details><summary>풀이와 답</summary>

같은 [파라미터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0)가 두 갈래에서 쓰이면 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)는 갈래마다 구해 **더한다**.

**주어진 값**
- $w = 2$ (공유되는 [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98))
- $x_1 = 3,\ x_2 = 4$ (입력)
- $J = y_1 + y_2$

**풀이**
1. 식을 그대로 쓴다: $y_1 = w x_1,\ y_2 = w x_2$
2. 값을 대입한다: $y_1 = 2 \times 3,\ y_2 = 2 \times 4$
3. 계산한다: $y_1 = 6,\ y_2 = 8$
4. 위쪽 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 구한다: $J = y_1 + y_2$ 이므로 $\frac{\partial J}{\partial y_1} = 1,\ \frac{\partial J}{\partial y_2} = 1$
5. 갈래별 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 쓴다: $\frac{\partial J}{\partial y_i}\frac{\partial y_i}{\partial w} = 1 \times x_i$
6. 값을 대입한다: 갈래 1은 $1 \times 3 = 3$, 갈래 2는 $1 \times 4 = 4$
7. 가지를 더한다: $\frac{\partial J}{\partial w} = 3 + 4 = 7$
8. 검산: $J = wx_1 + wx_2 = w(x_1 + x_2) = w \times 7$ 이므로 $\frac{dJ}{dw} = 7$

**답**: $y = (6, 8)$, $\partial J/\partial w = 7$. 공유 [파라미터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0)의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)는 쓰인 횟수만큼 더해진다.

</details>

### 문제 10
[Lec02 7.2](../../Lec02_%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%95%99%EC%8A%B5.md#72-mlp%EC%99%80-relu-%EA%B2%8C%EC%9D%B4%ED%8A%B8)의 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8). [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C) 값이 $z = (-2, 0, 3)$이고 위에서 $g_h = (4, -5, 6)$이 내려왔을 때 $g_z$를 구하라. [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)는 [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C)의 $z$로 정하고($z > 0$이면 1, 아니면 0), [역전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#backpropagation-%EC%97%AD%EC%A0%84%ED%8C%8C) 중에 들어오는 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)의 부호와는 상관없다.

<details><summary>풀이와 답</summary>

**주어진 값**
- $z = (-2, 0, 3)$ ([순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C)에서 저장해 둔 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 입력)
- $g_h = (4, -5, 6)$ (내려온 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))

**풀이**
1. 식을 그대로 쓴다: $g_z = g_h \odot \mathbb{1}[z > 0]$
2. [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)를 만든다: −2는 양수가 아니라 0, 0도 양수가 아니라 0 ([ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)는 0에서 [미분](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#derivative-%EB%AF%B8%EB%B6%84)이 정해지지 않아 [PyTorch](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pytorch-%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98)는 0으로 둔다), 3은 양수라 1 이므로 $\mathbb{1}[z > 0] = (0, 0, 1)$
3. 값을 대입한다: $g_z = (4, -5, 6) \odot (0, 0, 1)$
4. 같은 자리끼리 곱한다: 4 × 0 = 0, (−5) × 0 = 0, 6 × 1 = 6

**답**: $g_z = (0, 0, 6)$. 둘째 칸의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) −5는 부호와 상관없이 [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C)에서 꺼져 있었기 때문에 막힌다. 그래서 [역전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#backpropagation-%EC%97%AD%EC%A0%84%ED%8C%8C)에는 [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C)의 $z$를 저장해 두어야 한다.

</details>
