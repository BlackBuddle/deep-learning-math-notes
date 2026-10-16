---
tags: [MIT딥러닝, 예제]
cssclasses: [dl-note]
---

# 예제: PCA와 선형 오토인코더

개념 노트: [MM40_PCA와_선형_오토인코더](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/10_%ED%91%9C%ED%98%84_%ED%95%99%EC%8A%B5/MM40_PCA%EC%99%80_%EC%84%A0%ED%98%95_%EC%98%A4%ED%86%A0%EC%9D%B8%EC%BD%94%EB%8D%94.md) · [목차](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md)

풀이는 접혀 있으니 먼저 풀어 보고 펼쳐 확인한다. 쓰는 규칙은 셋이다: 길이 1인 방향 $\mathbf{w}$에 점 $\mathbf{x}$를 투영하면 [투영](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#projection-%ED%88%AC%EC%98%81) 길이는 $p = \mathbf{w}^\top\mathbf{x}$, 복원은 $\hat{\mathbf{x}} = p\,\mathbf{w}$, 복원 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)는 $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$이다. 피타고라스 식 $\|\mathbf{x}\|^2 = p^2 + \text{오차}$가 늘 성립한다. [공분산 행렬](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#covariance-matrix-%EA%B3%B5%EB%B6%84%EC%82%B0-%ED%96%89%EB%A0%AC)은 $\Sigma = \frac1N\sum_i \mathbf{x}_i\mathbf{x}_i^\top$(평균이 0인 데이터)이고, $\Sigma\mathbf{w} = \lambda\mathbf{w}$이면 $\mathbf{w}$는 [고유벡터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92), $\lambda$는 [고윳값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92)이다.

## 기본 (직접 대입)

### 문제 1

점 $\mathbf{x} = (2, 6)$을 길이 1인 방향 $\mathbf{w} = (0.6, 0.8)$에 투영할 때 [투영](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#projection-%ED%88%AC%EC%98%81) 길이, 복원, 복원 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mathbf{x} = (2, 6)$ (데이터 점)
- $\mathbf{w} = (0.6, 0.8)$ (방향. $0.6^2 + 0.8^2 = 0.36 + 0.64 = 1$ 이라 길이 1)

**풀이**
1. 식을 그대로 쓴다: $p = \mathbf{w}^\top\mathbf{x}$, $\hat{\mathbf{x}} = p\,\mathbf{w}$, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) $= \|\mathbf{x} - \hat{\mathbf{x}}\|^2$
2. [투영](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#projection-%ED%88%AC%EC%98%81) 길이에 값을 대입한다: $p = 0.6 \times 2 + 0.8 \times 6$
3. 곱셈을 한다: 0.6 × 2 = 1.2, 0.8 × 6 = 4.8 이므로 $p = 1.2 + 4.8$
4. 더한다: $p = 6$
5. 복원에 값을 대입한다: $\hat{\mathbf{x}} = 6 \times (0.6, 0.8)$
6. 칸마다 곱한다: 6 × 0.6 = 3.6, 6 × 0.8 = 4.8 이므로 $\hat{\mathbf{x}} = (3.6, 4.8)$
7. [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)에 값을 대입한다: $\|(2, 6) - (3.6, 4.8)\|^2$
8. 칸별로 뺀다: 2 − 3.6 = −1.6, 6 − 4.8 = 1.2 이므로 $\|(-1.6, 1.2)\|^2$
9. 제곱해서 더한다: $(-1.6)^2 + 1.2^2 = 2.56 + 1.44 = 4$

**답**: $p = 6$, $\hat{\mathbf{x}} = (3.6, 4.8)$, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) $= 4$. 확인: $\|\mathbf{x}\|^2 = 2^2 + 6^2 = 40 = 6^2 + 4 = 36 + 4$.

</details>

### 문제 2

점 $\mathbf{x} = (5, 0)$을 방향 $\mathbf{w} = (0.8, 0.6)$에 투영할 때 복원 $\hat{\mathbf{x}}$와 복원 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)를 구하고, 피타고라스 식이 맞는지 확인하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mathbf{x} = (5, 0)$ (데이터 점)
- $\mathbf{w} = (0.8, 0.6)$ (방향. $0.64 + 0.36 = 1$ 이라 길이 1)

**풀이**
1. 식을 그대로 쓴다: $p = \mathbf{w}^\top\mathbf{x}$, $\hat{\mathbf{x}} = p\,\mathbf{w}$, $\|\mathbf{x}\|^2 = p^2 + \|\mathbf{x} - \hat{\mathbf{x}}\|^2$
2. [투영](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#projection-%ED%88%AC%EC%98%81) 길이에 값을 대입한다: $p = 0.8 \times 5 + 0.6 \times 0$
3. 곱셈을 하고 더한다: 0.8 × 5 = 4, 0.6 × 0 = 0 이므로 $p = 4 + 0 = 4$
4. 복원에 값을 대입한다: $\hat{\mathbf{x}} = 4 \times (0.8, 0.6)$
5. 칸마다 곱한다: 4 × 0.8 = 3.2, 4 × 0.6 = 2.4 이므로 $\hat{\mathbf{x}} = (3.2, 2.4)$
6. [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)에 값을 대입한다: $\|(5, 0) - (3.2, 2.4)\|^2$
7. 칸별로 뺀다: 5 − 3.2 = 1.8, 0 − 2.4 = −2.4 이므로 $\|(1.8, -2.4)\|^2$
8. 제곱해서 더한다: $1.8^2 + (-2.4)^2 = 3.24 + 5.76 = 9$
9. 피타고라스 식에 대입해 확인한다: $5^2 + 0^2 = 4^2 + 9$ 이므로 $25 = 16 + 9$

**답**: $\hat{\mathbf{x}} = (3.2, 2.4)$, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) $= 9$. 담은 부분 16 + [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) 9 = 25 = $\|\mathbf{x}\|^2$.

</details>

### 문제 3

점 $\mathbf{x} = (1, 3)$을 가로축 방향 $\mathbf{w} = (1, 0)$에 투영하면 담은 부분과 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)는 얼마인가?

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mathbf{x} = (1, 3)$ (데이터 점)
- $\mathbf{w} = (1, 0)$ (가로축 방향, 길이 1)

**풀이**
1. 식을 그대로 쓴다: $p = \mathbf{w}^\top\mathbf{x}$, $\hat{\mathbf{x}} = p\,\mathbf{w}$, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) $= \|\mathbf{x} - \hat{\mathbf{x}}\|^2$
2. [투영](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#projection-%ED%88%AC%EC%98%81) 길이에 값을 대입한다: $p = 1 \times 1 + 0 \times 3$
3. 곱셈을 하고 더한다: 1 × 1 = 1, 0 × 3 = 0 이므로 $p = 1$
4. 복원에 값을 대입한다: $\hat{\mathbf{x}} = 1 \times (1, 0) = (1, 0)$
5. [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)에 값을 대입한다: $\|(1, 3) - (1, 0)\|^2$
6. 칸별로 뺀다: 1 − 1 = 0, 3 − 0 = 3 이므로 $\|(0, 3)\|^2$
7. 제곱해서 더한다: $0^2 + 3^2 = 9$
8. 담은 부분은 $p^2$이다: $1^2 = 1$

**답**: 담은 부분 **1**, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) **9**. 합은 $1 + 9 = 10 = \|\mathbf{x}\|^2$. 가로축은 세로로 긴 점을 거의 못 담는다.

</details>

### 문제 4

평균이 0인 점 $(2, 0)$, $(-2, 0)$, $(0, 1)$, $(0, -1)$의 [공분산 행렬](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#covariance-matrix-%EA%B3%B5%EB%B6%84%EC%82%B0-%ED%96%89%EB%A0%AC) $\Sigma$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- 점 $(2, 0)$, $(-2, 0)$, $(0, 1)$, $(0, -1)$ (평균이 0인 데이터)
- $N = 4$ (점의 수)

**풀이**
1. 식을 그대로 쓴다: $\Sigma = \frac{1}{N}\sum_{i=1}^{N} \mathbf{x}_i\mathbf{x}_i^\top$
2. 값을 대입한다: $\Sigma = \frac14\big[(2,0)(2,0)^\top + (-2,0)(-2,0)^\top + (0,1)(0,1)^\top + (0,-1)(0,-1)^\top\big]$
3. 바깥곱은 칸끼리 곱한 2×2 표다. $(2,0)$: $\begin{pmatrix}2\times2&2\times0\\0\times2&0\times0\end{pmatrix} = \begin{pmatrix}4&0\\0&0\end{pmatrix}$, $(-2,0)$도 같다
4. $(0,1)$: $\begin{pmatrix}0\times0&0\times1\\1\times0&1\times1\end{pmatrix} = \begin{pmatrix}0&0\\0&1\end{pmatrix}$, $(0,-1)$도 같다
5. 칸별로 더한다: $\begin{pmatrix}4+4+0+0&0\\0&0+0+1+1\end{pmatrix} = \begin{pmatrix}8&0\\0&2\end{pmatrix}$
6. 4로 나눈다: $\Sigma = \begin{pmatrix}2&0\\0&0.5\end{pmatrix}$

**답**: $\Sigma = \begin{pmatrix}2&0\\0&0.5\end{pmatrix}$. 가로 방향 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0) 2, 세로 방향 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0) 0.5.

</details>

## 응용

### 문제 5

점 $(3, 0)$, $(-3, 0)$, $(0, 1)$, $(0, -1)$을 방향 $0°$ ($\mathbf{w} = (1, 0)$), $45°$ ($\mathbf{w} = (0.7071, 0.7071)$), $90°$ ($\mathbf{w} = (0, 1)$)로 투영할 때 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0) $\sum p^2$과 복원 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)를 구하고, 가장 좋은 방향을 고르라.

<details><summary>풀이와 답</summary>

**주어진 값**
- 점 $(3, 0)$, $(-3, 0)$, $(0, 1)$, $(0, -1)$ (평균이 0인 데이터)
- 방향 세 가지 (모두 길이 1)

**풀이**
1. 전체 제곱합을 구한다: $\sum\|\mathbf{x}_i\|^2 = (9 + 0) + (9 + 0) + (0 + 1) + (0 + 1) = 20$
2. 식을 그대로 쓴다: $p = \mathbf{w}^\top\mathbf{x}$, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) $= 20 - \sum p^2$
3. $0°$의 [투영](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#projection-%ED%88%AC%EC%98%81) 길이: $(1,0)$과 네 점의 [내적](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#dot-product-%EB%82%B4%EC%A0%81)은 $1\times3 + 0\times0 = 3$, $-3$, $0$, $0$
4. $0°$의 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0): $3^2 + (-3)^2 + 0^2 + 0^2 = 9 + 9 + 0 + 0 = 18$, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)는 $20 - 18 = 2$
5. $45°$의 [투영](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#projection-%ED%88%AC%EC%98%81) 길이: $0.7071 \times 3 + 0.7071 \times 0 = 2.1213$, $-2.1213$, $0.7071 \times 0 + 0.7071 \times 1 = 0.7071$, $-0.7071$
6. $45°$의 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0): $2.1213^2 + 2.1213^2 + 0.7071^2 + 0.7071^2 = 4.5 + 4.5 + 0.5 + 0.5 = 10$, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)는 $20 - 10 = 10$
7. $90°$의 [투영](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#projection-%ED%88%AC%EC%98%81) 길이: $(0,1)$과 네 점의 [내적](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#dot-product-%EB%82%B4%EC%A0%81)은 $0$, $0$, $1$, $-1$
8. $90°$의 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0): $0 + 0 + 1 + 1 = 2$, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)는 $20 - 2 = 18$
9. 비교한다: 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0)이 가장 큰 쪽(18)이 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)가 가장 작다(2)

**답**: $0°$ (담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0) **18**, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) **2**), $45°$ (10, 10), $90°$ (2, 18). 가장 좋은 방향은 **$0°$**. 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0)과 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)의 합은 늘 20이다.

</details>

### 문제 6

평균이 0인 점 $(4, 4)$, $(-4, -4)$, $(1, -1)$, $(-1, 1)$이 있다. [공분산 행렬](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#covariance-matrix-%EA%B3%B5%EB%B6%84%EC%82%B0-%ED%96%89%EB%A0%AC) $\Sigma$를 구하고, $\mathbf{w} = (1, 1)$이 고유벡터인지 확인한 뒤 [고윳값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92)을 구하라. 한 방향만 남길 때 최소 복원 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)는 얼마인가? (다른 [고유벡터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92)는 $(1, -1)$이고 [고윳값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92)은 1이다.)

<details><summary>풀이와 답</summary>

**주어진 값**
- 점 $(4,4)$, $(-4,-4)$, $(1,-1)$, $(-1,1)$ (평균이 0인 데이터, $N = 4$)
- 확인할 방향 $\mathbf{w} = (1, 1)$
- 다른 [고유벡터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92) $(1, -1)$의 [고윳값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92) 1

**풀이**
1. 식을 그대로 쓴다: $\Sigma = \frac{1}{N}\sum_i \mathbf{x}_i\mathbf{x}_i^\top$, $\Sigma\mathbf{w} = \lambda\mathbf{w}$
2. 값을 대입한다: $\Sigma = \frac14\big[(4,4)(4,4)^\top + (-4,-4)(-4,-4)^\top + (1,-1)(1,-1)^\top + (-1,1)(-1,1)^\top\big]$
3. 바깥곱을 만든다: $(4,4)$와 $(-4,-4)$는 $\begin{pmatrix}16&16\\16&16\end{pmatrix}$, $(1,-1)$과 $(-1,1)$은 $\begin{pmatrix}1&-1\\-1&1\end{pmatrix}$
4. 칸별로 더한다: $\begin{pmatrix}16+16+1+1&16+16-1-1\\16+16-1-1&16+16+1+1\end{pmatrix} = \begin{pmatrix}34&30\\30&34\end{pmatrix}$
5. 4로 나눈다: $\Sigma = \begin{pmatrix}8.5&7.5\\7.5&8.5\end{pmatrix}$
6. $\Sigma\mathbf{w}$에 대입한다: $\begin{pmatrix}8.5&7.5\\7.5&8.5\end{pmatrix}\begin{pmatrix}1\\1\end{pmatrix}$
7. 행마다 펼친다: $(8.5\times1 + 7.5\times1,\ 7.5\times1 + 8.5\times1) = (16, 16)$
8. 배율을 구한다: $(16, 16) = 16 \times (1, 1)$ 이므로 $\Sigma\mathbf{w} = 16\,\mathbf{w}$ ([고유벡터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92)가 맞고 $\lambda_1 = 16$)
9. 전체 제곱합을 구한다: $(16+16) + (16+16) + (1+1) + (1+1) = 68$
10. 한 방향([고윳값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92) 16)이 담는 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0)은 $N\lambda_1$이다: $4 \times 16 = 64$
11. 최소 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)는 전체에서 뺀 값이다: $68 - 64 = 4$

**답**: $\Sigma = \begin{pmatrix}8.5&7.5\\7.5&8.5\end{pmatrix}$, $\lambda_1 = 16$, 최소 복원 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) **4** ($= N\lambda_2 = 4 \times 1$).

</details>

### 문제 7

$\Sigma = \begin{pmatrix}2&1\\1&2\end{pmatrix}$일 때 $(1, 0)$, $(1, 1)$, $(1, -1)$ 중 어느 것이 고유벡터인가? 고유벡터라면 [고윳값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92)도 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\Sigma = \begin{pmatrix}2&1\\1&2\end{pmatrix}$ ([공분산 행렬](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#covariance-matrix-%EA%B3%B5%EB%B6%84%EC%82%B0-%ED%96%89%EB%A0%AC))
- 후보 $(1, 0)$, $(1, 1)$, $(1, -1)$

**풀이**
1. 식을 그대로 쓴다: $\Sigma\mathbf{w} = \lambda\mathbf{w}$ (곱한 결과가 $\mathbf{w}$의 배수여야 [고유벡터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92))
2. $(1, 0)$을 대입한다: $\begin{pmatrix}2&1\\1&2\end{pmatrix}\begin{pmatrix}1\\0\end{pmatrix} = (2\times1 + 1\times0,\ 1\times1 + 2\times0) = (2, 1)$
3. 판정한다: $(2, 1)$은 $(1, 0)$의 배수가 아니다 (둘째 칸이 0이 아니다) → [고유벡터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92) 아님
4. $(1, 1)$을 대입한다: $(2\times1 + 1\times1,\ 1\times1 + 2\times1) = (3, 3)$
5. 판정한다: $(3, 3) = 3 \times (1, 1)$ 이므로 [고유벡터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92), $\lambda = 3$
6. $(1, -1)$을 대입한다: $(2\times1 + 1\times(-1),\ 1\times1 + 2\times(-1)) = (1, -1)$
7. 판정한다: $(1, -1) = 1 \times (1, -1)$ 이므로 [고유벡터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92), $\lambda = 1$

**답**: $(1, 1)$ ([고윳값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92) **3**)과 $(1, -1)$ ([고윳값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92) **1**)이 [고유벡터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92)이고, $(1, 0)$은 아니다. 주성분은 [고윳값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92)이 가장 큰 $(1, 1)$ 방향이다.

</details>

### 문제 8

점이 $N = 5$개이고 [공분산 행렬](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#covariance-matrix-%EA%B3%B5%EB%B6%84%EC%82%B0-%ED%96%89%EB%A0%AC)의 [고윳값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92)이 $\lambda_1 = 6$, $\lambda_2 = 2$다. 전체 제곱합, 주성분 하나가 담는 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0), 최소 복원 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8), 담은 비율을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $N = 5$ (점의 수)
- $\lambda_1 = 6$, $\lambda_2 = 2$ ([고윳값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#eigenvector-eigenvalue-%EA%B3%A0%EC%9C%A0%EB%B2%A1%ED%84%B0%EC%99%80-%EA%B3%A0%EC%9C%B3%EA%B0%92), 방향별 점 하나당 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0))

**풀이**
1. 식을 그대로 쓴다: 전체 $= N(\lambda_1 + \lambda_2)$, 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0) $= N\lambda_1$, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) $=$ 전체 $-$ 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0) $= N\lambda_2$
2. 전체에 값을 대입한다: $5 \times (6 + 2)$
3. 계산한다: 6 + 2 = 8, 5 × 8 = 40
4. 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0)에 값을 대입한다: $5 \times 6 = 30$
5. [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)를 구한다: $40 - 30 = 10$ (또한 $5 \times 2 = 10$)
6. 담은 비율은 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0)을 전체로 나눈 값이다: $30 / 40 = 0.75$

**답**: 전체 **40**, 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0) **30**, 최소 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) **10**, 담은 비율 **0.75**(75%).

</details>

## 강의와 연결

### 문제 9

Lec11 8.2는 $\min\|X - XWW^\top\|^2 = \max\|XW\|^2$이라고 한다([오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) 최소와 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0) 최대가 같다). 평균이 0인 점 $(2, 0)$, $(0, 2)$, $(-2, -2)$를 방향 $\mathbf{w}_a = (0.7071, 0.7071)$과 $\mathbf{w}_b = (1, 0)$으로 투영해, 두 방향의 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0)과 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)를 구하고 어느 쪽이 주성분에 가까운지 말하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- 점 $(2, 0)$, $(0, 2)$, $(-2, -2)$ (평균이 0인 데이터)
- $\mathbf{w}_a = (0.7071, 0.7071)$, $\mathbf{w}_b = (1, 0)$ (비교할 방향, 길이 1)

**풀이**
1. 전체 제곱합을 구한다: $(4 + 0) + (0 + 4) + (4 + 4) = 16$
2. 식을 그대로 쓴다: $p = \mathbf{w}^\top\mathbf{x}$, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) $= \|X\|^2 - \sum p^2$
3. $\mathbf{w}_a$의 [투영](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#projection-%ED%88%AC%EC%98%81) 길이: $0.7071\times2 + 0.7071\times0 = 1.4142$, $0.7071\times0 + 0.7071\times2 = 1.4142$, $0.7071\times(-2) + 0.7071\times(-2) = -2.8284$
4. $\mathbf{w}_a$의 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0): $1.4142^2 + 1.4142^2 + (-2.8284)^2 = 2 + 2 + 8 = 12$
5. $\mathbf{w}_a$의 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8): $16 - 12 = 4$
6. $\mathbf{w}_b$의 [투영](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#projection-%ED%88%AC%EC%98%81) 길이: $1\times2 + 0\times0 = 2$, $1\times0 + 0\times2 = 0$, $1\times(-2) + 0\times(-2) = -2$
7. $\mathbf{w}_b$의 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0): $2^2 + 0^2 + (-2)^2 = 4 + 0 + 4 = 8$
8. $\mathbf{w}_b$의 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8): $16 - 8 = 8$
9. 비교한다: 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0) 12 > 8 이고, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) 4 < 8 이다

**답**: $\mathbf{w}_a$ (담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0) **12**, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) **4**), $\mathbf{w}_b$ (**8**, **8**). 담은 [분산](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#variance-%EB%B6%84%EC%82%B0)이 큰 $\mathbf{w}_a$가 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)도 작고 주성분에 가깝다. 두 값의 합은 늘 16이다.

</details>

### 문제 10

Lec11 8.2의 선형 [오토인코더](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#autoencoder-%EC%98%A4%ED%86%A0%EC%9D%B8%EC%BD%94%EB%8D%94)에서 인코더 $W_f = (0.5,\ 0.5)$ (1×2), 디코더 $W_g = \begin{pmatrix}1\\1\end{pmatrix}$ (2×1)이다. 점 $\mathbf{x} = (3, 1)$을 넣었을 때 코드 $z = W_f\mathbf{x}$, 복원 $\hat{\mathbf{x}} = W_g z$, 복원 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)를 구하라. 같은 점을 $\mathbf{w} = (1, 1)/\sqrt{2}$로 투영한 [PCA](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pca-principal-component-analysis-%EC%A3%BC%EC%84%B1%EB%B6%84-%EB%B6%84%EC%84%9D) 결과와 비교하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $W_f = (0.5, 0.5)$ (인코더, 2차원을 1차원으로)
- $W_g = \begin{pmatrix}1\\1\end{pmatrix}$ (디코더, 1차원을 2차원으로)
- $\mathbf{x} = (3, 1)$ (입력 점)

**풀이**
1. 식을 그대로 쓴다: $z = W_f\mathbf{x}$, $\hat{\mathbf{x}} = W_g z$, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) $= \|\mathbf{x} - \hat{\mathbf{x}}\|^2$
2. 코드에 값을 대입한다: $z = (0.5, 0.5)\begin{pmatrix}3\\1\end{pmatrix} = 0.5 \times 3 + 0.5 \times 1$
3. 곱셈을 하고 더한다: 0.5 × 3 = 1.5, 0.5 × 1 = 0.5 이므로 $z = 2$
4. 복원에 값을 대입한다: $\hat{\mathbf{x}} = \begin{pmatrix}1\\1\end{pmatrix} \times 2$
5. 칸마다 곱한다: 1 × 2 = 2 (두 칸 모두) 이므로 $\hat{\mathbf{x}} = (2, 2)$
6. [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)에 값을 대입한다: $\|(3, 1) - (2, 2)\|^2$
7. 칸별로 빼고 제곱해 더한다: $(3-2)^2 + (1-2)^2 = 1 + 1 = 2$
8. [PCA](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pca-principal-component-analysis-%EC%A3%BC%EC%84%B1%EB%B6%84-%EB%B6%84%EC%84%9D)와 비교한다. $\sqrt{2}$는 제곱해서 2가 되는 수 1.4142이고 $\mathbf{w} = (0.7071, 0.7071)$이다. [투영](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#projection-%ED%88%AC%EC%98%81) 길이: $0.7071 \times 3 + 0.7071 \times 1 = 2.8284$
9. [PCA](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pca-principal-component-analysis-%EC%A3%BC%EC%84%B1%EB%B6%84-%EB%B6%84%EC%84%9D) 복원: $2.8284 \times (0.7071, 0.7071) = (2, 2)$, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)는 $(3-2)^2 + (1-2)^2 = 2$

**답**: $z = 2$, $\hat{\mathbf{x}} = (2, 2)$, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) **2**. [PCA](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pca-principal-component-analysis-%EC%A3%BC%EC%84%B1%EB%B6%84-%EB%B6%84%EC%84%9D)의 복원 $(2, 2)$, [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) 2와 같다. $W_f$와 $W_g$의 길이가 1이 아니어도 둘이 펼치는 방향(부분공간)이 $(1, 1)$로 [PCA](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pca-principal-component-analysis-%EC%A3%BC%EC%84%B1%EB%B6%84-%EB%B6%84%EC%84%9D)와 같기 때문이다.

</details>
