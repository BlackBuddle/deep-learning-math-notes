---
tags: [MIT딥러닝, 예제]
cssclasses: [dl-note]
---

# 예제: 스펙트럼 노름과 RMS 연산자 노름

개념 노트: [MM28_스펙트럼_노름과_RMS_연산자_노름](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/09_%EC%B5%9C%EC%A0%81%ED%99%94%EC%9D%98_%EA%B8%B0%ED%95%98/MM28_%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC_%EB%85%B8%EB%A6%84%EA%B3%BC_RMS_%EC%97%B0%EC%82%B0%EC%9E%90_%EB%85%B8%EB%A6%84.md) · [목차](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md)

풀이는 접혀 있으니 먼저 풀어 보고 펼쳐 확인한다.

## 기본 (직접 대입)

### 문제 1

행렬 $M = \begin{pmatrix}3&0\\0&4\end{pmatrix}$의 [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84), [프로베니우스 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#frobenius-norm-%ED%94%84%EB%A1%9C%EB%B2%A0%EB%8B%88%EC%9A%B0%EC%8A%A4-%EB%85%B8%EB%A6%84), 핵 [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84)을 구하라. (대각 행렬의 특잇값은 대각 칸의 절댓값이다. [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84) = 가장 큰 특잇값, [프로베니우스 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#frobenius-norm-%ED%94%84%EB%A1%9C%EB%B2%A0%EB%8B%88%EC%9A%B0%EC%8A%A4-%EB%85%B8%EB%A6%84) = 성분 제곱합의 제곱근, 핵 [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84) = 특잇값의 합.)

<details><summary>풀이와 답</summary>

**주어진 값**
- $M = \begin{pmatrix}3&0\\0&4\end{pmatrix}$
- 특잇값 $\sigma_1, \sigma_2 = 4, 3$ (대각 칸 3, 4의 절댓값을 큰 것부터)

**풀이**
1. [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84)의 식을 쓴다: $\lVert M \rVert_\ast = \sigma_{\max}$. 값을 대입한다: $\max(4, 3) = 4$
2. [프로베니우스 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#frobenius-norm-%ED%94%84%EB%A1%9C%EB%B2%A0%EB%8B%88%EC%9A%B0%EC%8A%A4-%EB%85%B8%EB%A6%84)의 식을 쓴다: $\lVert M \rVert_F = \sqrt{\sum M_{ij}^2}$. 값을 대입한다: $\sqrt{3^2 + 0^2 + 0^2 + 4^2}$
3. 제곱한다: $3^2 = 9$, $0^2 = 0$, $4^2 = 16$ 이므로 $\sqrt{9 + 0 + 0 + 16}$
4. 더하고 제곱근을 구한다: $9 + 16 = 25$ 이므로 $\sqrt{25} = 5$ ($5 \times 5 = 25$)
5. 핵 [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84)의 식을 쓴다: $\sum_i \sigma_i$. 값을 대입해 더한다: $4 + 3 = 7$

**답**: [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84) $4$, [프로베니우스 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#frobenius-norm-%ED%94%84%EB%A1%9C%EB%B2%A0%EB%8B%88%EC%9A%B0%EC%8A%A4-%EB%85%B8%EB%A6%84) $5$, 핵 [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84) $7$

</details>

### 문제 2

벡터 $v = (1, 7, 1, 7)$의 [RMS 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rms-norm-rms-%EB%85%B8%EB%A6%84) $\lVert v \rVert_{\mathrm{RMS}} = \lVert v \rVert_2 / \sqrt{d}$를 구하라. ($d$는 벡터의 차원.)

<details><summary>풀이와 답</summary>

**주어진 값**
- $v = (1, 7, 1, 7)$
- $d = 4$ (차원)

**풀이**
1. 식을 그대로 쓴다: $\lVert v \rVert_{\mathrm{RMS}} = \dfrac{\lVert v \rVert_2}{\sqrt{d}}$, $\lVert v \rVert_2 = \sqrt{\sum v_i^2}$
2. 값을 대입한다: $\lVert v \rVert_{\mathrm{RMS}} = \dfrac{\sqrt{1^2 + 7^2 + 1^2 + 7^2}}{\sqrt{4}}$
3. 제곱한다: $1^2 = 1$, $7^2 = 49$ 이므로 분자는 $\sqrt{1 + 49 + 1 + 49}$
4. 더한다: $1 + 49 + 1 + 49 = 100$ 이므로 $\dfrac{\sqrt{100}}{\sqrt{4}}$
5. 제곱근을 구한다: $\sqrt{100} = 10$, $\sqrt{4} = 2$ 이므로 $\dfrac{10}{2}$
6. 나눗셈을 한다: $10 \div 2 = 5$

**답**: $\lVert v \rVert_{\mathrm{RMS}} = 5$ (성분 하나의 평균 크기. 성분이 1, 7, 1, 7이니 그럴듯하다)

</details>

### 문제 3

행렬 $M = \begin{pmatrix}3&0\\0&2\end{pmatrix}$에 길이 1인 입력 $v = (0.6, 0.8)$을 넣었을 때 출력 길이를 구하고, $M$의 [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84)(가장 큰 특잇값 3)과 비교하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $M = \begin{pmatrix}3&0\\0&2\end{pmatrix}$ (특잇값 3, 2)
- $v = (0.6, 0.8)$ (입력, 길이 $\sqrt{0.6^2 + 0.8^2} = \sqrt{0.36 + 0.64} = 1$)

**풀이**
1. 식을 그대로 쓴다: $\lVert Mv \rVert_2 = \sqrt{(Mv)_1^2 + (Mv)_2^2}$
2. 값을 대입한다: $Mv = \begin{pmatrix}3&0\\0&2\end{pmatrix}\begin{pmatrix}0.6\\0.8\end{pmatrix}$
3. [행렬 곱](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#matrix-multiplication-%ED%96%89%EB%A0%AC-%EA%B3%B1)을 줄마다 펼친다: $Mv = \begin{pmatrix}3 \times 0.6 + 0 \times 0.8\\0 \times 0.6 + 2 \times 0.8\end{pmatrix}$
4. 곱셈과 덧셈을 한다: $Mv = \begin{pmatrix}1.8 + 0\\0 + 1.6\end{pmatrix} = \begin{pmatrix}1.8\\1.6\end{pmatrix}$
5. 성분을 제곱한다: $1.8^2 = 3.24$, $1.6^2 = 2.56$
6. 더한다: $3.24 + 2.56 = 5.8$
7. 제곱근을 구한다 (계산기): $\sqrt{5.8} = 2.4083$
8. 비교한다: $2.4083 < 3$

**답**: 출력 길이 $2.4083$, [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84) $3$보다 작다 ([스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84)은 길이 1 입력이 낼 수 있는 최대 출력 길이이고, $v = (1, 0)$을 넣으면 정확히 3이 나온다)

</details>

### 문제 4

입력 차원 $d_\text{in} = 8$, 출력 차원 $d_\text{out} = 2$인 행렬의 [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84)이 $3$이다. RMS→[RMS 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rms-norm-rms-%EB%85%B8%EB%A6%84)을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $d_\text{in} = 8$ (입력 차원)
- $d_\text{out} = 2$ (출력 차원)
- $\lVert W \rVert_\ast = 3$ ([스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84))

**풀이**
1. 식을 그대로 쓴다: $\lVert W \rVert_{\mathrm{RMS}\to\mathrm{RMS}} = \sqrt{\dfrac{d_\text{in}}{d_\text{out}}}\, \lVert W \rVert_\ast$
2. 값을 대입한다: $\sqrt{\dfrac{8}{2}} \times 3$
3. 분수를 계산한다: $8 \div 2 = 4$ 이므로 $\sqrt{4} \times 3$
4. 제곱근을 구한다: $\sqrt{4} = 2$ 이므로 $2 \times 3$
5. 곱한다: $2 \times 3 = 6$

**답**: RMS→[RMS 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rms-norm-rms-%EB%85%B8%EB%A6%84) $= 6$

</details>

## 응용

### 문제 5

행렬 $W = \begin{pmatrix}3&0\\0&4\\0&0\end{pmatrix}$ (입력 2차원, 출력 3차원)의 RMS→[RMS 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rms-norm-rms-%EB%85%B8%EB%A6%84)을 구하라. ($W$의 두 열 $(3, 0, 0)$, $(0, 4, 0)$은 서로 직각이고 길이가 3, 4라 특잇값은 4, 3이다.)

<details><summary>풀이와 답</summary>

**주어진 값**
- $d_\text{in} = 2$ (열 수), $d_\text{out} = 3$ (행 수)
- 특잇값 $4, 3$ 이므로 [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84) $\lVert W \rVert_\ast = 4$

**풀이**
1. 식을 그대로 쓴다: $\lVert W \rVert_{\mathrm{RMS}\to\mathrm{RMS}} = \sqrt{\dfrac{d_\text{in}}{d_\text{out}}}\, \lVert W \rVert_\ast$
2. 값을 대입한다: $\sqrt{\dfrac{2}{3}} \times 4$
3. 분수를 계산한다: $2 \div 3 = 0.6667$ 이므로 $\sqrt{0.6667} \times 4$
4. 제곱근을 구한다 (계산기): $\sqrt{0.6667} = 0.8165$ 이므로 $0.8165 \times 4$
5. 곱한다: $0.8165 \times 4 = 3.2660$

**답**: RMS→[RMS 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rms-norm-rms-%EB%85%B8%EB%A6%84) $= 3.2660$

</details>

### 문제 6

너비 $d = 100$인 [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)을 3개 쌓는다. 각 [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)의 [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 성분을 $N(0, 1)$로 뽑으면 [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) 하나가 입력의 RMS를 대략 $\sqrt{d}$배로 만든다. $N(0, 1/d)$로 뽑으면 대략 1배다. 입력 RMS가 1일 때 3층을 지난 출력 RMS를 각각 어림하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $d = 100$ (너비)
- $N(0, 1)$ 초기화: [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)마다 RMS $\times \sqrt{d}$
- $N(0, 1/d)$ 초기화: [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)마다 RMS $\times 1$
- [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) 수 $3$, 입력 RMS $1$

**풀이**
1. 배율을 구한다: $N(0, 1)$이면 $\sqrt{d} = \sqrt{100} = 10$ ($10 \times 10 = 100$), $N(0, 1/d)$이면 $1$
2. $N(0, 1)$의 식을 쓴다: 출력 RMS $= 1 \times (\sqrt{d})^3$. 값을 대입한다: $1 \times 10^3$
3. 거듭제곱을 푼다 (10을 3번 곱한다): $10 \times 10 \times 10 = 1000$
4. $N(0, 1/d)$의 식을 쓴다: 출력 RMS $= 1 \times 1^3$. 값을 대입해 계산한다: $1 \times 1 \times 1 \times 1 = 1$

**답**: $N(0, 1)$이면 약 $1000$ ([층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)을 쌓을수록 곱으로 폭발), $N(0, 1/d)$이면 약 $1$ (일정)

</details>

### 문제 7

3행 6열 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $G = \begin{pmatrix}1&1&1&1&1&1\\1&1&1&1&1&1\\1&1&1&1&1&1\end{pmatrix}$ ($= u x^\top$, $u = (1, 1, 1)$, $x = (1, 1, 1, 1, 1, 1)$)를 RMS→[RMS 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rms-norm-rms-%EB%85%B8%EB%A6%84)이 1이 되도록 나눈 업데이트 $\Delta W$의 성분 값을 구하라. (길이 $\lVert u \rVert_2$, $\lVert x \rVert_2$인 두 벡터의 곱 $u x^\top$의 [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84)은 $\lVert u \rVert_2 \lVert x \rVert_2$ 이다.)

<details><summary>풀이와 답</summary>

**주어진 값**
- $G$: 3행 6열, 성분이 모두 1 ($d_\text{out} = 3$, $d_\text{in} = 6$)
- $u = (1, 1, 1)$, $x = (1, 1, 1, 1, 1, 1)$

**풀이**
1. [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84)의 식을 쓴다: $\lVert G \rVert_\ast = \lVert u \rVert_2 \lVert x \rVert_2$. 값을 대입한다: $\sqrt{1^2 + 1^2 + 1^2} \times \sqrt{1^2 + 1^2 + 1^2 + 1^2 + 1^2 + 1^2}$
2. 제곱근 안을 더한다: $\sqrt{3} \times \sqrt{6}$
3. 제곱근을 구한다 (계산기): $\sqrt{3} = 1.7321$, $\sqrt{6} = 2.4495$ 이므로 $1.7321 \times 2.4495 = 4.2426$
4. RMS→[RMS 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rms-norm-rms-%EB%85%B8%EB%A6%84)의 식을 쓴다: $\lVert G \rVert_{\mathrm{RMS}\to\mathrm{RMS}} = \sqrt{\dfrac{d_\text{in}}{d_\text{out}}}\, \lVert G \rVert_\ast$. 값을 대입한다: $\sqrt{\dfrac{6}{3}} \times 4.2426$
5. 계산한다: $6 \div 3 = 2$, $\sqrt{2} = 1.4142$ 이므로 $1.4142 \times 4.2426 = 6$
6. [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84)이 1이 되도록 나눈다: $\Delta W = G / 6$, 성분은 $1 \div 6 = 0.1667$
7. 확인한다: $\Delta W$의 RMS→[RMS 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rms-norm-rms-%EB%85%B8%EB%A6%84)은 $6 \div 6 = 1$

**답**: $\Delta W$의 성분은 모두 $\frac16 = 0.1667$ ($G$의 RMS→[RMS 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rms-norm-rms-%EB%85%B8%EB%A6%84)은 $6$이었다)

</details>

### 문제 8

성분이 모두 1인 $d \times d$ [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 행렬 $G$와 입력 $x = (1, \ldots, 1)$ ($d$개)가 있다. [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $\eta = 0.01$을 그냥 곱한 업데이트 $\Delta W = \eta G$가 만드는 출력 변화 $\Delta W x$의 RMS를 $d = 200$과 $d = 800$에서 구하고 비교하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\eta = 0.01$ ([학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))
- $G$의 성분은 모두 1, $x$의 성분은 모두 1
- $d = 200$, $d = 800$ (너비)

**풀이**
1. 식을 쓴다: $(\Delta W x)_i = \sum_{j=1}^{d} \eta \times 1 \times 1 = \eta\, d$ (성분 하나는 $d$개의 항을 더한 것)
2. $d = 200$을 대입한다: $(\Delta W x)_i = 0.01 \times 200$
3. 곱셈을 한다: $0.01 \times 200 = 2$. 모든 성분이 2이므로 RMS도 $2$
4. $d = 800$을 대입한다: $(\Delta W x)_i = 0.01 \times 800 = 8$. 모든 성분이 8이므로 RMS도 $8$
5. 비교한다: $8 \div 2 = 4$

**답**: $d = 200$이면 $2$, $d = 800$이면 $8$. 너비가 4배가 되면 출력 변화도 4배가 되어 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)을 다시 맞춰야 한다 (업데이트를 $G / d$로 RMS→[RMS 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rms-norm-rms-%EB%85%B8%EB%A6%84) 1에 맞추면 어느 너비에서나 출력 변화가 $1$)

</details>

## 강의와 연결

### 문제 9

[Lec07 7](../../Lec07_%EC%B5%9C%EC%A0%81%ED%99%94%EC%9D%98_%EC%8A%A4%EC%BC%80%EC%9D%BC%EB%A7%81_%EA%B7%9C%EC%B9%99.md#7-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8-%ED%81%AC%EA%B8%B0%EC%99%80-%ED%96%89%EB%A0%AC-%EB%85%B8%EB%A6%84)은 업데이트가 크다 작다고 할 때 '어떤 [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84)으로'인지 물어야 한다고 말한다. 업데이트 $\Delta W = \begin{pmatrix}0.3&0\\0&0.4\end{pmatrix}$의 [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84), [프로베니우스 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#frobenius-norm-%ED%94%84%EB%A1%9C%EB%B2%A0%EB%8B%88%EC%9A%B0%EC%8A%A4-%EB%85%B8%EB%A6%84), 핵 [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84)을 구하고, RMS→[RMS 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rms-norm-rms-%EB%85%B8%EB%A6%84) 1로 맞춘 업데이트를 구하라. ($d_\text{in} = d_\text{out} = 2$)

<details><summary>풀이와 답</summary>

**주어진 값**
- $\Delta W = \begin{pmatrix}0.3&0\\0&0.4\end{pmatrix}$
- 특잇값 $0.4, 0.3$ (대각 칸의 절댓값을 큰 것부터)
- $d_\text{in} = d_\text{out} = 2$

**풀이**
1. [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84): 가장 큰 특잇값이므로 $\max(0.4, 0.3) = 0.4$
2. [프로베니우스 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#frobenius-norm-%ED%94%84%EB%A1%9C%EB%B2%A0%EB%8B%88%EC%9A%B0%EC%8A%A4-%EB%85%B8%EB%A6%84): $\sqrt{0.3^2 + 0 + 0 + 0.4^2} = \sqrt{0.09 + 0.16} = \sqrt{0.25} = 0.5$
3. 핵 [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84): 특잇값의 합 $0.4 + 0.3 = 0.7$
4. RMS→[RMS 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rms-norm-rms-%EB%85%B8%EB%A6%84)의 식을 쓰고 대입한다: $\sqrt{\dfrac{d_\text{in}}{d_\text{out}}}\,\lVert \Delta W \rVert_\ast = \sqrt{\dfrac{2}{2}} \times 0.4$
5. 계산한다: $2 \div 2 = 1$, $\sqrt{1} = 1$ 이므로 $1 \times 0.4 = 0.4$
6. [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84) 1로 맞추려면 $0.4$로 나눈다: $0.3 \div 0.4 = 0.75$, $0.4 \div 0.4 = 1$

**답**: 같은 업데이트가 [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84)에 따라 $0.4$ (스펙트럼, RMS→RMS), $0.5$ (프로베니우스), $0.7$ (핵)으로 다르게 잰다. RMS→[RMS 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rms-norm-rms-%EB%85%B8%EB%A6%84) 1로 맞춘 업데이트는 $\begin{pmatrix}0.75&0\\0&1\end{pmatrix}$

</details>

### 문제 10

[Lec07 8](../../Lec07_%EC%B5%9C%EC%A0%81%ED%99%94%EC%9D%98_%EC%8A%A4%EC%BC%80%EC%9D%BC%EB%A7%81_%EA%B7%9C%EC%B9%99.md#8-rms-%EC%97%B0%EC%82%B0%EC%9E%90-%EB%85%B8%EB%A6%84%EA%B3%BC-%EB%84%88%EB%B9%84-%EC%8A%A4%EC%BC%80%EC%9D%BC%EB%A7%81)은 초기 [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)의 RMS→[RMS 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rms-norm-rms-%EB%85%B8%EB%A6%84)을 1 근처로 맞추자고 한다. 너비 $d = 1024$인 정사각 [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)에서 성분을 $N(0, 1/d)$로 뽑으면 [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84)이 약 2, $N(0, 1)$로 뽑으면 $\sqrt{d}$배인 약 $2\sqrt{d}$이다. 두 경우의 RMS→[RMS 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#rms-norm-rms-%EB%85%B8%EB%A6%84)을 어림하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $d_\text{in} = d_\text{out} = 1024$
- $N(0, 1/d)$: [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84) 약 $2$
- $N(0, 1)$: [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84) 약 $2\sqrt{d}$

**풀이**
1. 식을 그대로 쓴다: $\lVert W \rVert_{\mathrm{RMS}\to\mathrm{RMS}} = \sqrt{\dfrac{d_\text{in}}{d_\text{out}}}\, \lVert W \rVert_\ast$
2. 배율을 계산한다: $d_\text{in} = d_\text{out}$ 이므로 $\sqrt{\dfrac{1024}{1024}} = \sqrt{1} = 1$
3. $N(0, 1/d)$를 대입한다: $1 \times 2 = 2$
4. $N(0, 1)$의 [스펙트럼 노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#spectral-norm-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EB%85%B8%EB%A6%84)을 구한다: $\sqrt{1024} = 32$ ($32 \times 32 = 1024$) 이므로 $2 \times 32 = 64$
5. $N(0, 1)$을 대입한다: $1 \times 64 = 64$

**답**: $N(0, 1/d)$이면 약 $2$, $N(0, 1)$이면 약 $64$ ([층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)을 쌓을수록 $64$배씩 커져 폭발한다)

</details>

