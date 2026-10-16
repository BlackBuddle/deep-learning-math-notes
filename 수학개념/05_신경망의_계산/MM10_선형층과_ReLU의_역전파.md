---
title: 선형층과 ReLU의 역전파
type: note
tags: [MIT딥러닝, math]
created: 2026-10-15
cssclasses: [dl-note]
---

# 선형층과 ReLU의 역전파 (Backprop through linear layers, ReLU and branches)

[목차로 돌아가기](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md) · 먼저 볼 것: [MM09_야코비안과_연쇄법칙](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/05_%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%9D%98_%EA%B3%84%EC%82%B0/MM09_%EC%95%BC%EC%BD%94%EB%B9%84%EC%95%88%EA%B3%BC_%EC%97%B0%EC%87%84%EB%B2%95%EC%B9%99.md), [MM05_활성화_함수의_미분](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/05_%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%9D%98_%EA%B3%84%EC%82%B0/MM05_%ED%99%9C%EC%84%B1%ED%99%94_%ED%95%A8%EC%88%98%EC%9D%98_%EB%AF%B8%EB%B6%84.md), [M05_벡터와_행렬](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/03_%EC%84%A0%ED%98%95%EB%8C%80%EC%88%98/M05_%EB%B2%A1%ED%84%B0%EC%99%80_%ED%96%89%EB%A0%AC.md) · 쓰는 곳: [Lec02 7.1](../../Lec02_%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%95%99%EC%8A%B5.md#71-%EC%84%A0%ED%98%95%EC%B8%B5), [Lec02 7.2](../../Lec02_%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%95%99%EC%8A%B5.md#72-mlp%EC%99%80-relu-%EA%B2%8C%EC%9D%B4%ED%8A%B8), [Lec02 8.1](../../Lec02_%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%95%99%EC%8A%B5.md#81-%ED%95%A9%EC%B9%98%EA%B8%B0%EC%99%80-%EA%B0%80%EC%A7%80%EC%B9%98%EA%B8%B0), [Lec02 8.2](../../Lec02_%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%95%99%EC%8A%B5.md#82-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-%EA%B3%B5%EC%9C%A0)

📚 학습 순서 39/99 (5. 신경망의 계산): ← 이전: [MM09_야코비안과_연쇄법칙](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/05_%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%9D%98_%EA%B3%84%EC%82%B0/MM09_%EC%95%BC%EC%BD%94%EB%B9%84%EC%95%88%EA%B3%BC_%EC%97%B0%EC%87%84%EB%B2%95%EC%B9%99.md) · 다음: [MM08_SGD와_모멘텀](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/05_%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%9D%98_%EA%B3%84%EC%82%B0/MM08_SGD%EC%99%80_%EB%AA%A8%EB%A9%98%ED%85%80.md) →

> [!NOTE]
> **3줄 요약**
> - [선형층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#linear-layer-%EC%84%A0%ED%98%95%EC%B8%B5) $y = Wx$의 [역전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#backpropagation-%EC%97%AD%EC%A0%84%ED%8C%8C)는 곱셈 두 번이다: 입력 쪽 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $g_{in} = g_{out}W$, [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $\partial J/\partial W = g_{out}^\top x^\top$ (출력 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)와 입력의 바깥곱).
> - [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)는 역방향에서 **[게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)**다. [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C)에서 양수였던 칸만 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 통과시킨다. 예: z = (−1, 2.5)이면 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8) (0, 1), [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) (−7, 3.5)가 (0, 3.5)가 된다.
> - 값이 두 갈래로 쓰이면([가지치기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pruning-%EA%B0%80%EC%A7%80%EC%B9%98%EA%B8%B0), [파라미터 공유](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-sharing-%EA%B0%80%EC%A4%91%EC%B9%98-%EA%B3%B5%EC%9C%A0)) [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)는 **더한다**. u = 3x, v = x², J = u + v이면 x = 2에서 3 + 4 = 7.

## 1. 개념 설명

비유: 여러 단계로 된 조립 라인에서 완성품에 흠([손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98))이 생겼다. 책임을 거꾸로 따져 나간다.

- **[선형층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#linear-layer-%EC%84%A0%ED%98%95%EC%B8%B5)**: 출력 칸 $i$는 입력 칸마다 $W_{ij}$만큼 섞어 만든 것이다. 그러니 출력 칸이 받은 책임($g_{out,i}$)은 같은 비율 $W_{ij}$로 입력 칸들에 나뉘어 돌아간다($g_{in} = g_{out}W$). 그리고 $W_{ij}$ 자체의 책임은 "그 칸이 받은 책임 × 그 칸에 들어온 재료의 양" $= g_{out,i}\,x_j$다.
- **[ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)**: [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C)에서 0으로 막혔던 칸은 결과에 아무 영향도 주지 않았으니 책임도 0이다. 열렸던 칸은 받은 책임을 그대로 넘긴다.
- **[가지치기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pruning-%EA%B0%80%EC%A7%80%EC%B9%98%EA%B8%B0)**: 같은 재료가 두 라인에 쓰였다면 그 재료의 책임은 두 라인에서 온 책임의 합이다.

$$
x_{out} = W x_{in}, \qquad g_{in} = g_{out}\,W, \qquad \frac{\partial J}{\partial W} = g_{out}^\top\, x_{in}^\top, \qquad g_{z} = g_{h} \odot \mathbb{1}[z > 0]
$$

<details><summary>기호 풀이</summary>

| 기호 | 읽는 법 | 뜻 | 예제 값 |
| --- | --- | --- | --- |
| $x_{in},\ x_{out}$ | 엑스 인, 엑스 아웃 | [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)의 입력과 출력 | (1, 2), (−5, 4) |
| $W$ | 더블유 | [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 행렬 | 두 행 (1, −3), (2, 1) |
| $g_{out},\ g_{in}$ | 지 아웃, 지 인 | [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) 출력·입력 쪽 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) | (0, 3), (6, 3) |
| $g_{out}^\top x_{in}^\top$ | 바깥곱 | (i, j) 칸이 $g_i x_j$인 행렬 | 2행 (3, 6) |
| $z,\ h$ | 제트, 에이치 | [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)에 들어가는 값과 나온 값 | (−5, 4), (0, 4) |
| $g_z,\ g_h$ | 지 제트, 지 에이치 | z 쪽, h 쪽 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) | (0, 3), (4, 3) |
| $\odot$ | [원소별 곱](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#hadamard-product-%EC%9B%90%EC%86%8C%EB%B3%84-%EA%B3%B1) | 같은 자리끼리 곱함 | (4, 3) ⊙ (0, 1) |
| $\mathbb{1}[z > 0]$ | 지시함수 | z가 양수인 칸은 1, 아니면 0 | (0, 1) |

</details>

<details><summary>예제와 풀이</summary>

**문제**: 입력 x = (1, 2)가 [선형층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#linear-layer-%EC%84%A0%ED%98%95%EC%B8%B5) $z = Wx$와 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)를 지나고, 위에서 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 출력 쪽 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $g_h = (4, 3)$이 내려왔을 때 $g_z$, 입력 쪽 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $g_{in}$, [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $\partial J/\partial W$는? ([선형층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#linear-layer-%EC%84%A0%ED%98%95%EC%B8%B5)의 출력이 z라서 [선형층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#linear-layer-%EC%84%A0%ED%98%95%EC%B8%B5)의 $g_{out}$은 $g_z$다)

**주어진 값**:
- $x_{in} = (1, 2)$ ([선형층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#linear-layer-%EC%84%A0%ED%98%95%EC%B8%B5) 입력)
- $W = \begin{pmatrix} 1 & -3 \\ 2 & 1 \end{pmatrix}$ ([가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 행렬)
- $g_h = (4, 3)$ ([ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 출력 h 쪽에서 내려온 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))

**풀이**
1. 식을 그대로 쓴다: $x_{out} = W x_{in}, \qquad g_{in} = g_{out}\,W, \qquad \frac{\partial J}{\partial W} = g_{out}^\top\, x_{in}^\top, \qquad g_{z} = g_{h} \odot \mathbb{1}[z > 0]$
2. 값을 대입한다: $z = \begin{pmatrix} 1 & -3 \\ 2 & 1 \end{pmatrix}\begin{pmatrix} 1 \\ 2 \end{pmatrix}, \quad g_z = (4, 3) \odot \mathbb{1}[z > 0], \quad g_{in} = g_z\begin{pmatrix} 1 & -3 \\ 2 & 1 \end{pmatrix}, \quad \frac{\partial J}{\partial W} = g_z^\top (1,\ 2)$
3. z 1번째 칸: W의 1행과 x를 곱한다: $1 \times 1 = 1$, $(-3) \times 2 = -6$
4. 더한다: $z_1 = 1 + (-6) = -5$
5. z 2번째 칸: W의 2행과 x를 곱한다: $2 \times 1 = 2$, $1 \times 2 = 2$
6. 더한다: $z_2 = 2 + 2 = 4$
7. [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)를 만든다: −5는 0 이하라 0, 4는 양수라 1 → $\mathbb{1}[z > 0] = (0, 1)$
8. 같은 자리끼리 곱한다: $4 \times 0 = 0$, $3 \times 1 = 3$ → $g_z = (0, 3)$
9. $g_{in}$ 1번째 칸: $g_z$와 W의 1번째 열을 곱한다: $0 \times 1 = 0$, $3 \times 2 = 6$
10. 더한다: $0 + 6 = 6$
11. $g_{in}$ 2번째 칸: 2번째 열과 곱한다: $0 \times (-3) = 0$, $3 \times 1 = 3$
12. 더한다: $0 + 3 = 3$ → $g_{in} = (6, 3)$
13. $\partial J/\partial W$ 1행: $g_z$의 1번째 칸 0을 $(1, 2)$에 곱한다: $(0, 0)$
14. $\partial J/\partial W$ 2행: $g_z$의 2번째 칸 3을 $(1, 2)$에 곱한다: $(3, 6)$

**답**: $g_z = (0, 3)$, $g_{in} = (6, 3)$, $\frac{\partial J}{\partial W} = \begin{pmatrix} 0 & 0 \\ 3 & 6 \end{pmatrix}$

</details>

<details><summary>내 이해</summary>



</details>

## 2. 필요한 이유

- [Lec02 7.1](../../Lec02_%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%95%99%EC%8A%B5.md#71-%EC%84%A0%ED%98%95%EC%B8%B5)의 치트 시트 "순방향 $Wx$, 역방향 $g_{out}W$, [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $x_{in}g_{out}$"을 실제 숫자로 따라가 본다.
- [Lec02 7.2](../../Lec02_%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%95%99%EC%8A%B5.md#72-mlp%EC%99%80-relu-%EA%B2%8C%EC%9D%B4%ED%8A%B8)의 "[ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)는 역방향에서 대각 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8) 행렬"이라는 말을 확인한다. [역전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#backpropagation-%EC%97%AD%EC%A0%84%ED%8C%8C)를 하려면 [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C)의 중간값(z)을 저장해야 하는 이유도 여기서 보인다.
- [Lec02 8.1](../../Lec02_%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%95%99%EC%8A%B5.md#81-%ED%95%A9%EC%B9%98%EA%B8%B0%EC%99%80-%EA%B0%80%EC%A7%80%EC%B9%98%EA%B8%B0): [가지치기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pruning-%EA%B0%80%EC%A7%80%EC%B9%98%EA%B8%B0)에서 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 더하는 규칙을 가장 작은 예로 확인한다.

## 3. 숫자로 보기

2층 MLP: $x = (1, 2)$, $W_1 = \begin{pmatrix} 1 & -1 \\ 0.5 & 1 \end{pmatrix}$, $W_2 = (2,\ -1)$, [정답](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5) $t = 1$, [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98) $J = \frac{1}{2}(y - t)^2$.

**[순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C)** (한 줄에 한 동작)
- $z_1 = 1×1 + (-1)×2 = -1$
- $z_2 = 0.5×1 + 1×2 = 2.5$
- $h = \mathrm{ReLU}(z) = (0,\ 2.5)$ (첫 칸은 음수라 0)
- $y = 2×0 + (-1)×2.5 = -2.5$
- $J = \frac{1}{2}(-2.5 - 1)^2 = \frac{1}{2} × 12.25 = 6.125$

**[역전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#backpropagation-%EC%97%AD%EC%A0%84%ED%8C%8C)** (위에서 아래로)
- $g_y = \partial J/\partial y = y - t = -3.5$
- $\partial J/\partial W_2 = g_y × h = -3.5 × (0, 2.5) = (0,\ -8.75)$
- $g_h = g_y W_2 = -3.5 × (2, -1) = (-7,\ 3.5)$
- [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8): $z = (-1, 2.5)$ → $(0, 1)$ → $g_z = (-7×0,\ 3.5×1) = (0,\ 3.5)$
- $\partial J/\partial W_1 = g_z^\top x^\top$: 1행 $0 × (1, 2) = (0, 0)$, 2행 $3.5 × (1, 2) = (3.5, 7)$
- $g_x = g_z W_1 = (0×1 + 3.5×0.5,\ 0×(-1) + 3.5×1) = (1.75,\ 3.5)$

| 위치 | [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C) 값 | [역전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#backpropagation-%EC%97%AD%EC%A0%84%ED%8C%8C) [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) |
|---|---|---|
| x | (1, 2) | (1.75, 3.5) |
| z | (−1, 2.5) | (0, 3.5) |
| h | (0, 2.5) | (−7, 3.5) |
| y | −2.5 | −3.5 |
| J | 6.125 | 1 |

| [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) | [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) | 읽는 법 |
|---|---|---|
| $W_2$ | (0, −8.75) | h₁ = 0이라 W₂의 첫 칸은 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)에 영향이 없다 |
| $W_1$ | 1행 (0, 0), 2행 (3.5, 7) | z₁이 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)에서 막혀 W₁의 1행은 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 0 |

![MM10_fig_backprop.png](../../attachments/MM10_fig_backprop.png)
*▲ 직접 그린 그림: 위 표의 값. 초록은 [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C), 빨강은 [역전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#backpropagation-%EC%97%AD%EC%A0%84%ED%8C%8C)*

**[가지치기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pruning-%EA%B0%80%EC%A7%80%EC%B9%98%EA%B8%B0)**: $x = 2$가 두 갈래 $u = 3x$, $v = x^2$로 쓰이고 $J = u + v$.
- 가지 u: $\partial J/\partial u × \partial u/\partial x = 1 × 3 = 3$
- 가지 v: $\partial J/\partial v × \partial v/\partial x = 1 × 2x = 4$
- 합: $3 + 4 = 7$ (직접 [미분](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#derivative-%EB%AF%B8%EB%B6%84) $J = 3x + x^2$ → $3 + 2x = 7$과 같다)

<details><summary>기호 풀이</summary>

| 기호 | 읽는 법 | 뜻 | 예제 값 |
| --- | --- | --- | --- |
| $g_{out}$ | 지 아웃 | [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) 출력에 대한 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) (행벡터) | $g_h = (-7, 3.5)$ |
| $g_{in}$ | 지 인 | [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) 입력에 대한 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) | $g_z = (0, 3.5)$ |
| $g^\top x^\top$ | 바깥곱 | 열벡터 × 행벡터 = 행렬. (i, j) 칸이 $g_i x_j$ | 2행 (3.5, 7) |
| $\odot$ | [원소별 곱](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#hadamard-product-%EC%9B%90%EC%86%8C%EB%B3%84-%EA%B3%B1) | 같은 자리끼리 곱함 | (−7, 3.5) ⊙ (0, 1) |
| $\mathbb{1}[z > 0]$ | 지시함수 | 조건이 참이면 1, 거짓이면 0 | (0, 1) |
| $H'$ | 에이치 프라임 | [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)를 대각선에 놓은 행렬 (강의 표기) | diag(0, 1) |

</details>

## 4. 코드로 확인

```python
# [함수] import numpy as np: 배열 계산 라이브러리 numpy를 np라는 이름으로 불러온다
import numpy as np
# [변수] x: 입력 (1, 2), t = 1.0: 정답
x, t = np.array([1.0, 2.0]), 1.0
# [변수] W1: 첫 선형층 2×2. 본문 3절의 값
W1 = np.array([[1.0, -1.0], [0.5, 1.0]])
# [변수] W2: 둘째 선형층 1×2
W2 = np.array([[2.0, -1.0]])
# [함수] def forward(x, W1, W2): 순전파로 중간값과 손실을 모두 돌려준다
def forward(x, W1, W2):
    # [변수] z = W1 x = (-1, 2.5): 첫 선형층 출력
    z = W1 @ x
    # [함수] np.maximum(z, 0): ReLU. 음수는 0으로 → h = (0, 2.5)
    h = np.maximum(z, 0)
    # [변수] y = W2 h = -2.5: 둘째 선형층 출력 (칸 1개)
    y = W2 @ h
    # [변수] J = (y - t)² / 2 = 6.125: 손실
    # [쓸모] z, h는 역전파에서 다시 쓰려고 함께 돌려준다 (Lec02 7.2 질의응답: 중간값 저장)
    return z, h, y, 0.5 * (y[0] - t) ** 2
# [변수] 순전파 결과를 받아 둔다
z, h, y, J = forward(x, W1, W2)
# [변수] gy = y - t = -3.5: 손실을 y로 미분한 값 (맨 위 기울기)
gy = y - t
# [함수] np.outer: 바깥곱. + 0.0은 -0.0을 0.0으로 바꿔 출력을 깔끔하게 한다
# [변수] gW2 = outer(gy, h) = [[0, -8.75]]: W2의 기울기 (출력 기울기 × 입력)
# [연결] 본문 1절 식 ∂J/∂W = g_outᵀ x_inᵀ, Lec02 7.1 "가중치 기울기 = 입력 × 출력 기울기"
gW2 = np.outer(gy, h) + 0.0
# [변수] gh = gy W2 = (-7, 3.5): h로 거꾸로 보낸 기울기
# [연결] Lec02 7.1 g_in = g_out · W
gh = gy @ W2
# [함수] (z > 0): ReLU 게이트. 순전파에서 양수였던 칸만 1(True) → (0, 1)
# [변수] gz = gh × 게이트 = (0, 3.5): 꺼졌던 첫 칸으로는 기울기가 가지 않는다
# [연결] Lec02 7.2 대각 게이트 행렬 H′
gz = gh * (z > 0) + 0.0
# [변수] gW1 = outer(gz, x) = [[0, 0], [3.5, 7]], gx = gz W1 = (1.75, 3.5)
gW1, gx = np.outer(gz, x) + 0.0, gz @ W1
# [연결] 출력 1행: 순전파 z, h, y, J
print(z, h, y, J)
# [연결] 출력 2행: 역전파 기울기 gy, gh, gz, gx (본문 3절 표와 그림)
print(gy, gh, gz, gx)
# [함수] ravel(): 행렬을 한 줄로 펴서 출력
# [연결] 출력 3행: W1의 기울기 (0, 0, 3.5, 7), W2의 기울기 (0, -8.75)
print(gW1.ravel(), gW2.ravel())

# [변수] e = 1e-6: 수치 미분용 아주 작은 걸음
e = 1e-6
# [함수] def num_grad(k): W1의 k번째 칸(한 줄로 폈을 때)을 e만큼 늘려 손실 변화를 잰다
def num_grad(k):
    # [함수] W1.copy(): 원본을 건드리지 않으려고 복사한다
    W = W1.copy()
    # [함수] ravel()[k] += e: k번째 칸만 e만큼 늘린다
    W.ravel()[k] += e
    # [변수] (새 J - 원래 J) / e: k번째 칸의 수치 기울기
    return (forward(x, W, W2)[3] - J) / e
# [함수] np.round(..., 4): 넷째 자리 반올림
# [연결] 출력 4행: W1의 수치 기울기. 역전파 결과 (0, 0, 3.5, 7)과 같다 = 역전파가 맞다
print(np.round([num_grad(k) for k in range(4)], 4))

# [변수] xb = 2.0: 가지치기 예의 입력. x가 u = 3x와 v = x² 두 가지로 갈라진다
xb = 2.0
# [변수] 가지 u의 기울기 3, 가지 v의 기울기 2x = 4. J = u + v이므로 각 가지의 위쪽 기울기는 1
# [연결] 출력 5행: 두 가지의 기울기 합 7.0과 수치 미분 7.0 (Lec02 8.1 가지치기)
print(3 * 1 + 2 * xb * 1, round(((3 * (xb + e) + (xb + e) ** 2) - (3 * xb + xb ** 2)) / e, 4))
```

**실행 결과**:

```text
[-1.   2.5] [0.  2.5] [-2.5] 6.125
[-3.5] [-7.   3.5] [0.  3.5] [1.75 3.5 ]
[0.  0.  3.5 7. ] [ 0.   -8.75]
[0.  0.  3.5 7. ]
7.0 7.0
```

> [!WARNING]
> **⚠ 주의**
> - 강의 슬라이드는 [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 $x_{in} \cdot g_{out}$ (입력 열벡터 × 출력 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 행벡터)로 써서 $W$의 **[전치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#transpose-%EC%A0%84%EC%B9%98)** 모양이 나오고, 그래서 갱신식에 $(\partial J/\partial W)^\top$가 붙는다. 이 노트와 [PyTorch](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pytorch-%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98)는 처음부터 $W$와 같은 모양 $g_{out}^\top x_{in}^\top$으로 쓴다. 숫자는 같다.
> - [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)의 [게이트](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gate-%EA%B2%8C%EC%9D%B4%ED%8A%B8)는 **[순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C)의 z**로 정한다. [역전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#backpropagation-%EC%97%AD%EC%A0%84%ED%8C%8C) 중에 들어오는 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)의 부호와는 상관없다. 그래서 [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C) 값을 저장해 두어야 한다.
> - z가 정확히 0인 칸에서 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)는 [미분](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#derivative-%EB%AF%B8%EB%B6%84) 불가능하다. [PyTorch](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pytorch-%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98)는 이때 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 0으로 둔다(`z > 0`과 같은 규칙).
> - [가지치기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pruning-%EA%B0%80%EC%A7%80%EC%B9%98%EA%B8%B0)에서 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 **평균**이 아니라 **합**으로 모은다. 공유 [파라미터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0)를 여러 번 쓰면 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)도 그만큼 커진다.

## 5. 개념 재확인

<details><summary>1. 선형층 W = [(1, 2), (3, 4)] (두 행), x = (1, −1), 출력 기울기 g_out = (0.5, −1)일 때 y, g_in, ∂J/∂W와 η = 0.1로 한 번 갱신한 W는?</summary>

| 단계 | 하는 일 | 식 | 계산 | 결과 |
|---|---|---|---|---|
| 1 | [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C) | $y = Wx$ | $(1 - 2,\ 3 - 4)$ | (−1, −1) |
| 2 | 입력 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) | $g_{in} = g_{out}W$ | $(0.5 - 3,\ 1 - 4)$ | (−2.5, −3) |
| 3 | [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) | $g_{out}^\top x^\top$ | 1행 0.5×(1, −1), 2행 −1×(1, −1) | 1행 (0.5, −0.5), 2행 (−1, 1) |
| 4 | 갱신 | $W - \eta\,\partial J/\partial W$ | $W - 0.1 × (\cdot)$ | 1행 (0.95, 2.05), 2행 (3.1, 3.9) |

**① 쓰는 식**
$$y = Wx, \qquad g_{in} = g_{out}W, \qquad \frac{\partial J}{\partial W} = g_{out}^\top x^\top, \qquad W \leftarrow W - \eta\frac{\partial J}{\partial W}$$
- 위 식 ← 첫째 식은 ② ([순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C)), 둘째는 ③ (입력 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)), 셋째는 ④ ([가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)), 넷째는 ⑤ (갱신)에서 쓴다.
```python
# [함수] import numpy as np: 배열 계산 라이브러리
import numpy as np
# [변수] W: 두 행 (1, 2), (3, 4). 문제에서 준 값
W = np.array([[1.0, 2.0], [3.0, 4.0]])    # ①
# [변수] x = (1, -1), g = (0.5, -1): 출력 기울기 g_out, eta = 0.1: 학습률
x, g, eta = np.array([1.0, -1.0]), np.array([0.5, -1.0]), 0.1
# [변수] y = Wx = (-1, -1)
print(W @ x)                              # ②
# [변수] g_in = g W = (-2.5, -3)
print(g @ W)                              # ③
# [함수] np.outer: 바깥곱 → [[0.5, -0.5], [-1, 1]]
dW = np.outer(g, x)                       # ④
# [연결] 갱신 결과 [[0.95, 2.05], [3.1, 3.9]]. 기울기 반대 방향으로 0.1배 이동
print(dW.ravel(), (W - eta * dW).ravel()) # ⑤
```

**주어진 값**
- $W = \begin{pmatrix} 1 & 2 \ 3 & 4 \end{pmatrix}$ ([가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 행렬)
- $x = (1, -1)$ (입력)
- $g_{out} = (0.5, -1)$ (출력 쪽 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))
- $\eta = 0.1$ ([학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))

**② [순전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#feed-forward-%EC%88%9C%EC%A0%84%ED%8C%8C)**
1. 식을 그대로 쓴다: $y = Wx$
2. 값을 대입한다: $y = \begin{pmatrix} 1 & 2 \ 3 & 4 \end{pmatrix}\begin{pmatrix} 1 \ -1 \end{pmatrix}$
3. 1번째 칸을 펼친다 (W의 1행과 x): $y_1 = 1 \times 1 + 2 \times (-1)$
4. 계산한다: 1 − 2 = −1 이므로 $y_1 = -1$
5. 2번째 칸을 펼친다 (W의 2행과 x): $y_2 = 3 \times 1 + 4 \times (-1)$
6. 계산한다: 3 − 4 = −1 이므로 $y_2 = -1$

**③ 입력 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)**
1. 식을 그대로 쓴다: $g_{in} = g_{out}\,W$
2. 값을 대입한다: $g_{in} = (0.5,\ -1)\begin{pmatrix} 1 & 2 \ 3 & 4 \end{pmatrix}$
3. 모양을 확인한다: [1×2][2×2] = [1×2]
4. 1번째 칸을 펼친다 (W의 1열과 곱함): $0.5 \times 1 + (-1) \times 3$
5. 계산한다: 0.5 − 3 = −2.5
6. 2번째 칸을 펼친다 (W의 2열과 곱함): $0.5 \times 2 + (-1) \times 4$
7. 계산한다: 1 − 4 = −3 이므로 $g_{in} = (-2.5, -3)$

**④ [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)**
1. 식을 그대로 쓴다: $\frac{\partial J}{\partial W} = g_{out}^\top\, x^\top$ ((i, j) 칸이 $g_i \times x_j$인 바깥곱)
2. 값을 대입한다: $\begin{pmatrix} 0.5 \ -1 \end{pmatrix}(1,\ -1)$
3. 1행을 계산한다: $0.5 \times 1 = 0.5$, $0.5 \times (-1) = -0.5$ 이므로 1행 (0.5, −0.5)
4. 2행을 계산한다: $(-1) \times 1 = -1$, $(-1) \times (-1) = 1$ 이므로 2행 (−1, 1)

**⑤ 갱신**
1. 식을 그대로 쓴다: $W \leftarrow W - \eta\,\frac{\partial J}{\partial W}$
2. 값을 대입한다: $W \leftarrow \begin{pmatrix} 1 & 2 \ 3 & 4 \end{pmatrix} - 0.1 \times \begin{pmatrix} 0.5 & -0.5 \ -1 & 1 \end{pmatrix}$
3. [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)을 칸마다 곱한다: 0.1 × 0.5 = 0.05, 0.1 × (−0.5) = −0.05, 0.1 × (−1) = −0.1, 0.1 × 1 = 0.1
4. 칸끼리 뺀다 (1행): 1 − 0.05 = 0.95, 2 − (−0.05) = 2.05
5. 칸끼리 뺀다 (2행): 3 − (−0.1) = 3.1, 4 − 0.1 = 3.9

**답**: y = **(−1, −1)**, g_in = **(−2.5, −3)**, ∂J/∂W = **1행 (0.5, −0.5), 2행 (−1, 1)**, 새 W = **1행 (0.95, 2.05), 2행 (3.1, 3.9)**

</details>

<details><summary>2. 지름길 연결 y = Wx + x (W의 두 행 (1, 2), (0, 1)), J = y₁ + y₂일 때 x에 대한 기울기는? 가지치기 규칙으로 구하라.</summary>

| 단계 | 하는 일 | 식 | 계산 | 결과 |
|---|---|---|---|---|
| 1 | 맨 위 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) | $g_y = \partial J/\partial y$ | J가 두 칸의 합 | (1, 1) |
| 2 | 선형 가지 | $g_y W$ | $(1×1 + 1×0,\ 1×2 + 1×1)$ | (1, 3) |
| 3 | 지름길 가지 | $g_y × 1$ | 그대로 통과 | (1, 1) |
| 4 | 가지 합 | 두 가지 더하기 | $(1 + 1,\ 3 + 1)$ | (2, 4) |

**① 쓰는 식**
$$g_x = \underbrace{g_y W}_{\text{선형 가지}} + \underbrace{g_y I}_{\text{지름길 가지}}$$
- 위 식 ← ② (맨 위 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))를 두 가지에 똑같이 보내 ③ (선형 가지)와 ④ (지름길 가지)를 구하고, ⑤ (가지 합)에서 더한다.
```python
# [함수] import numpy as np: 배열 계산 라이브러리
import numpy as np
# [변수] W: 두 행 (1, 2), (0, 1). 문제에서 준 값
W = np.array([[1.0, 2.0], [0.0, 1.0]])    # ①
# [변수] gy = (1, 1): J = y1 + y2를 y로 미분
gy = np.array([1.0, 1.0])                 # ②
# [변수] lin = gy W = (1, 3): 선형층 가지로 돌아온 기울기
lin = gy @ W                              # ③
# [변수] skip = gy = (1, 1): 지름길(항등) 가지는 기울기를 그대로 보낸다
skip = gy                                 # ④
# [연결] Lec02 8.1 가지치기: 두 가지의 기울기를 더한다 → (2, 4)
print(lin, skip, lin + skip)              # ⑤
```

**주어진 값**
- $y = Wx + x$ (선형 가지 $Wx$와 지름길 가지 $x$가 더해진다)
- $W = \begin{pmatrix} 1 & 2 \ 0 & 1 \end{pmatrix}$
- $J = y_1 + y_2$ ([손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98))

**② 맨 위 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)**
1. 식을 그대로 쓴다: $g_y = \left(\frac{\partial J}{\partial y_1},\ \frac{\partial J}{\partial y_2}\right)$
2. 값을 대입한다: $J = y_1 + y_2$ 이므로 $\frac{\partial J}{\partial y_1} = 1$, $\frac{\partial J}{\partial y_2} = 1$
3. 모은다: $g_y = (1, 1)$

**③ 선형 가지**
1. 식을 그대로 쓴다: $g_y\,W$
2. 값을 대입한다: $(1,\ 1)\begin{pmatrix} 1 & 2 \ 0 & 1 \end{pmatrix}$
3. 1번째 칸을 펼친다: $1 \times 1 + 1 \times 0$
4. 계산한다: 1 + 0 = 1
5. 2번째 칸을 펼친다: $1 \times 2 + 1 \times 1$
6. 계산한다: 2 + 1 = 3 이므로 선형 가지는 (1, 3)

**④ 지름길 가지**
1. 식을 그대로 쓴다: $g_y\,I$ (지름길은 $x$를 그대로 더하므로 [야코비안](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#jacobian-%EC%95%BC%EC%BD%94%EB%B9%84%EC%95%88)이 단위행렬 $I$)
2. 값을 대입한다: $(1,\ 1)\begin{pmatrix} 1 & 0 \ 0 & 1 \end{pmatrix}$
3. 계산한다: 1×1 + 1×0 = 1, 1×0 + 1×1 = 1 이므로 지름길 가지는 (1, 1)

**⑤ 가지 합**
1. 식을 그대로 쓴다: $g_x = g_y W + g_y I$
2. 값을 대입한다: $g_x = (1, 3) + (1, 1)$
3. 칸끼리 더한다: 1 + 1 = 2, 3 + 1 = 4 이므로 $g_x = (2, 4)$
4. 검산: $J = (x_1 + 2x_2) + x_2 + x_1 + x_2 = 2x_1 + 4x_2$ 이므로 $\left(\frac{\partial J}{\partial x_1},\ \frac{\partial J}{\partial x_2}\right) = (2, 4)$

**답**: $g_x$ = **(2, 4)**. x가 두 갈래로 쓰였으니 두 갈래의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 더한다.

</details>

## 예제 풀기

[예제 10문제 풀기](../../%EC%98%88%EC%A0%9C/05_%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%9D%98_%EA%B3%84%EC%82%B0/%EC%98%88%EC%A0%9C_MM10_%EC%84%A0%ED%98%95%EC%B8%B5%EA%B3%BC_ReLU%EC%9D%98_%EC%97%AD%EC%A0%84%ED%8C%8C.md)

<details><summary>내 이해</summary>



</details>
