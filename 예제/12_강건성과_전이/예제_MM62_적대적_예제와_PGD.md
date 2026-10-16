---
tags: [MIT딥러닝, 예제]
cssclasses: [dl-note]
---

# 예제: 적대적 예제와 투영 경사 상승

개념 노트: [MM62_적대적_예제와_PGD](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/12_%EA%B0%95%EA%B1%B4%EC%84%B1%EA%B3%BC_%EC%A0%84%EC%9D%B4/MM62_%EC%A0%81%EB%8C%80%EC%A0%81_%EC%98%88%EC%A0%9C%EC%99%80_PGD.md) · [목차](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md)

풀이는 접혀 있으니 먼저 풀어 보고 펼쳐 확인한다. 쓰는 식은 셋뿐이다. $L_\infty$ [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84) $\lVert\delta\rVert_\infty = \max_i |\delta_i|$, 칸별 자르기 $\min(\max(v_i, -\epsilon), \epsilon)$, 선형 모형의 최악 교란 $\delta = \epsilon\,\text{sign}(\mathbf{w})$ (점수 변화 $\epsilon\lVert\mathbf{w}\rVert_1$).

## 기본 (직접 대입)

### 문제 1

교란 $\delta = (0.03, -0.04)$의 $L_\infty$ [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84)과 $L_2$ [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84)을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\delta = (0.03, -0.04)$ (교란)

**풀이**
1. 식을 그대로 쓴다: $\lVert\delta\rVert_\infty = \max_i |\delta_i|$, $\lVert\delta\rVert_2 = \sqrt{\delta_1^2 + \delta_2^2}$
2. 값을 대입한다: $\lVert\delta\rVert_\infty = \max(|0.03|, |-0.04|)$, $\lVert\delta\rVert_2 = \sqrt{0.03^2 + (-0.04)^2}$
3. 절댓값을 구한다: $|0.03| = 0.03$, $|-0.04| = 0.04$
4. 가장 큰 값을 고른다: $\max(0.03, 0.04) = 0.04$
5. 제곱을 한다: $0.03^2 = 0.0009$, $(-0.04)^2 = 0.0016$ 이므로 $\lVert\delta\rVert_2 = \sqrt{0.0009 + 0.0016}$
6. 더한다: $0.0009 + 0.0016 = 0.0025$ 이므로 $\lVert\delta\rVert_2 = \sqrt{0.0025}$
7. 제곱근을 구한다 ($\sqrt{a}$는 제곱해서 $a$가 되는 양수): $0.05 \times 0.05 = 0.0025$ 이므로 $\sqrt{0.0025} = 0.05$

**답**: $\lVert\delta\rVert_\infty = 0.04$, $\lVert\delta\rVert_2 = 0.05$. $L_\infty$는 가장 크게 바뀐 칸만 본다.

</details>

### 문제 2

교란 후보 $\mathbf{v} = (0.12, -0.3, 0.04, -0.05)$를 반지름 $\epsilon = 0.1$인 $L_\infty$ 공으로 투영하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mathbf{v} = (0.12, -0.3, 0.04, -0.05)$ ([투영](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#projection-%ED%88%AC%EC%98%81) 전 교란 후보)
- $\epsilon = 0.1$ (공의 반지름)

**풀이**
1. 식을 그대로 쓴다: $\big(\Pi_\Delta(\mathbf{v})\big)_i = \min\big(\max(v_i, -\epsilon), \epsilon\big)$
2. 값을 대입한다: $\Pi_\Delta(\mathbf{v}) = \big(\min(\max(0.12, -0.1), 0.1),\ \min(\max(-0.3, -0.1), 0.1),\ \min(\max(0.04, -0.1), 0.1),\ \min(\max(-0.05, -0.1), 0.1)\big)$
3. 첫째 칸: $\max(0.12, -0.1) = 0.12$, 다음 $\min(0.12, 0.1) = 0.1$
4. 둘째 칸: $\max(-0.3, -0.1) = -0.1$, 다음 $\min(-0.1, 0.1) = -0.1$
5. 셋째 칸: $\max(0.04, -0.1) = 0.04$, 다음 $\min(0.04, 0.1) = 0.04$
6. 넷째 칸: $\max(-0.05, -0.1) = -0.05$, 다음 $\min(-0.05, 0.1) = -0.05$

**답**: $(0.1, -0.1, 0.04, -0.05)$. 범위 $[-0.1, 0.1]$ 밖의 두 칸만 잘리고 안쪽 두 칸은 그대로다.

</details>

### 문제 3

[가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) $\mathbf{w} = (3, -1, 2, -4)$인 선형 모형의 점수를 $\epsilon = 0.02$ 안의 교란으로 가장 크게 올리려 한다. 최악 교란 $\delta$와 점수 변화를 구하고, $\epsilon\lVert\mathbf{w}\rVert_1$과 같은지 확인하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mathbf{w} = (3, -1, 2, -4)$ ([가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98))
- $\epsilon = 0.02$ (칸마다 최대 변화)

**풀이**
1. 식을 그대로 쓴다: $\delta = \epsilon\,\text{sign}(\mathbf{w})$, 점수 변화 $= \mathbf{w}^\top\delta$, 확인값 $= \epsilon\lVert\mathbf{w}\rVert_1$
2. 값을 대입한다: $\delta = 0.02 \times \text{sign}(3, -1, 2, -4)$, 점수 변화 $= (3, -1, 2, -4)^\top\delta$, 확인값 $= 0.02 \times \lVert(3, -1, 2, -4)\rVert_1$
3. 부호를 구한다: $\text{sign}(3, -1, 2, -4) = (1, -1, 1, -1)$
4. $\epsilon$을 곱한다: $\delta = (0.02, -0.02, 0.02, -0.02)$
5. 칸끼리 곱한다: $3 \times 0.02 = 0.06$, $(-1) \times (-0.02) = 0.02$, $2 \times 0.02 = 0.04$, $(-4) \times (-0.02) = 0.08$
6. 더한다: $0.06 + 0.02 + 0.04 + 0.08 = 0.2$
7. 확인한다. 절댓값을 더한다: $\lVert\mathbf{w}\rVert_1 = 3 + 1 + 2 + 4 = 10$
8. $\epsilon$을 곱한다: $0.02 \times 10 = 0.2$

**답**: 최악 교란 $(0.02, -0.02, 0.02, -0.02)$, 점수 변화 $0.2 = \epsilon\lVert\mathbf{w}\rVert_1$. 칸마다 [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)와 같은 부호로 움직였기 때문에 모든 항이 양수다.

</details>

### 문제 4

CIFAR-10 크기의 사진 $32 \times 32 \times 3 = 3072$ 픽셀 대신, 픽셀이 $d = 1000$개인 작은 사진을 생각하자. 선형 모형의 [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 크기가 모두 $|w_i| = 0.02$이고 $\epsilon = 8/255$(픽셀 값 0~1 기준)일 때, 최악 교란이 점수를 움직이는 양을 구하라. 한 픽셀 몫은 얼마인가?

<details><summary>풀이와 답</summary>

선형 모형의 최악 점수 변화 $\epsilon\lVert\mathbf{w}\rVert_1$을 쓴다. $L_\infty$ 공 안에서 모든 픽셀을 [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)의 부호 쪽으로 $\epsilon$만큼 움직이면 점수가 정확히 이만큼 바뀌기 때문이다.

**주어진 값**
- $d = 1000$ (픽셀 수)
- $|w_i| = 0.02$ (모든 [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)의 크기)
- $\epsilon = 8/255$ (255단계 중 8단계를 0~1 값으로 바꾼 것)

**풀이**
1. 식을 그대로 쓴다: 점수 변화 $= \epsilon\lVert\mathbf{w}\rVert_1$, $\lVert\mathbf{w}\rVert_1 = \sum_{i=1}^{d} |w_i|$
2. 값을 대입한다: 점수 변화 $= \dfrac{8}{255} \times \sum_{i=1}^{1000} 0.02$
3. Σ를 계산한다 (같은 값 1000개의 합): $1000 \times 0.02 = 20$ 이므로 $\lVert\mathbf{w}\rVert_1 = 20$
4. $\epsilon$을 곱한다 (반올림하지 않고 분수로 곱한다): $\dfrac{8}{255} \times 20 = \dfrac{160}{255}$
5. 나눈다: $160 / 255 = 0.6275$
6. 한 픽셀 몫을 구한다: $\epsilon \times |w_i| = \dfrac{8}{255} \times 0.02 = 0.0006$ ($8 \times 0.02 = 0.16$, $0.16 / 255 = 0.00063$)

**답**: 점수 변화 약 $0.6275$, 한 픽셀 몫은 약 $0.0006$. 한 픽셀은 티도 안 나지만 1000개가 같은 쪽으로 모이면 점수가 0.6 넘게 움직인다.

</details>

## 응용

### 문제 5

현재 교란이 $\delta = (0.05, -0.1)$이고 점수의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 방향이 $\mathbf{w} = (2, -3)$이다. 부호 걸음 $\alpha = 0.08$, 반지름 $\epsilon = 0.1$로 PGD를 한 번 걷고 투영한 새 $\delta$를 구하라.

<details><summary>풀이와 답</summary>

PGD 한 걸음은 "부호로 한 걸음 가서 공 안으로 자르기"다: $\delta \leftarrow \Pi_\Delta\big(\delta + \alpha\,\text{sign}(\nabla)\big)$.

**주어진 값**
- $\delta = (0.05, -0.1)$ (지금 교란)
- $\nabla = \mathbf{w} = (2, -3)$ (점수의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))
- $\alpha = 0.08$ (걸음 크기), $\epsilon = 0.1$ (공의 반지름)

**풀이**
1. 식을 그대로 쓴다: $\mathbf{v} = \delta + \alpha\,\text{sign}(\nabla)$, $\delta_{\text{new}} = \min(\max(v_i, -\epsilon), \epsilon)$
2. 값을 대입한다: $\mathbf{v} = (0.05, -0.1) + 0.08 \times \text{sign}(2, -3)$
3. 부호를 구한다: $\text{sign}(2, -3) = (1, -1)$ 이므로 $\mathbf{v} = (0.05, -0.1) + 0.08 \times (1, -1)$
4. $\alpha$를 곱한다: $0.08 \times 1 = 0.08$, $0.08 \times (-1) = -0.08$ 이므로 $\mathbf{v} = (0.05, -0.1) + (0.08, -0.08)$
5. 칸끼리 더한다: $0.05 + 0.08 = 0.13$, $-0.1 + (-0.08) = -0.18$ 이므로 $\mathbf{v} = (0.13, -0.18)$
6. 첫째 칸을 자른다: $\max(0.13, -0.1) = 0.13$, 다음 $\min(0.13, 0.1) = 0.1$
7. 둘째 칸을 자른다: $\max(-0.18, -0.1) = -0.1$, 다음 $\min(-0.1, 0.1) = -0.1$

**답**: $\delta_{\text{new}} = (0.1, -0.1)$. 두 칸 모두 공의 모서리에 닿았다.

</details>

### 문제 6

원래 교란 $\delta = (0, 0)$에서 점수의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 $\mathbf{g} = (0.9, -0.6)$이다. [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)(부호가 아님)로 $\eta = 0.5$만큼 한 걸음 간 뒤 $\epsilon = 0.15$ 공으로 투영하라. 공 안에서 끝난 걸음인가?

<details><summary>풀이와 답</summary>

[기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 걸음은 $\delta \leftarrow \Pi_\Delta(\delta + \eta\,\mathbf{g})$다. 부호 걸음과 달리 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)의 크기가 걸음에 그대로 들어간다.

**주어진 값**
- $\delta = (0, 0)$ (출발점)
- $\mathbf{g} = (0.9, -0.6)$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))
- $\eta = 0.5$ (걸음 크기), $\epsilon = 0.15$ (공의 반지름)

**풀이**
1. 식을 그대로 쓴다: $\mathbf{v} = \delta + \eta\,\mathbf{g}$, $\delta_{\text{new}} = \min(\max(v_i, -\epsilon), \epsilon)$
2. 값을 대입한다: $\mathbf{v} = (0, 0) + 0.5 \times (0.9, -0.6)$
3. $\eta$를 곱한다: $0.5 \times 0.9 = 0.45$, $0.5 \times (-0.6) = -0.3$ 이므로 $\mathbf{v} = (0, 0) + (0.45, -0.3)$
4. 더한다: $\mathbf{v} = (0.45, -0.3)$
5. 공 안인지 본다: $|0.45| > 0.15$, $|-0.3| > 0.15$ 이므로 두 칸 모두 공 밖이다
6. 첫째 칸을 자른다: $\max(0.45, -0.15) = 0.45$, 다음 $\min(0.45, 0.15) = 0.15$
7. 둘째 칸을 자른다: $\max(-0.3, -0.15) = -0.15$, 다음 $\min(-0.15, 0.15) = -0.15$

**답**: $\delta_{\text{new}} = (0.15, -0.15)$. 큰 걸음이라 공 밖으로 나갔다가 [투영](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#projection-%ED%88%AC%EC%98%81)으로 모서리에 돌아왔다.

</details>

### 문제 7

$\mathbf{w} = (1, -2, 0)$인 선형 모형의 점수를 올리는 PGD를 $\delta = (0, 0, 0)$에서 시작해 부호 걸음 $\alpha = 0.06$, $\epsilon = 0.1$로 두 번 걷는다. 걸음마다 $\delta$와 두 번째 뒤의 점수 변화 $\mathbf{w}^\top\delta$를 구하라.

<details><summary>풀이와 답</summary>

걸음마다 $\delta \leftarrow \Pi_\Delta\big(\delta + \alpha\,\text{sign}(\mathbf{w})\big)$를 쓴다. 점수의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 $\mathbf{w}$이므로 부호는 매번 같다.

**주어진 값**
- $\mathbf{w} = (1, -2, 0)$ ([가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) = 점수의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))
- $\alpha = 0.06$ (걸음 크기), $\epsilon = 0.1$ (반지름)
- $\delta_0 = (0, 0, 0)$ (출발점)

**풀이**
1. 식을 그대로 쓴다: $\delta_{t+1} = \Pi_\Delta\big(\delta_t + \alpha\,\text{sign}(\mathbf{w})\big)$, 점수 변화 $= \mathbf{w}^\top\delta$
2. 부호를 구한다 ($\text{sign}(0) = 0$): $\text{sign}(1, -2, 0) = (1, -1, 0)$
3. 첫 걸음을 대입한다: $\mathbf{v}_1 = (0, 0, 0) + 0.06 \times (1, -1, 0)$
4. 곱하고 더한다: $\mathbf{v}_1 = (0.06, -0.06, 0)$
5. 공 안인지 본다: 세 칸 모두 $|v_i| \le 0.1$ 이므로 자를 것이 없다: $\delta_1 = (0.06, -0.06, 0)$
6. 둘째 걸음을 대입한다: $\mathbf{v}_2 = (0.06, -0.06, 0) + 0.06 \times (1, -1, 0)$
7. 곱하고 더한다: $0.06 + 0.06 = 0.12$, $-0.06 + (-0.06) = -0.12$, $0 + 0 = 0$ 이므로 $\mathbf{v}_2 = (0.12, -0.12, 0)$
8. 자른다: $0.12 > 0.1$ 이므로 0.1, $-0.12 < -0.1$ 이므로 $-0.1$, 셋째 칸은 그대로 0: $\delta_2 = (0.1, -0.1, 0)$
9. 점수 변화의 식에 대입한다: $\mathbf{w}^\top\delta_2 = 1 \times 0.1 + (-2) \times (-0.1) + 0 \times 0$
10. 곱한다: $1 \times 0.1 = 0.1$, $(-2) \times (-0.1) = 0.2$, $0 \times 0 = 0$
11. 더한다: $0.1 + 0.2 + 0 = 0.3$

**답**: $\delta_1 = (0.06, -0.06, 0)$, $\delta_2 = (0.1, -0.1, 0)$, 점수 변화 $0.3$. 확인: $\epsilon\lVert\mathbf{w}\rVert_1 = 0.1 \times 3 = 0.3$으로 최악 교란에 닿았다. [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)가 0인 칸은 움직일 이유가 없어 그대로다.

</details>

### 문제 8

점수가 $s = \mathbf{w}^\top\mathbf{x} + b$이고 $\mathbf{w} = (2, -1, 1)$, $\mathbf{x} = (0.5, 0.4, 0.3)$, $b = -0.2$이다. $s > 0$이면 "예", 아니면 "아니오"로 답한다. (a) 지금 점수는? (b) $\epsilon = 0.1$로 점수를 가장 크게 낮추면 답이 바뀌는가? (c) 답을 바꾸는 데 필요한 가장 작은 $\epsilon$은?

<details><summary>풀이와 답</summary>

점수를 가장 크게 낮추는 교란의 점수 변화는 $-\epsilon\lVert\mathbf{w}\rVert_1$이다(부호만 반대로 한 최악 교란).

**주어진 값**
- $\mathbf{w} = (2, -1, 1)$, $\mathbf{x} = (0.5, 0.4, 0.3)$, $b = -0.2$

**풀이**
1. 식을 그대로 쓴다: $s = \mathbf{w}^\top\mathbf{x} + b = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$
2. 값을 대입한다: $s = 2 \times 0.5 + (-1) \times 0.4 + 1 \times 0.3 + (-0.2)$
3. 곱한다: $2 \times 0.5 = 1$, $(-1) \times 0.4 = -0.4$, $1 \times 0.3 = 0.3$
4. 더한다: $1 - 0.4 + 0.3 - 0.2 = 0.7$ 이므로 (a)의 점수는 $0.7$ ("예")
5. (b) $\lVert\mathbf{w}\rVert_1$을 구한다: $2 + 1 + 1 = 4$
6. 점수 변화를 구한다: $-\epsilon\lVert\mathbf{w}\rVert_1 = -0.1 \times 4 = -0.4$
7. 새 점수를 구한다: $0.7 + (-0.4) = 0.3$. $0.3 > 0$ 이므로 답은 그대로 "예"
8. (c) 점수가 0 이하가 되는 조건을 쓴다: $0.7 - \epsilon \times 4 \le 0$
9. $\epsilon$에 대해 푼다: $4\epsilon \ge 0.7$ 이므로 $\epsilon \ge 0.7 / 4 = 0.175$

**답**: (a) $0.7$, (b) 새 점수 $0.3$이라 바뀌지 않는다, (c) $\epsilon = 0.175$부터 바뀐다. $\epsilon = 0.2$이면 새 점수가 $0.7 - 0.8 = -0.1$이다.

</details>

## 강의와 연결

### 문제 9

Lec17 5.1의 돼지 분류기는 점수 $s = 3x_1 - 2x_2 - 0.5$, $\mathbf{x} = (0.6, 0.4)$이고 $P(\text{돼지}) = \sigma(s)$이다. $\sigma(z) = 1/(1 + e^{-z})$ ($e$는 약 2.718인 수). (a) 처음 $P(\text{여객기}) = 1 - P(\text{돼지})$를 구하라. (b) $\epsilon = 0.15$ 안에서 점수를 가장 크게 낮추는 교란을 더한 뒤의 $P(\text{여객기})$를 구하라.

<details><summary>풀이와 답</summary>

여객기 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)을 올리려면 돼지 점수 $s$를 낮추면 된다. 선형 점수의 최악 교란 $\delta = -\epsilon\,\text{sign}(\mathbf{w})$를 쓴다.

**주어진 값**
- $\mathbf{w} = (3, -2)$, $b = -0.5$ (분류기)
- $\mathbf{x} = (0.6, 0.4)$ (사진의 두 픽셀), $\epsilon = 0.15$

**풀이**
1. 식을 그대로 쓴다: $s = 3x_1 - 2x_2 - 0.5$, $P(\text{돼지}) = \sigma(s) = \dfrac{1}{1 + e^{-s}}$
2. 값을 대입한다: $s = 3 \times 0.6 - 2 \times 0.4 - 0.5$
3. 곱한다: $3 \times 0.6 = 1.8$, $2 \times 0.4 = 0.8$ 이므로 $s = 1.8 - 0.8 - 0.5$
4. 계산한다: $s = 0.5$ 이므로 $P(\text{돼지}) = \dfrac{1}{1 + e^{-0.5}}$
5. [지수](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#e-exponent-e%EC%99%80-%EC%A7%80%EC%88%98)를 계산한다 ($e^{-0.5}$는 $1/\sqrt{e}$로 약 $0.6065$, 계산기·코드로 구한다): $P(\text{돼지}) = \dfrac{1}{1 + 0.6065}$
6. 나눈다: $1 / 1.6065 = 0.6225$
7. 1에서 뺀다: $P(\text{여객기}) = 1 - 0.6225 = 0.3775$ (a의 답)
8. (b) 최악 교란을 쓴다. 부호를 구한다: $\text{sign}(3, -2) = (1, -1)$ 이므로 $\delta = -0.15 \times (1, -1) = (-0.15, 0.15)$
9. 교란을 더한다: $\mathbf{x} + \delta = (0.6 - 0.15,\ 0.4 + 0.15) = (0.45, 0.55)$
10. 새 점수를 구한다: $s' = 3 \times 0.45 - 2 \times 0.55 - 0.5 = 1.35 - 1.1 - 0.5 = -0.25$
11. 확인한다: 점수 변화 $-\epsilon\lVert\mathbf{w}\rVert_1 = -0.15 \times 5 = -0.75$ 이므로 $0.5 - 0.75 = -0.25$ 로 같다
12. [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)로 바꾼다 ($e^{0.25} = 1.2840$): $P(\text{돼지}) = 1 / (1 + 1.2840) = 0.4378$
13. 1에서 뺀다: $P(\text{여객기}) = 1 - 0.4378 = 0.5622$

**답**: (a) $0.3775$, (b) $0.5622$. 사람 눈에 안 보이는 $\epsilon = 0.15$ 교란 하나로 [분류](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#regression-classification-%ED%9A%8C%EA%B7%80%EC%99%80-%EB%B6%84%EB%A5%98)가 돼지에서 여객기 쪽으로 뒤집혔다.

</details>

### 문제 10

문제 9의 분류기에서 $\delta = (0, 0)$일 때 $P(\text{여객기})$의 $\delta$에 대한 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 구하고, $\eta = 0.5$로 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 걸음을 한 번 간 뒤 $\epsilon = 0.15$로 투영하라 (Lec17 5.2). $\sigma'(s) = \sigma(s)\big(1 - \sigma(s)\big)$를 쓴다.

<details><summary>풀이와 답</summary>

$P(\text{여객기}) = 1 - \sigma(s)$이고 $s = \mathbf{w}^\top(\mathbf{x} + \delta) + b$이므로, [연쇄법칙](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#chain-rule-%EC%97%B0%EC%87%84%EB%B2%95%EC%B9%99)으로 $\nabla_\delta P(\text{여객기}) = -\sigma'(s)\,\mathbf{w}$이다.

**주어진 값**
- $\mathbf{w} = (3, -2)$, $s = 0.5$ ($\delta = 0$일 때 점수), $P(\text{돼지}) = \sigma(0.5) = 0.6225$
- $\eta = 0.5$ (걸음 크기), $\epsilon = 0.15$ (반지름)

**풀이**
1. 식을 그대로 쓴다: $\nabla_\delta P(\text{여객기}) = -\sigma(s)\big(1 - \sigma(s)\big)\,\mathbf{w}$, $\delta_{\text{new}} = \Pi_\Delta(\delta + \eta\,\nabla)$
2. 값을 대입한다: $\nabla = -0.6225 \times (1 - 0.6225) \times (3, -2)$
3. 괄호를 계산한다: $1 - 0.6225 = 0.3775$ 이므로 $\nabla = -0.6225 \times 0.3775 \times (3, -2)$
4. 앞의 곱셈을 한다: $0.6225 \times 0.3775 = 0.2350$ 이므로 $\nabla = -0.2350 \times (3, -2)$
5. 칸마다 곱한다: $-0.2350 \times 3 = -0.705$, $-0.2350 \times (-2) = 0.47$ 이므로 $\nabla = (-0.705, 0.47)$
6. 걸음 식에 대입한다: $\mathbf{v} = (0, 0) + 0.5 \times (-0.705, 0.47)$
7. $\eta$를 곱한다: $0.5 \times (-0.705) = -0.3525$, $0.5 \times 0.47 = 0.235$ 이므로 $\mathbf{v} = (-0.3525, 0.235)$
8. 첫째 칸을 자른다: $\max(-0.3525, -0.15) = -0.15$, 다음 $\min(-0.15, 0.15) = -0.15$
9. 둘째 칸을 자른다: $\max(0.235, -0.15) = 0.235$, 다음 $\min(0.235, 0.15) = 0.15$

**답**: [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $(-0.705, 0.47)$, 한 걸음 뒤 $\mathbf{v} = (-0.3525, 0.235)$는 공 밖이고, 투영하면 $(-0.15, 0.15)$로 문제 9의 최악 교란과 같은 모서리다.

</details>
