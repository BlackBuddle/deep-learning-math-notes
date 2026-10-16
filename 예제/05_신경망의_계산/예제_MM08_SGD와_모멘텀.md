---
title: 예제 MM08 SGD와 모멘텀
type: note
tags: [MIT딥러닝, 예제]
created: 2026-10-15
cssclasses: [dl-note]
---

# 예제: 미니배치 기울기와 모멘텀

개념 노트: [MM08_SGD와_모멘텀](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/05_%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%9D%98_%EA%B3%84%EC%82%B0/MM08_SGD%EC%99%80_%EB%AA%A8%EB%A9%98%ED%85%80.md) · [목차](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md)

풀이는 접혀 있으니 먼저 직접 풀어 보고, 펼쳐서 확인한다.

## 기본 (직접 대입)

### 문제 1
데이터 $x = (2, 4, 6, 8)$, 데이터 하나의 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98) $J_i = (\theta - x_i)^2$ (그 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)는 $\frac{\partial J_i}{\partial \theta} = 2(\theta - x_i)$)일 때 $\theta = 0$에서 전체 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 구하라.

<details><summary>풀이와 답</summary>

전체 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)는 데이터별 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)의 평균이다.

**주어진 값**
- $x_1 = 2,\ x_2 = 4,\ x_3 = 6,\ x_4 = 8$ (데이터)
- $N = 4$ (데이터 개수)
- $\theta = 0$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 재는 위치)

**풀이**
1. 식을 그대로 쓴다: $\frac{\partial J}{\partial \theta} = \frac{1}{N}\sum_{i=1}^{N} 2(\theta - x_i)$
2. 값을 대입한다 (Σ를 네 항으로 펼침): $\frac{1}{4}\big(2(0 - 2) + 2(0 - 4) + 2(0 - 6) + 2(0 - 8)\big)$
3. 괄호 안 뺄셈을 한다: $\frac{1}{4}\big(2 \times (-2) + 2 \times (-4) + 2 \times (-6) + 2 \times (-8)\big)$
4. 곱셈을 한다: $\frac{1}{4}\big((-4) + (-8) + (-12) + (-16)\big)$
5. 더한다: −4 − 8 − 12 − 16 = −40 이므로 $\frac{1}{4} \times (-40)$
6. 4로 나눈다: −40 ÷ 4 = −10

**답**: 전체 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) = −10 (음수라 θ를 늘려야 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 준다)

</details>

### 문제 2
문제 1의 데이터를 두 개씩 나눈 배치 {2, 4}와 {6, 8}의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 각각 구하고, 두 배치 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)의 평균이 전체 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)와 같은지 확인하라. ($\theta = 0$)

<details><summary>풀이와 답</summary>

**주어진 값**
- 배치 A = {2, 4}, 배치 B = {6, 8} ([미니배치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#mini-batch-%EB%AF%B8%EB%8B%88%EB%B0%B0%EC%B9%98))
- $\theta = 0$
- 문제 1에서 구한 전체 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) −10

**풀이**
1. 식을 그대로 쓴다: 배치 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $= \frac{1}{|B|}\sum_{i \in B} 2(\theta - x_i)$ (B의 데이터 수로 나눈 평균)
2. 배치 A에 대입한다: $\frac{1}{2}\big(2(0 - 2) + 2(0 - 4)\big)$
3. 곱셈을 한다: $\frac{1}{2}\big((-4) + (-8)\big)$
4. 더하고 나눈다: −4 − 8 = −12, −12 ÷ 2 = −6
5. 배치 B에 대입한다: $\frac{1}{2}\big(2(0 - 6) + 2(0 - 8)\big)$
6. 곱셈을 한다: $\frac{1}{2}\big((-12) + (-16)\big)$
7. 더하고 나눈다: −12 − 16 = −28, −28 ÷ 2 = −14
8. 두 배치 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)의 평균: (−6 + (−14)) ÷ 2 = −20 ÷ 2 = −10

**답**: 배치 A = −6, 배치 B = −14. 평균은 −10으로 전체 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)와 같다. 배치 하나는 전체와 다르다(잡음).

</details>

### 문제 3
데이터 $x = (2, 4, 6, 8)$, $\theta = 0$, [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $\eta = 0.1$에서 (가) 전체 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)(−10)로 한 번 갱신한 θ와 (나) 배치 {2, 4}의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)(−6)로 한 번 갱신한 θ를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\theta = 0$ (지금 [파라미터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0))
- $\eta = 0.1$ ([학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))
- 전체 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) −10, 배치 {2, 4}의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) −6 (문제 1, 2)

**풀이**
1. 식을 그대로 쓴다: $\theta \leftarrow \theta - \eta \nabla J$
2. (가) 값을 대입한다: $\theta \leftarrow 0 - 0.1 \times (-10)$
3. 곱셈을 한다: 0.1 × (−10) = −1 이므로 $\theta \leftarrow 0 - (-1)$
4. 음수를 빼는 것은 더하는 것이다: $\theta = 0 + 1 = 1$
5. (나) 값을 대입한다: $\theta \leftarrow 0 - 0.1 \times (-6)$
6. 곱셈을 한다: 0.1 × (−6) = −0.6 이므로 $\theta \leftarrow 0 - (-0.6)$
7. 음수를 빼는 것은 더하는 것이다: $\theta = 0 + 0.6 = 0.6$

**답**: 전체 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)로는 θ = 1, 배치 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)로는 θ = 0.6. 데이터의 평균 4가 바닥이라, 둘 다 바닥 쪽으로 가지만 배치 쪽이 덜 간다.

</details>

### 문제 4
$J(\theta) = \theta^2/2$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $= \theta$)에서 $\theta = 2$, 직전 속도 $v = 0.5$, $\alpha = 0.5$, $\eta = 0.1$일 때 [모멘텀](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#momentum-%EB%AA%A8%EB%A9%98%ED%85%80)으로 한 번 갱신한 새 속도와 θ를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\theta = 2$ (지금 [파라미터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0))
- $v = 0.5$ (직전 속도)
- $\alpha = 0.5$ ([모멘텀](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#momentum-%EB%AA%A8%EB%A9%98%ED%85%80) 세기)
- $\eta = 0.1$ ([학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))
- $\nabla J(\theta) = \theta = 2$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))

**풀이**
1. 식을 그대로 쓴다: $v \leftarrow \alpha v - \eta \nabla J(\theta), \quad \theta \leftarrow \theta + v$
2. 값을 대입한다: $v \leftarrow 0.5 \times 0.5 - 0.1 \times 2, \quad \theta \leftarrow 2 + v$
3. 곱셈을 한다: 0.5 × 0.5 = 0.25, 0.1 × 2 = 0.2 이므로 $v \leftarrow 0.25 - 0.2$
4. 뺄셈을 한다: 0.25 − 0.2 = 0.05 이므로 $v = 0.05$
5. 속도만큼 움직인다: $\theta \leftarrow 2 + 0.05 = 2.05$

**답**: 새 속도 $v = 0.05$, $\theta = 2.05$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)는 θ를 줄이라고 하지만 직전 속도가 양수라 오히려 조금 늘었다. 이것이 관성이다)

</details>

## 응용

### 문제 5
$J(\theta) = \theta^2/2$, 출발 $\theta_0 = 4$, $v_0 = 0$, $\alpha = 0.5$, $\eta = 0.1$에서 [모멘텀](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#momentum-%EB%AA%A8%EB%A9%98%ED%85%80)으로 세 번 갱신한 $\theta_3$를 구하고, [모멘텀](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#momentum-%EB%AA%A8%EB%A9%98%ED%85%80) 없이($\alpha = 0$) 세 번 갱신한 값과 비교하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\theta_0 = 4$ (출발점), $v_0 = 0$ (처음 속도)
- $\alpha = 0.5$, $\eta = 0.1$
- $\nabla J(\theta) = \theta$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))

**풀이**
1. 식을 그대로 쓴다: $v \leftarrow \alpha v - \eta \theta, \quad \theta \leftarrow \theta + v$
2. 1번째 값을 대입한다: $v_1 = 0.5 \times 0 - 0.1 \times 4, \quad \theta_1 = 4 + v_1$
3. 계산한다: 0.5 × 0 = 0, 0.1 × 4 = 0.4 이므로 $v_1 = -0.4$, $\theta_1 = 4 - 0.4 = 3.6$
4. 2번째 값을 대입한다: $v_2 = 0.5 \times (-0.4) - 0.1 \times 3.6, \quad \theta_2 = 3.6 + v_2$
5. 곱셈을 한다: 0.5 × (−0.4) = −0.2, 0.1 × 3.6 = 0.36 이므로 $v_2 = -0.2 - 0.36$
6. 뺄셈을 한다: −0.2 − 0.36 = −0.56 이므로 $\theta_2 = 3.6 - 0.56 = 3.04$
7. 3번째 값을 대입한다: $v_3 = 0.5 \times (-0.56) - 0.1 \times 3.04, \quad \theta_3 = 3.04 + v_3$
8. 곱셈을 한다: 0.5 × (−0.56) = −0.28, 0.1 × 3.04 = 0.304 이므로 $v_3 = -0.28 - 0.304$
9. 뺄셈을 한다: −0.28 − 0.304 = −0.584 이므로 $\theta_3 = 3.04 - 0.584 = 2.456$
10. [모멘텀](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#momentum-%EB%AA%A8%EB%A9%98%ED%85%80) 없을 때: $\theta \leftarrow \theta - 0.1\theta = 0.9\theta$ 이므로 세 번이면 $4 \times 0.9 \times 0.9 \times 0.9$
11. 계산한다: 4 × 0.9 = 3.6, 3.6 × 0.9 = 3.24, 3.24 × 0.9 = 2.916

**답**: [모멘텀](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#momentum-%EB%AA%A8%EB%A9%98%ED%85%80) 있을 때 $\theta_3 = 2.456$, 없을 때 2.916. 바닥(0)에 [모멘텀](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#momentum-%EB%AA%A8%EB%A9%98%ED%85%80) 쪽이 더 가깝다.

</details>

### 문제 6
[기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $g = (12, -16)$, 한도 $m = 10$일 때 칸별 클리핑과 [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84) 클리핑 결과를 각각 구하고, 방향이 유지되는 쪽을 말하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $g = (12, -16)$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 벡터)
- $m = 10$ (한도)

**풀이**
1. 칸별 식을 쓴다: $\mathrm{clip}(v, -m, m)$ (−m보다 작으면 −m, m보다 크면 m, 아니면 그대로)
2. 값을 대입한다: $\mathrm{clip}(12, -10, 10)$, $\mathrm{clip}(-16, -10, 10)$
3. 자른다: 12는 10보다 크므로 10, −16은 −10보다 작으므로 −10 이므로 (10, −10)
4. 길이 식을 쓴다: $\|g\| = \sqrt{g_1^2 + g_2^2}$ (√는 제곱하면 안의 수가 되는 양수)
5. 값을 대입한다: $\sqrt{12^2 + (-16)^2}$
6. 제곱을 한다: 12² = 144, (−16)² = 256 이므로 $\sqrt{144 + 256}$
7. 더하고 제곱근을 구한다: 144 + 256 = 400, 20 × 20 = 400 이므로 $\|g\| = 20$
8. [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84) 클리핑 식을 쓴다: $g \cdot \min\!\left(1, \frac{m}{\|g\|}\right)$
9. 값을 대입한다: $(12, -16) \cdot \min\!\left(1, \frac{10}{20}\right)$
10. 배율을 구한다: 10 ÷ 20 = 0.5 이므로 min(1, 0.5) = 0.5
11. 칸마다 곱한다: 12 × 0.5 = 6, −16 × 0.5 = −8 이므로 (6, −8)
12. 방향을 비교한다: 원래 12 ÷ (−16) = −0.75, [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84) 6 ÷ (−8) = −0.75 (같음), 칸별 10 ÷ (−10) = −1 (다름)

**답**: 칸별 (10, −10), [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84) (6, −8). 방향이 유지되는 쪽은 [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84) 클리핑(길이만 20에서 10으로 줄었다).

</details>

### 문제 7
[기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $g = (3, 4)$, 한도 $m = 10$일 때 [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84) 클리핑 $g \cdot \min(1, m/\|g\|)$의 결과를 구하라. 왜 $\min(1, \cdot)$이 필요한가?

<details><summary>풀이와 답</summary>

**주어진 값**
- $g = (3, 4)$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 벡터)
- $m = 10$ (한도)

**풀이**
1. 길이 식을 쓴다: $\|g\| = \sqrt{g_1^2 + g_2^2}$
2. 값을 대입한다: $\sqrt{3^2 + 4^2}$
3. 제곱을 한다: 3² = 9, 4² = 16 이므로 $\sqrt{9 + 16}$
4. 더하고 제곱근을 구한다: 9 + 16 = 25, 5 × 5 = 25 이므로 $\|g\| = 5$
5. 식을 그대로 쓴다: $g \cdot \min\!\left(1, \frac{m}{\|g\|}\right)$
6. 값을 대입한다: $(3, 4) \cdot \min\!\left(1, \frac{10}{5}\right)$
7. 나눗셈을 한다: 10 ÷ 5 = 2 이므로 $\min(1, 2)$
8. 작은 쪽을 고른다: min(1, 2) = 1
9. 곱한다: $(3 \times 1,\ 4 \times 1) = (3, 4)$

**답**: (3, 4) 그대로. 길이 5가 한도 10보다 짧아 줄일 필요가 없다. min(1, ·)이 없으면 배율 2가 곱해져 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 오히려 커진다. 클리핑은 한도를 **넘을 때만** 줄인다.

</details>

### 문제 8
$J(\theta) = \theta^2/2$, $\theta_0 = 1$, $v_0 = 0$, $\eta = 0.1$에서 $\alpha = 0.9$로 세 번 갱신한 $\theta_3$를 구하고, 본문의 $\alpha = 0.5$ 결과($\theta_3 = 0.614$)와 비교하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\theta_0 = 1$, $v_0 = 0$
- $\alpha = 0.9$ (큰 [모멘텀](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#momentum-%EB%AA%A8%EB%A9%98%ED%85%80)), $\eta = 0.1$
- $\nabla J(\theta) = \theta$

**풀이**
1. 식을 그대로 쓴다: $v \leftarrow \alpha v - \eta \theta, \quad \theta \leftarrow \theta + v$
2. 1번째 값을 대입한다: $v_1 = 0.9 \times 0 - 0.1 \times 1, \quad \theta_1 = 1 + v_1$
3. 계산한다: 0.9 × 0 = 0, 0.1 × 1 = 0.1 이므로 $v_1 = -0.1$, $\theta_1 = 1 - 0.1 = 0.9$
4. 2번째 값을 대입한다: $v_2 = 0.9 \times (-0.1) - 0.1 \times 0.9, \quad \theta_2 = 0.9 + v_2$
5. 곱셈을 한다: 0.9 × (−0.1) = −0.09, 0.1 × 0.9 = 0.09 이므로 $v_2 = -0.09 - 0.09$
6. 뺄셈을 한다: −0.09 − 0.09 = −0.18 이므로 $\theta_2 = 0.9 - 0.18 = 0.72$
7. 3번째 값을 대입한다: $v_3 = 0.9 \times (-0.18) - 0.1 \times 0.72, \quad \theta_3 = 0.72 + v_3$
8. 곱셈을 한다: 0.9 × (−0.18) = −0.162, 0.1 × 0.72 = 0.072 이므로 $v_3 = -0.162 - 0.072$
9. 뺄셈을 한다: −0.162 − 0.072 = −0.234 이므로 $\theta_3 = 0.72 - 0.234 = 0.486$

**답**: $\theta_3 = 0.486$ ($\alpha = 0.5$의 0.614보다 더 내려왔다). 하지만 속도 −0.234가 크게 남아 있어 바닥 0을 지나쳐 −0.6까지 갔다가 되돌아온다. [모멘텀](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#momentum-%EB%AA%A8%EB%A9%98%ED%85%80)은 클수록 좋은 것이 아니다.

</details>

## 강의와 연결

### 문제 9
[Lec02 2.2](../../Lec02_%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%95%99%EC%8A%B5.md#22-%ED%99%95%EB%A5%A0%EC%A0%81-%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95)의 SGD 잡음. 데이터 $x = (1, 2, 3, 6)$, $J_i = (\theta - x_i)^2$일 때 $\theta = 3$에서 전체 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)와 배치 {1, 2}, {3, 6}의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 구하라. 전체 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 0이면 바닥인데 배치 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)는 어떤가?

<details><summary>풀이와 답</summary>

**주어진 값**
- $x = (1, 2, 3, 6)$ (데이터), $N = 4$
- $\theta = 3$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 재는 위치)
- $\frac{\partial J_i}{\partial \theta} = 2(\theta - x_i)$ (데이터별 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))

**풀이**
1. 식을 그대로 쓴다: 데이터별 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $2(\theta - x_i)$
2. 값을 대입한다: $2(3 - 1),\ 2(3 - 2),\ 2(3 - 3),\ 2(3 - 6)$
3. 괄호 안 뺄셈을 한다: $2 \times 2,\ 2 \times 1,\ 2 \times 0,\ 2 \times (-3)$
4. 곱셈을 한다: 4, 2, 0, −6
5. 전체 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)(평균)를 구한다: (4 + 2 + 0 − 6) ÷ 4 = 0 ÷ 4 = 0
6. 배치 {1, 2}를 구한다: (4 + 2) ÷ 2 = 6 ÷ 2 = 3
7. 배치 {3, 6}을 구한다: (0 − 6) ÷ 2 = −6 ÷ 2 = −3

**답**: 전체 0, 배치 {1, 2} = 3, 배치 {3, 6} = −3. 바닥(θ = 3, 데이터의 평균)에서도 배치 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)는 0이 아니어서, SGD는 바닥 근처에서 계속 흔들린다. 이것이 "잡음"이다.

</details>

### 문제 10
[Lec02 3.2](../../Lec02_%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%95%99%EC%8A%B5.md#32-%EC%A7%84%ED%99%94-%EC%A0%84%EB%9E%B5%EA%B3%BC-%EA%B7%B8%EB%9E%98%EB%94%94%EC%96%B8%ED%8A%B8-%ED%81%B4%EB%A6%AC%ED%95%91)의 [기울기 폭주](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#exploding-gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0-%ED%8F%AD%EC%A3%BC). [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 $g = (-60, 80)$으로 폭주했다. 한도 $m = 5$의 [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84) 클리핑을 한 뒤, [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $\eta = 0.1$로 한 걸음 움직이는 크기를 클리핑 전과 비교하라.

<details><summary>풀이와 답</summary>

한 걸음의 이동량은 $\eta g$이고, 그 길이는 $\eta \|g\|$다.

**주어진 값**
- $g = (-60, 80)$ (폭주한 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))
- $m = 5$ (한도)
- $\eta = 0.1$ ([학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))

**풀이**
1. 길이 식을 쓴다: $\|g\| = \sqrt{g_1^2 + g_2^2}$
2. 값을 대입한다: $\sqrt{(-60)^2 + 80^2}$
3. 제곱을 한다: (−60)² = 3600, 80² = 6400 이므로 $\sqrt{3600 + 6400}$
4. 더하고 제곱근을 구한다: 3600 + 6400 = 10000, 100 × 100 = 10000 이므로 $\|g\| = 100$
5. 클리핑 식을 쓴다: $g \cdot \min\!\left(1, \frac{m}{\|g\|}\right)$
6. 값을 대입한다: $(-60, 80) \cdot \min\!\left(1, \frac{5}{100}\right)$
7. 배율을 구한다: 5 ÷ 100 = 0.05 이므로 min(1, 0.05) = 0.05
8. 칸마다 곱한다: −60 × 0.05 = −3, 80 × 0.05 = 4 이므로 클리핑한 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) (−3, 4)
9. 클리핑 전 이동량의 길이: $\eta \|g\| = 0.1 \times 100 = 10$
10. 클리핑 후 이동량의 길이: $\eta \|(-3, 4)\| = 0.1 \times \sqrt{9 + 16} = 0.1 \times 5 = 0.5$

**답**: 클리핑한 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)는 (−3, 4)로 방향은 그대로이고 길이만 100에서 5로 줄었다. 한 걸음 크기는 10에서 0.5가 되어 폭주해도 [파라미터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0)가 멀리 튀지 않는다.

</details>
