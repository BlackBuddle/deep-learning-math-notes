---
title: 예제 MM70 MAML의 안쪽·바깥쪽 경사하강
type: note
tags: [MIT딥러닝, 예제]
created: 2026-10-15
cssclasses: [dl-note]
---

# 예제: MAML의 안쪽·바깥쪽 경사하강

개념 노트: [MM70_MAML의_안쪽_바깥쪽_경사하강](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/12_%EA%B0%95%EA%B1%B4%EC%84%B1%EA%B3%BC_%EC%A0%84%EC%9D%B4/MM70_MAML%EC%9D%98_%EC%95%88%EC%AA%BD_%EB%B0%94%EA%B9%A5%EC%AA%BD_%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95.md) · [목차](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md)

풀이는 접혀 있으니 먼저 풀어 보고 펼쳐 확인한다. 문제 번호 순서는 쉬운 것에서 어려운 것, 마지막 둘은 강의 내용과 잇는 문제다. 따로 말이 없으면 과제의 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)은 $(\theta - c)^2$이고, 그 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)는 $2(\theta - c)$, 2차 [미분](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#derivative-%EB%AF%B8%EB%B6%84)은 2다.

## 기본 (직접 대입)

### 문제 1
초기값 $\theta = 2$, 과제의 [최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92) $c = 5$, 안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $\alpha = 0.1$이다. 한 걸음 적응한 $\theta'$와, 적응 전후의 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\theta = 2$ (초기값)
- $c = 5$ (과제의 [최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92))
- $\alpha = 0.1$ (안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))

**풀이**
1. 식을 그대로 쓴다: $\theta' = \theta - \alpha\, \nabla_\theta \mathcal{L}(\theta)$, $\mathcal{L}(\theta) = (\theta - c)^2$
2. 값을 대입한다: $\theta' = 2 - 0.1 \times 2(2 - 5)$
3. 괄호를 계산한다: $2 - 5 = -3$ 이므로 $\theta' = 2 - 0.1 \times 2 \times (-3)$
4. [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 곱한다: $2 \times (-3) = -6$ 이므로 $\theta' = 2 - 0.1 \times (-6)$
5. 곱한다: $0.1 \times (-6) = -0.6$ 이므로 $\theta' = 2 - (-0.6)$
6. 뺀다: $2 + 0.6 = 2.6$
7. 적응 전 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)을 구한다: $(2 - 5)^2 = (-3)^2 = 9$
8. 적응 후 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)을 구한다: $(2.6 - 5)^2 = (-2.4)^2 = 5.76$

**답**: $\theta' = $ **2.6**, [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)은 **9에서 5.76**으로 줄었다.

</details>

### 문제 2
안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $\alpha = 0.25$, 과제 $c = 4$, 초기값 $\theta = 0$이다. $\theta'$를 구하고, [최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92)까지 남은 거리가 처음의 몇 배가 되었는지 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\theta = 0$ (초기값)
- $c = 4$ (과제의 [최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92))
- $\alpha = 0.25$ (안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))

**풀이**
1. 식을 그대로 쓴다: $\theta' = \theta - \alpha \times 2(\theta - c)$
2. 값을 대입한다: $\theta' = 0 - 0.25 \times 2(0 - 4)$
3. 괄호를 계산한다: $0 - 4 = -4$ 이므로 $\theta' = 0 - 0.25 \times 2 \times (-4)$
4. 곱한다: $0.25 \times 2 \times (-4) = -2$ 이므로 $\theta' = 0 - (-2)$
5. 뺀다: $0 + 2 = 2$
6. 처음 거리를 구한다: $|0 - 4| = 4$
7. 적응한 뒤 거리를 구한다: $|2 - 4| = 2$
8. 나눈다: $2 / 4 = 0.5$

**답**: $\theta' = $ **2**, 거리는 처음의 **0.5배**. 이 과제에서는 한 걸음에 거리가 $1 - 2\alpha = 0.5$배가 된다.

</details>

### 문제 3
[손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 $3(\theta - c)^2$이고 안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)이 $\alpha = 0.1$이다. 초기값을 1 바꾸면 한 걸음 적응한 결과 $\theta'$는 얼마나 바뀌는가? (이 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)의 2차 [미분](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#derivative-%EB%AF%B8%EB%B6%84)은 $\mathcal{L}'' = 6$이다.)

<details><summary>풀이와 답</summary>

**주어진 값**
- $\mathcal{L}''(\theta) = 6$ ([손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98) $3(\theta - c)^2$의 2차 [미분](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#derivative-%EB%AF%B8%EB%B6%84). 1차 [미분](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#derivative-%EB%AF%B8%EB%B6%84)이 $6(\theta - c)$이므로 한 번 더 [미분](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#derivative-%EB%AF%B8%EB%B6%84))
- $\alpha = 0.1$ (안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))

**풀이**
1. 식을 그대로 쓴다: $\dfrac{d\theta'}{d\theta} = 1 - \alpha\, \mathcal{L}''(\theta)$
2. 값을 대입한다: $\dfrac{d\theta'}{d\theta} = 1 - 0.1 \times 6$
3. 곱셈을 먼저 한다: $0.1 \times 6 = 0.6$ 이므로 $\dfrac{d\theta'}{d\theta} = 1 - 0.6$
4. 뺀다: $1 - 0.6 = 0.4$

**답**: **0.4**. 초기값을 1 옮기면 적응 결과는 0.4만 옮겨진다.

</details>

### 문제 4
[손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 $2(\theta - c)^2$, 안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)이 $\alpha = 0.2$다. 초기값 $\theta = 3$, [최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92) $c = 0$일 때 한 걸음 적응한 $\theta'$를 구하고, 거리가 몇 배가 되었는지 $1 - 2\alpha a$ 식으로 확인하라. ([손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)을 $a(\theta - c)^2$ 꼴로 쓰면 $a = 2$다.)

<details><summary>풀이와 답</summary>

**주어진 값**
- $a = 2$ ([손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)의 곡률 계수), $c = 0$ ([최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92))
- $\theta = 3$ (초기값), $\alpha = 0.2$ (안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))

**풀이**
1. 식을 그대로 쓴다: $\theta' = \theta - \alpha \times 2a(\theta - c)$
2. 값을 대입한다: $\theta' = 3 - 0.2 \times 2 \times 2 \times (3 - 0)$
3. 괄호를 계산한다: $3 - 0 = 3$ 이므로 $\theta' = 3 - 0.2 \times 2 \times 2 \times 3$
4. 곱한다: $0.2 \times 2 \times 2 \times 3 = 2.4$ 이므로 $\theta' = 3 - 2.4$
5. 뺀다: $3 - 2.4 = 0.6$
6. 거리의 배수 식을 쓰고 대입한다: $1 - 2\alpha a = 1 - 2 \times 0.2 \times 2$
7. 곱한다: $2 \times 0.2 \times 2 = 0.8$ 이므로 $1 - 0.8 = 0.2$
8. 확인한다: 처음 거리 3에 0.2를 곱하면 $3 \times 0.2 = 0.6$으로 $\theta'$와 같다.

**답**: $\theta' = $ **0.6**, 거리는 **0.2배**. 곡률이 큰(가파른) 과제는 한 걸음에 [최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92) 가까이 간다.

</details>

## 응용

### 문제 5
두 과제의 [최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92)이 $c_1 = 2$, $c_2 = 4$, 초기값 $\theta = 0$, 안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $\alpha = 0.25$, 바깥쪽(메타) [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $\beta = 0.1$이다. 정식 [MAML](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#maml-model-agnostic-meta-learning-%EB%AA%A8%EB%8D%B8-%EB%AC%B4%EA%B4%80-%EB%A9%94%ED%83%80-%ED%95%99%EC%8A%B5)로 초기값을 한 번 갱신하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\theta = 0$ (초기값), $\alpha = 0.25$ (안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)), $\beta = 0.1$ (바깥쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))
- $c_1 = 2$, $c_2 = 4$ (두 과제의 [최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92))
- [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)의 2차 [미분](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#derivative-%EB%AF%B8%EB%B6%84) $\mathcal{L}'' = 2$ (두 과제 모두)

**풀이**
1. 안쪽 식을 쓴다: $\theta'_i = \theta - \alpha \times 2(\theta - c_i)$
2. 값을 대입한다: $\theta'_1 = 0 - 0.25 \times 2(0 - 2)$, $\theta'_2 = 0 - 0.25 \times 2(0 - 4)$
3. 계산한다: $0.25 \times 2 \times (-2) = -1$, $0.25 \times 2 \times (-4) = -2$ 이므로 $\theta'_1 = 0 + 1 = 1$, $\theta'_2 = 0 + 2 = 2$
4. [연쇄법칙](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#chain-rule-%EC%97%B0%EC%87%84%EB%B2%95%EC%B9%99)의 인자 식을 쓰고 대입한다: $\dfrac{d\theta'}{d\theta} = 1 - \alpha\, \mathcal{L}'' = 1 - 0.25 \times 2 = 1 - 0.5 = 0.5$
5. 바깥쪽 식을 쓴다: $\theta \leftarrow \theta - \beta\,\big(\nabla_\theta \mathcal{L}_1(\theta'_1) + \nabla_\theta \mathcal{L}_2(\theta'_2)\big)$, $\nabla_\theta \mathcal{L}_i(\theta'_i) = 2(\theta'_i - c_i) \times 0.5$
6. 값을 대입한다: $\theta \leftarrow 0 - 0.1\,\big(2(1 - 2) \times 0.5 + 2(2 - 4) \times 0.5\big)$
7. 괄호 안 뺄셈: $1 - 2 = -1$, $2 - 4 = -2$ 이므로 $\theta \leftarrow 0 - 0.1\,\big(2 \times (-1) \times 0.5 + 2 \times (-2) \times 0.5\big)$
8. 각 항을 곱한다: $2 \times (-1) \times 0.5 = -1$, $2 \times (-2) \times 0.5 = -2$ 이므로 $\theta \leftarrow 0 - 0.1 \times (-1 + (-2))$
9. 더한다: $-1 + (-2) = -3$ 이므로 $\theta \leftarrow 0 - 0.1 \times (-3)$
10. 곱하고 뺀다: $0.1 \times (-3) = -0.3$ 이므로 $\theta = 0 + 0.3 = 0.3$

**답**: $\theta = $ **0.3**

</details>

### 문제 6
문제 5와 같은 설정에서, 2차 항을 버린 1차 근사(FOMAML)로 갱신하면 $\theta$는? 정식 [MAML](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#maml-model-agnostic-meta-learning-%EB%AA%A8%EB%8D%B8-%EB%AC%B4%EA%B4%80-%EB%A9%94%ED%83%80-%ED%95%99%EC%8A%B5)의 결과와 비교하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\theta = 0$, $\beta = 0.1$ (초기값, 바깥쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))
- $\theta'_1 = 1$, $\theta'_2 = 2$ (문제 5에서 구한 적응 결과)
- $c_1 = 2$, $c_2 = 4$ ([최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92))

**풀이**
1. 식을 그대로 쓴다: $\theta \leftarrow \theta - \beta \sum_i \mathcal{L}_i'(\theta'_i)$ (인자 $\frac{d\theta'}{d\theta}$를 1로 둔다)
2. Σ를 풀고 값을 대입한다: $\theta \leftarrow 0 - 0.1\,\big(2(1 - 2) + 2(2 - 4)\big)$
3. 괄호 안 뺄셈: $1 - 2 = -1$, $2 - 4 = -2$ 이므로 $\theta \leftarrow 0 - 0.1\,\big(2 \times (-1) + 2 \times (-2)\big)$
4. 곱한다: $2 \times (-1) = -2$, $2 \times (-2) = -4$ 이므로 $\theta \leftarrow 0 - 0.1 \times (-2 + (-4))$
5. 더한다: $-2 + (-4) = -6$ 이므로 $\theta \leftarrow 0 - 0.1 \times (-6)$
6. 곱하고 뺀다: $0.1 \times (-6) = -0.6$ 이므로 $\theta = 0 + 0.6 = 0.6$
7. 비교한다: $0.6 / 0.3 = 2$

**답**: FOMAML은 $\theta = $ **0.6**으로, 정식 [MAML](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#maml-model-agnostic-meta-learning-%EB%AA%A8%EB%8D%B8-%EB%AC%B4%EA%B4%80-%EB%A9%94%ED%83%80-%ED%95%99%EC%8A%B5)(0.3)보다 **2배** 멀리 간다. 인자 0.5를 곱하지 않았기 때문이다.

</details>

### 문제 7
[손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98) $(\theta - 1)^2$에서 초기값 $\theta = 0$, 안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)을 너무 크게 잡아 $\alpha = 1.5$로 했다. 한 걸음 적응한 $\theta'$, 적응 전후의 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98), 그리고 인자 $\frac{d\theta'}{d\theta}$를 구하고 무슨 일이 일어났는지 말하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\theta = 0$ (초기값), $c = 1$ ([최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92))
- $\alpha = 1.5$ (안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0), 너무 크다)
- $\mathcal{L}'' = 2$ (2차 [미분](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#derivative-%EB%AF%B8%EB%B6%84))

**풀이**
1. 식을 그대로 쓴다: $\theta' = \theta - \alpha \times 2(\theta - c)$
2. 값을 대입한다: $\theta' = 0 - 1.5 \times 2(0 - 1)$
3. 괄호를 계산한다: $0 - 1 = -1$ 이므로 $\theta' = 0 - 1.5 \times 2 \times (-1)$
4. 곱한다: $1.5 \times 2 \times (-1) = -3$ 이므로 $\theta' = 0 - (-3)$
5. 뺀다: $0 + 3 = 3$
6. 적응 전 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98): $(0 - 1)^2 = 1$
7. 적응 후 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98): $(3 - 1)^2 = 2^2 = 4$
8. 인자 식을 쓰고 대입한다: $\dfrac{d\theta'}{d\theta} = 1 - \alpha\, \mathcal{L}'' = 1 - 1.5 \times 2$
9. 곱하고 뺀다: $1.5 \times 2 = 3$ 이므로 $1 - 3 = -2$

**답**: $\theta' = $ **3**, [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)은 **1에서 4**로 오히려 커졌고 인자는 **-2**다. 한 걸음이 [최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92) 1을 훌쩍 넘어가 버렸다(넘쳐 흐름). $1 - \alpha\mathcal{L}''$가 음수가 되는 것이 신호다.

</details>

### 문제 8
두 과제의 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 $\mathcal{L}_1 = 1 \times (\theta - 0)^2$, $\mathcal{L}_2 = 2 \times (\theta - 3)^2$이고 안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)은 $\alpha = 0.25$다. (a) 적응 전 평균 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)을 최소로 하는 초기값([사전 학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pre-training-%EC%82%AC%EC%A0%84-%ED%95%99%EC%8A%B5))을 구하라. (b) 한 걸음 적응한 뒤 과제 $i$의 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)은 $a_i (1 - 2\alpha a_i)^2 (\theta - c_i)^2$이다. 이 식으로 [사전 학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pre-training-%EC%82%AC%EC%A0%84-%ED%95%99%EC%8A%B5) 초기값과 $\theta = 0$의 적응 뒤 평균 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)을 구하고 비교하라. (평균 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)을 최소로 하는 초기값은 $\sum a_i c_i / \sum a_i$이다.)

<details><summary>풀이와 답</summary>

**주어진 값**
- $a_1 = 1$, $c_1 = 0$, $a_2 = 2$, $c_2 = 3$ (곡률 계수와 [최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92))
- $\alpha = 0.25$ (안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))

**풀이**
1. (a) 식을 그대로 쓴다: $\theta_{\text{pre}} = \dfrac{\sum_i a_i c_i}{\sum_i a_i}$
2. 값을 대입한다: $\theta_{\text{pre}} = \dfrac{1 \times 0 + 2 \times 3}{1 + 2}$
3. 분자와 분모를 계산한다: $1 \times 0 + 2 \times 3 = 6$, $1 + 2 = 3$ 이므로 $\theta_{\text{pre}} = 6 / 3 = 2$
4. (b) 거리의 배수를 구한다: $1 - 2\alpha a_1 = 1 - 2 \times 0.25 \times 1 = 0.5$, $1 - 2\alpha a_2 = 1 - 2 \times 0.25 \times 2 = 0$
5. 가파른 과제 2는 한 걸음에 [최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92)에 닿는다 (배수 0). 그래서 적응 뒤 과제 2의 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)은 $\theta$와 상관없이 0이다.
6. $\theta = 2$에서 과제 1의 적응 뒤 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98): $1 \times 0.5^2 \times (2 - 0)^2 = 1 \times 0.25 \times 4 = 1$
7. $\theta = 2$에서 평균을 낸다: $(1 + 0) / 2 = 0.5$
8. $\theta = 0$에서 과제 1의 적응 뒤 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98): $1 \times 0.5^2 \times (0 - 0)^2 = 0$
9. $\theta = 0$에서 평균을 낸다: $(0 + 0) / 2 = 0$

**답**: (a) [사전 학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pre-training-%EC%82%AC%EC%A0%84-%ED%95%99%EC%8A%B5) 초기값은 **2**. (b) 적응 뒤 평균 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)은 $\theta = 2$에서 **0.5**, $\theta = 0$에서 **0**. 적응 뒤 성능만 보면 완만한 과제 1의 [최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92) 0에 가까운 초기값이 낫다.

</details>

## 강의와 연결

### 문제 9
[Lec19 9.1](../../Lec19_%EC%A0%84%EC%9D%B4_%ED%95%99%EC%8A%B5_%EB%8D%B0%EC%9D%B4%ED%84%B0.md#91-maml-%EC%9E%98-%EC%A0%81%EC%9D%91%ED%95%98%EB%8A%94-%EC%B4%88%EA%B8%B0%EA%B0%92-%EB%B0%B0%EC%9A%B0%EA%B8%B0)에서 강사는 "[역전파](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#backpropagation-%EC%97%AD%EC%A0%84%ED%8C%8C)를 통과해 역전파한다"고 말한다. 안쪽 고리를 $K$걸음 펼치면 $\frac{d\theta'_K}{d\theta}$는 걸음마다의 인자 $1 - 2\alpha a$를 $K$번 곱한 값이 된다. $a = 1$, $\alpha = 0.25$일 때 $K = 3$걸음의 인자를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $a = 1$ ([손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98) $(\theta - c)^2$의 곡률 계수)
- $\alpha = 0.25$ (안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))
- $K = 3$ (안쪽 걸음 수)

**풀이**
1. 식을 그대로 쓴다: $\dfrac{d\theta'_K}{d\theta} = (1 - 2\alpha a)^{K}$ ([연쇄법칙](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#chain-rule-%EC%97%B0%EC%87%84%EB%B2%95%EC%B9%99): 한 걸음의 인자를 $K$번 곱한다)
2. 값을 대입한다: $\dfrac{d\theta'_3}{d\theta} = (1 - 2 \times 0.25 \times 1)^{3}$
3. 괄호 안을 계산한다: $2 \times 0.25 \times 1 = 0.5$, $1 - 0.5 = 0.5$ 이므로 $0.5^3$
4. 두 번 곱한다: $0.5 \times 0.5 = 0.25$
5. 한 번 더 곱한다: $0.25 \times 0.5 = 0.125$

**답**: **0.125**. 한 걸음이면 0.5였다. 걸음 수가 늘수록 인자를 곱하는 단계도, 저장할 forward·backward도 걸음 수에 비례해 늘어나서 비싸다.

</details>

### 문제 10
[Lec19 9.1](../../Lec19_%EC%A0%84%EC%9D%B4_%ED%95%99%EC%8A%B5_%EB%8D%B0%EC%9D%B4%ED%84%B0.md#91-maml-%EC%9E%98-%EC%A0%81%EC%9D%91%ED%95%98%EB%8A%94-%EC%B4%88%EA%B8%B0%EA%B0%92-%EB%B0%B0%EC%9A%B0%EA%B8%B0)의 아이디어를 숫자로 확인한다. 과제는 $\mathcal{L}_1 = 1 \times (\theta - 0)^2$, $\mathcal{L}_2 = 2 \times (\theta - 3)^2$, 안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $\alpha = 0.2$다. [MAML](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#maml-model-agnostic-meta-learning-%EB%AA%A8%EB%8D%B8-%EB%AC%B4%EA%B4%80-%EB%A9%94%ED%83%80-%ED%95%99%EC%8A%B5) 초기값 $\theta = 0.5455$에서 한 걸음 적응한 뒤의 평균 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98) $\bar{\mathcal{L}} = \frac{1}{2}\sum_i a_i(\theta'_i - c_i)^2$을 구하고, [사전 학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pre-training-%EC%82%AC%EC%A0%84-%ED%95%99%EC%8A%B5) 초기값 $\theta = 2$의 값 0.76과 비교하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\theta = 0.5455$ ([MAML](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#maml-model-agnostic-meta-learning-%EB%AA%A8%EB%8D%B8-%EB%AC%B4%EA%B4%80-%EB%A9%94%ED%83%80-%ED%95%99%EC%8A%B5) 초기값)
- $a_1 = 1$, $c_1 = 0$, $a_2 = 2$, $c_2 = 3$ (곡률 계수와 [최적값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#optimal-value-%EC%B5%9C%EC%A0%81%EA%B0%92))
- $\alpha = 0.2$ (안쪽 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))

**풀이**
1. 식을 그대로 쓴다: $\theta'_i = \theta - \alpha \times 2a_i(\theta - c_i)$, $\bar{\mathcal{L}} = \frac{1}{2}\sum_i a_i(\theta'_i - c_i)^2$
2. 값을 대입한다: $\theta'_1 = 0.5455 - 0.2 \times 2 \times 1 \times (0.5455 - 0)$, $\theta'_2 = 0.5455 - 0.2 \times 2 \times 2 \times (0.5455 - 3)$
3. 괄호를 계산한다: $0.5455 - 0 = 0.5455$, $0.5455 - 3 = -2.4545$
4. 곱한다: $0.2 \times 2 \times 1 \times 0.5455 = 0.2182$, $0.2 \times 2 \times 2 \times (-2.4545) = -1.9636$
5. 적응 결과를 구한다: $\theta'_1 = 0.5455 - 0.2182 = 0.3273$, $\theta'_2 = 0.5455 + 1.9636 = 2.5091$
6. 최적값과의 차: $0.3273 - 0 = 0.3273$, $2.5091 - 3 = -0.4909$
7. 제곱한다: $0.3273^2 = 0.1071$, $(-0.4909)^2 = 0.2410$
8. 곡률을 곱한다: $1 \times 0.1071 = 0.1071$, $2 \times 0.2410 = 0.4820$
9. Σ를 풀어 더한다: $0.1071 + 0.4820 = 0.5891$
10. 2로 나눈다: $0.5891 / 2 = 0.2945$ (반올림 전 값은 0.2945)
11. 비교한다: [사전 학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#pre-training-%EC%82%AC%EC%A0%84-%ED%95%99%EC%8A%B5) 초기값 $\theta = 2$의 0.76보다 작다.

**답**: **약 0.2945**. 같은 한 걸음 적응이라도 [MAML](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#maml-model-agnostic-meta-learning-%EB%AA%A8%EB%8D%B8-%EB%AC%B4%EA%B4%80-%EB%A9%94%ED%83%80-%ED%95%99%EC%8A%B5) 초기값이 적응 뒤 평균 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)을 0.76에서 0.2945로 줄인다.

</details>
