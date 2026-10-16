---
tags: [MIT딥러닝, 예제]
cssclasses: [dl-note]
---

# 예제: Adam 업데이트 규칙

개념 노트: [MM83_Adam_업데이트_규칙](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/05_%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%9D%98_%EA%B3%84%EC%82%B0/MM83_Adam_%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8_%EA%B7%9C%EC%B9%99.md) · [목차](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md)

풀이는 접혀 있으니 먼저 풀어 보고 펼쳐 확인한다. 별말이 없으면 $\beta_1 = 0.9$, $\beta_2 = 0.999$, $m_0 = v_0 = 0$이고, $\epsilon = 0.00000001$은 아주 작아서 계산에서는 무시한다. 쓰는 식은 $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, $\hat m_t = m_t / (1-\beta_1^t)$, $\hat v_t = v_t / (1-\beta_2^t)$, $\theta_t = \theta_{t-1} - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)$이다.

## 기본 (직접 대입)

### 문제 1

[Adam](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#adam-%EC%95%84%EB%8B%B4-%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%80)의 첫 걸음($t = 1$)에서 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 $g_1 = 3$으로 나왔다. $m_1$, $v_1$과 바로잡은 $\hat m_1$, $\hat v_1$을 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $t = 1$ (첫 걸음), $g_1 = 3$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))
- $m_0 = 0$, $v_0 = 0$ (시작 값)
- $\beta_1 = 0.9$, $\beta_2 = 0.999$ (감쇠 계수)

**풀이**
1. 식을 그대로 쓴다: $m_1 = \beta_1 m_0 + (1-\beta_1) g_1$, $v_1 = \beta_2 v_0 + (1-\beta_2) g_1^2$, $\hat m_1 = \dfrac{m_1}{1-\beta_1^{1}}$, $\hat v_1 = \dfrac{v_1}{1-\beta_2^{1}}$
2. 값을 대입한다: $m_1 = 0.9 \times 0 + (1 - 0.9) \times 3$, $v_1 = 0.999 \times 0 + (1 - 0.999) \times 3^2$
3. 괄호 안 뺄셈과 제곱을 먼저 한다: $1 - 0.9 = 0.1$, $1 - 0.999 = 0.001$, $3^2 = 9$ 이므로 $m_1 = 0.9 \times 0 + 0.1 \times 3$, $v_1 = 0.999 \times 0 + 0.001 \times 9$
4. 곱셈을 한다: $0.9 \times 0 = 0$, $0.1 \times 3 = 0.3$, $0.999 \times 0 = 0$, $0.001 \times 9 = 0.009$ 이므로 $m_1 = 0 + 0.3$, $v_1 = 0 + 0.009$
5. 더한다: $m_1 = 0.3$, $v_1 = 0.009$
6. 보정 분모를 계산한다: $1 - 0.9^1 = 1 - 0.9 = 0.1$, $1 - 0.999^1 = 0.001$
7. 나눈다: $\hat m_1 = 0.3 \div 0.1 = 3$, $\hat v_1 = 0.009 \div 0.001 = 9$

**답**: $m_1 = 0.3$, $v_1 = 0.009$, $\hat m_1 = 3$, $\hat v_1 = 9$ (바로잡으면 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 3과 그 제곱 9가 그대로 돌아온다)

</details>

### 문제 2

문제 1에서 $\hat m_1 = 3$, $\hat v_1 = 9$를 얻었다. [파라미터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0)가 $\theta_0 = 2$이고 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)이 $\eta = 0.05$이면 한 걸음 뒤 $\theta_1$은?

<details><summary>풀이와 답</summary>

**주어진 값**
- $\theta_0 = 2$ (지금 [파라미터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0))
- $\eta = 0.05$ ([학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))
- $\hat m_1 = 3$, $\hat v_1 = 9$ (바로잡은 값)

**풀이**
1. 식을 그대로 쓴다: $\theta_1 = \theta_0 - \eta\, \dfrac{\hat m_1}{\sqrt{\hat v_1} + \epsilon}$
2. 값을 대입한다: $\theta_1 = 2 - 0.05 \times \dfrac{3}{\sqrt{9} + 0.00000001}$
3. 제곱근을 구한다 ($\sqrt{a}$는 제곱해서 $a$가 되는 양수. $3 \times 3 = 9$): $\sqrt{9} = 3$ 이므로 $\theta_1 = 2 - 0.05 \times \dfrac{3}{3 + 0.00000001}$
4. 분모를 더하고 나눈다: $3 + 0.00000001 \approx 3$ 이므로 $3 \div 3 = 1$ 이고 $\theta_1 = 2 - 0.05 \times 1$
5. 곱한다: $0.05 \times 1 = 0.05$ 이므로 $\theta_1 = 2 - 0.05$
6. 뺀다: $2 - 0.05 = 1.95$

**답**: $\theta_1 = 1.95$ (걸음 크기가 정확히 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $0.05$)

</details>

### 문제 3

[기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 $g = 0.2$인 첫 걸음에서 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $\eta = 0.1$일 때 [Adam](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#adam-%EC%95%84%EB%8B%B4-%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%80)의 걸음 크기와, 같은 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)의 SGD(걸음 $= \eta g$)의 걸음 크기를 비교하라. (첫 걸음에서는 $\hat m_1 = g$, $\hat v_1 = g^2$이다.)

<details><summary>풀이와 답</summary>

**주어진 값**
- $g = 0.2$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)), $\eta = 0.1$ ([학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))
- $\hat m_1 = 0.2$, $\hat v_1 = 0.2^2$ (첫 걸음의 바로잡은 값)

**풀이**
1. [Adam](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#adam-%EC%95%84%EB%8B%B4-%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%80)의 걸음 식을 쓴다: $\eta\, \dfrac{\hat m_1}{\sqrt{\hat v_1} + \epsilon}$
2. 값을 대입한다: $0.1 \times \dfrac{0.2}{\sqrt{0.2^2}}$
3. 제곱과 제곱근을 계산한다: $0.2^2 = 0.04$, $\sqrt{0.04} = 0.2$ ($0.2 \times 0.2 = 0.04$) 이므로 $0.1 \times \dfrac{0.2}{0.2}$
4. 나눈다: $0.2 \div 0.2 = 1$ 이므로 $0.1 \times 1$
5. 곱한다: $0.1 \times 1 = 0.1$
6. SGD의 걸음 식을 쓰고 대입한다: $\eta g = 0.1 \times 0.2$
7. 곱한다: $0.1 \times 0.2 = 0.02$

**답**: [Adam](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#adam-%EC%95%84%EB%8B%B4-%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%80) $0.1$, SGD $0.02$. [Adam](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#adam-%EC%95%84%EB%8B%B4-%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%80)의 첫 걸음은 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 크기와 상관없이 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $0.1$이다.

</details>

### 문제 4

[기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 $g_1 = 1$, $g_2 = 3$으로 이어졌다. 두 번째 걸음($t = 2$)의 1차 모멘트 $m_2$와 바로잡은 $\hat m_2$를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $g_1 = 1$, $g_2 = 3$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))
- $m_0 = 0$, $\beta_1 = 0.9$
- $t = 2$

**풀이**
1. 첫 걸음의 식을 쓰고 대입한다: $m_1 = \beta_1 m_0 + (1-\beta_1) g_1 = 0.9 \times 0 + (1 - 0.9) \times 1$
2. 계산한다: $1 - 0.9 = 0.1$, $0.9 \times 0 = 0$, $0.1 \times 1 = 0.1$ 이므로 $m_1 = 0 + 0.1 = 0.1$
3. 둘째 걸음의 식을 쓰고 대입한다: $m_2 = \beta_1 m_1 + (1-\beta_1) g_2 = 0.9 \times 0.1 + (1 - 0.9) \times 3$
4. 곱셈을 한다: $0.9 \times 0.1 = 0.09$, $0.1 \times 3 = 0.3$ 이므로 $m_2 = 0.09 + 0.3$
5. 더한다: $0.09 + 0.3 = 0.39$
6. 보정 식을 쓰고 대입한다: $\hat m_2 = \dfrac{m_2}{1 - \beta_1^2} = \dfrac{0.39}{1 - 0.9^2}$
7. 분모를 계산한다: $0.9^2 = 0.81$ 이므로 $1 - 0.81 = 0.19$
8. 나눈다: $0.39 \div 0.19 = 2.0526$

**답**: $m_2 = 0.39$, $\hat m_2 = 2.0526$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 1과 3의 가중 평균이라 1과 3 사이 값이고, 최근 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 3 쪽에 더 가깝다)

</details>

## 응용

### 문제 5

[기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 두 걸음 모두 $g = 4$로 같다. [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $\eta = 0.01$일 때 두 번째 걸음($t = 2$)의 걸음 크기를 구하라. 보정이 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 되돌려 주는지도 확인하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $g_1 = g_2 = 4$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)), $\eta = 0.01$ ([학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))
- $m_0 = v_0 = 0$, $\beta_1 = 0.9$, $\beta_2 = 0.999$

**풀이**
1. 식을 그대로 쓴다: $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
2. 첫 걸음을 계산한다: $m_1 = 0.1 \times 4 = 0.4$, $v_1 = 0.001 \times 4^2 = 0.001 \times 16 = 0.016$
3. 둘째 걸음에 대입한다: $m_2 = 0.9 \times 0.4 + 0.1 \times 4$, $v_2 = 0.999 \times 0.016 + 0.001 \times 16$
4. 곱셈을 한다: $0.9 \times 0.4 = 0.36$, $0.1 \times 4 = 0.4$, $0.999 \times 0.016 = 0.015984$, $0.001 \times 16 = 0.016$
5. 더한다: $m_2 = 0.36 + 0.4 = 0.76$, $v_2 = 0.015984 + 0.016 = 0.031984$
6. 보정 분모를 계산한다: $1 - 0.9^2 = 1 - 0.81 = 0.19$, $1 - 0.999^2 = 1 - 0.998001 = 0.001999$
7. 나눈다: $\hat m_2 = 0.76 \div 0.19 = 4$, $\hat v_2 = 0.031984 \div 0.001999 = 16$
8. 걸음 식에 대입한다: $\eta\, \dfrac{\hat m_2}{\sqrt{\hat v_2}} = 0.01 \times \dfrac{4}{\sqrt{16}}$
9. 제곱근을 구하고 나눈다: $\sqrt{16} = 4$ ($4 \times 4 = 16$) 이므로 $0.01 \times \dfrac{4}{4} = 0.01 \times 1$
10. 곱한다: $0.01 \times 1 = 0.01$

**답**: 걸음 크기 $0.01$. [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 같으면 보정 뒤 $\hat m = g = 4$, $\hat v = g^2 = 16$이 정확히 돌아온다.

</details>

### 문제 6

[파라미터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0)가 두 개이고 $\theta_0 = (1, 1)$, 첫 걸음의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 $g = (0.5, -8)$, [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $\eta = 0.1$이다. [Adam](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#adam-%EC%95%84%EB%8B%B4-%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%80)의 첫 걸음 뒤 $\theta_1$은? (원소마다 따로 계산한다.)

<details><summary>풀이와 답</summary>

**주어진 값**
- $\theta_0 = (1, 1)$ ([파라미터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) 두 칸)
- $g = (0.5, -8)$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 두 칸)
- $\eta = 0.1$, $\beta_1 = 0.9$, $\beta_2 = 0.999$

**풀이**
1. 첫 걸음의 보정 값을 구한다: $m_1 = (1-\beta_1) g = 0.1\,g$ 이고 $\hat m_1 = 0.1\,g / 0.1 = g$. $v_1 = 0.001\,g^2$ 이고 $\hat v_1 = 0.001\,g^2 / 0.001 = g^2$
2. 식을 쓴다: $\theta_1 = \theta_0 - \eta\, \dfrac{g}{\sqrt{g^2}}$ ($\sqrt{g^2} = |g|$, 부호를 뗀 크기)
3. 첫째 칸에 대입한다: $\theta_1 = 1 - 0.1 \times \dfrac{0.5}{0.5}$
4. 나누고 곱하고 뺀다: $0.5 \div 0.5 = 1$, $0.1 \times 1 = 0.1$, $1 - 0.1 = 0.9$
5. 둘째 칸에 대입한다: $\theta_1 = 1 - 0.1 \times \dfrac{-8}{8}$
6. 나눈다: $-8 \div 8 = -1$ 이므로 $\theta_1 = 1 - 0.1 \times (-1)$
7. 곱하고 뺀다: $0.1 \times (-1) = -0.1$, $1 - (-0.1) = 1.1$

**답**: $\theta_1 = (0.9,\ 1.1)$. [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 크기가 16배 차이 나도($0.5$ vs $8$) 두 칸 모두 $0.1$만큼 움직였고, 방향은 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)의 반대다.

</details>

### 문제 7

[기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 $g_1 = 2$, 다음에 $g_2 = -2$로 부호가 바뀌었다. [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) $\eta = 0.1$일 때 두 번째 걸음($t = 2$)이 $\theta$를 얼마나, 어느 쪽으로 움직이는지 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $g_1 = 2$, $g_2 = -2$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))
- $m_0 = v_0 = 0$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 0.1$

**풀이**
1. 첫 걸음을 계산한다: $m_1 = 0.1 \times 2 = 0.2$, $v_1 = 0.001 \times 2^2 = 0.001 \times 4 = 0.004$
2. 둘째 걸음의 식에 대입한다: $m_2 = 0.9 \times 0.2 + 0.1 \times (-2)$, $v_2 = 0.999 \times 0.004 + 0.001 \times (-2)^2$
3. 곱셈을 한다: $0.9 \times 0.2 = 0.18$, $0.1 \times (-2) = -0.2$, $0.999 \times 0.004 = 0.003996$, $0.001 \times 4 = 0.004$
4. 더한다: $m_2 = 0.18 + (-0.2) = -0.02$, $v_2 = 0.003996 + 0.004 = 0.007996$
5. 보정 분모를 계산한다: $1 - 0.9^2 = 0.19$, $1 - 0.999^2 = 0.001999$
6. 나눈다: $\hat m_2 = -0.02 \div 0.19 = -0.1053$, $\hat v_2 = 0.007996 \div 0.001999 = 4$
7. 걸음 식을 쓰고 대입한다: $-\eta\, \dfrac{\hat m_2}{\sqrt{\hat v_2}} = -0.1 \times \dfrac{-0.1053}{\sqrt{4}}$
8. 제곱근을 구하고 나눈다: $\sqrt{4} = 2$ 이므로 $-0.1053 \div 2 = -0.0526$ 이고 $-0.1 \times (-0.0526)$
9. 곱한다: 음수 × 음수는 양수이므로 $0.1 \times 0.0526 = 0.0053$

**답**: $\theta$가 $+0.0053$ 움직인다. [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $-2$는 "$\theta$를 키워라"는 뜻이라 방향은 맞지만, $m$에 남은 지난 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)(+2)의 기억이 상쇄해서 첫 걸음(0.1)의 약 1/19로 작아진다.

</details>

### 문제 8

[기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 걸음마다 똑같이 $g = 0.5$이고 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)이 $\eta = 0.002$이다. [파라미터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0)가 $0.01$만큼 움직이려면 [Adam](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#adam-%EC%95%84%EB%8B%B4-%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%80)은 몇 걸음, 같은 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0)의 SGD(걸음 $= \eta g$)는 몇 걸음이 필요한가?

<details><summary>풀이와 답</summary>

**주어진 값**
- $g = 0.5$ (매번 같은 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0))
- $\eta = 0.002$ ([학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0))
- 목표 이동 거리 $0.01$

**풀이**
1. [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 같으면 보정 뒤 $\hat m = g = 0.5$, $\hat v = g^2 = 0.5^2$ (문제 5와 같은 이유)
2. 제곱과 제곱근을 계산한다: $0.5^2 = 0.25$, $\sqrt{0.25} = 0.5$ ($0.5 \times 0.5 = 0.25$)
3. [Adam](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#adam-%EC%95%84%EB%8B%B4-%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%80)의 한 걸음 식을 쓰고 대입한다: $\eta\, \dfrac{\hat m}{\sqrt{\hat v}} = 0.002 \times \dfrac{0.5}{0.5}$
4. 나누고 곱한다: $0.5 \div 0.5 = 1$ 이므로 $0.002 \times 1 = 0.002$
5. [Adam](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#adam-%EC%95%84%EB%8B%B4-%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%80)의 걸음 수를 구한다: $0.01 \div 0.002 = 5$
6. SGD의 한 걸음을 구한다: $\eta g = 0.002 \times 0.5 = 0.001$
7. SGD의 걸음 수를 구한다: $0.01 \div 0.001 = 10$

**답**: [Adam](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#adam-%EC%95%84%EB%8B%B4-%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%80) $5$걸음, SGD $10$걸음. [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 $0.5$로 작으면 SGD는 걸음이 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)에 비례해 작아지지만 [Adam](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#adam-%EC%95%84%EB%8B%B4-%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%80)은 [학습률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#learning-rate-%ED%95%99%EC%8A%B5%EB%A5%A0) 그대로 걷는다. ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 5이면 SGD는 $1$걸음 만에 간다. [Adam](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#adam-%EC%95%84%EB%8B%B4-%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%80)은 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)와 상관없이 5걸음이다.)

</details>

## 강의와 연결

### 문제 9

[PyTorch 튜토리얼 8](../../Tut_PyTorch_%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC.md#8-%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%80)의 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98) $\theta^2$ ([기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $2\theta$)에서 $\theta_0 = 1$, $\eta = 0.1$이다. 첫 걸음에서 $g_1 = 2$이므로 $\theta_1 = 0.9$, $m_1 = 0.2$, $v_1 = 0.004$이다. 두 번째 걸음의 $\theta_2$를 구하라. (노트의 표: $0.8004$)

<details><summary>풀이와 답</summary>

**주어진 값**
- $\theta_1 = 0.9$ (첫 걸음 뒤 [파라미터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0))
- $m_1 = 0.2$, $v_1 = 0.004$ (첫 걸음 뒤 모멘트)
- $\eta = 0.1$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $t = 2$

**풀이**
1. 둘째 걸음의 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 구한다: $g_2 = 2\theta_1 = 2 \times 0.9 = 1.8$
2. 모멘트 식에 대입한다: $m_2 = 0.9 \times 0.2 + 0.1 \times 1.8$, $v_2 = 0.999 \times 0.004 + 0.001 \times 1.8^2$
3. 제곱과 곱셈을 한다: $1.8^2 = 3.24$ 이고 $0.9 \times 0.2 = 0.18$, $0.1 \times 1.8 = 0.18$, $0.999 \times 0.004 = 0.003996$, $0.001 \times 3.24 = 0.00324$
4. 더한다: $m_2 = 0.18 + 0.18 = 0.36$, $v_2 = 0.003996 + 0.00324 = 0.007236$
5. 보정 분모를 계산한다: $1 - 0.9^2 = 0.19$, $1 - 0.999^2 = 0.001999$
6. 나눈다: $\hat m_2 = 0.36 \div 0.19 = 1.8947$, $\hat v_2 = 0.007236 \div 0.001999 = 3.6198$
7. 제곱근을 구한다 (계산기): $\sqrt{3.6198} = 1.9026$ ($1.9026 \times 1.9026 = 3.6199$)
8. 걸음 크기를 구한다: $0.1 \times \dfrac{1.8947}{1.9026} = 0.1 \times 0.9959 = 0.0996$
9. [파라미터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0)를 고친다: $\theta_2 = 0.9 - 0.0996 = 0.8004$

**답**: $\theta_2 = 0.8004$. 노트의 표와 같다. 걸음 크기가 $0.0996$으로 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 $2$에서 $1.8$로 줄어도 거의 $0.1$ 그대로다.

</details>

### 문제 10

[Lec07 6](../../Lec07_%EC%B5%9C%EC%A0%81%ED%99%94%EC%9D%98_%EC%8A%A4%EC%BC%80%EC%9D%BC%EB%A7%81_%EA%B7%9C%EC%B9%99.md#6-%EC%B5%9C%EA%B8%89%EA%B0%95%ED%95%98%EB%B2%95)은 "[지수 이동 평균](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#exponential-moving-average-%EC%A7%80%EC%88%98-%EC%9D%B4%EB%8F%99-%ED%8F%89%EA%B7%A0)을 끈 [Adam](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#adam-%EC%95%84%EB%8B%B4-%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%80)은 [부호 경사하강법](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#sign-gradient-descent-%EB%B6%80%ED%98%B8-%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95)"이라고 설명한다. $\beta_1 = \beta_2 = 0$으로 두면 $m_t = g_t$, $v_t = g_t^2$이고 보정 분모 $1 - 0^t = 1$이다. $\theta = 1$, $\eta = 0.1$에서 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 $g = -5$일 때와 $g = 0.04$일 때의 $\theta$ 갱신 결과를 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $\theta = 1$, $\eta = 0.1$
- $g = -5$ 와 $g = 0.04$ (두 경우)
- $\beta_1 = \beta_2 = 0$ 이므로 $\hat m = g$, $\hat v = g^2$

**풀이**
1. 식을 그대로 쓴다: $\theta_{\text{new}} = \theta - \eta\, \dfrac{\hat m}{\sqrt{\hat v} + \epsilon}$
2. $g = -5$를 대입한다: $\theta_{\text{new}} = 1 - 0.1 \times \dfrac{-5}{\sqrt{(-5)^2}}$
3. 제곱과 제곱근을 계산한다: $(-5)^2 = 25$, $\sqrt{25} = 5$ 이므로 $\theta_{\text{new}} = 1 - 0.1 \times \dfrac{-5}{5}$
4. 나눈다: $-5 \div 5 = -1$ 이므로 $\theta_{\text{new}} = 1 - 0.1 \times (-1)$
5. 곱하고 뺀다: $0.1 \times (-1) = -0.1$, $1 - (-0.1) = 1.1$
6. $g = 0.04$를 대입한다: $\theta_{\text{new}} = 1 - 0.1 \times \dfrac{0.04}{\sqrt{0.04^2}}$
7. 제곱과 제곱근을 계산한다: $0.04^2 = 0.0016$, $\sqrt{0.0016} = 0.04$ 이므로 $\theta_{\text{new}} = 1 - 0.1 \times \dfrac{0.04}{0.04}$
8. 나누고 곱하고 뺀다: $0.04 \div 0.04 = 1$, $0.1 \times 1 = 0.1$, $1 - 0.1 = 0.9$

**답**: $g = -5$이면 $\theta = 1.1$, $g = 0.04$이면 $\theta = 0.9$. [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 125배 차이 나도 걸음은 똑같이 $0.1$이고 방향만 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) 부호의 반대다. 그래서 [부호 경사하강법](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#sign-gradient-descent-%EB%B6%80%ED%98%B8-%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95)이다.

</details>
