---
title: 조건부 GAN 목적 함수
type: note
tags: [MIT딥러닝, math]
created: 2026-10-15
cssclasses: [dl-note]
---

# 조건부 GAN 목적 함수 (Conditional GAN objective, L1 loss)

[목차로 돌아가기](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md) · 먼저 볼 것: [M07_로그](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/01_%EC%88%98%EC%99%80_%ED%95%A8%EC%88%98/M07_%EB%A1%9C%EA%B7%B8.md), [M10_엔트로피](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/04_%ED%99%95%EB%A5%A0%EA%B3%BC_%EC%A0%95%EB%B3%B4/M10_%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC.md), [MM22_기댓값과_일반화_간극](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/04_%ED%99%95%EB%A5%A0%EA%B3%BC_%EC%A0%95%EB%B3%B4/MM22_%EA%B8%B0%EB%8C%93%EA%B0%92%EA%B3%BC_%EC%9D%BC%EB%B0%98%ED%99%94_%EA%B0%84%EA%B7%B9.md), [MM58_조건부_확률과_독립](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/04_%ED%99%95%EB%A5%A0%EA%B3%BC_%EC%A0%95%EB%B3%B4/MM58_%EC%A1%B0%EA%B1%B4%EB%B6%80_%ED%99%95%EB%A5%A0%EA%B3%BC_%EB%8F%85%EB%A6%BD.md) · 쓰는 곳: [Lec16 4.1](../../Lec16_%EC%A1%B0%EA%B1%B4%EB%B6%80_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8.md#41-%EC%83%9D%EC%84%B1%EC%9E%90%EC%99%80-%ED%8C%90%EB%B3%84%EC%9E%90%EC%9D%98-%EA%B2%8C%EC%9E%84), [Lec16 4.2](../../Lec16_%EC%A1%B0%EA%B1%B4%EB%B6%80_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8.md#42-%ED%8C%90%EB%B3%84%EC%9E%90%EA%B0%80-%EC%9E%85%EB%A0%A5%EA%B3%BC-%EC%B6%9C%EB%A0%A5%EC%9D%84-%ED%95%A8%EA%BB%98-%EB%B3%B8%EB%8B%A4), [Lec16 4.3](../../Lec16_%EC%A1%B0%EA%B1%B4%EB%B6%80_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8.md#43-l1-%EC%86%90%EC%8B%A4-%EB%8D%94%ED%95%98%EA%B8%B0), [Lec16 4.4](../../Lec16_%EC%A1%B0%EA%B1%B4%EB%B6%80_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8.md#44-%ED%8C%A8%EC%B9%98-%ED%8C%90%EB%B3%84%EC%9E%90)

📚 학습 순서 84/99 (11. 생성 모델): ← 이전: [MM57_가우스_KL과_재매개변수화](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/11_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8/MM57_%EA%B0%80%EC%9A%B0%EC%8A%A4_KL%EA%B3%BC_%EC%9E%AC%EB%A7%A4%EA%B0%9C%EB%B3%80%EC%88%98%ED%99%94.md) · 다음: [MM60_순환_일관성_손실](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/11_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8/MM60_%EC%88%9C%ED%99%98_%EC%9D%BC%EA%B4%80%EC%84%B1_%EC%86%90%EC%8B%A4.md) →

> [!NOTE]
> **3줄 요약**
> - 강의의 판별자 $d$는 "이 쌍이 **가짜(합성)**일 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)"을 낸다. 판별자는 $\log d(\mathbf{x}, g(\mathbf{x})) + \log(1 - d(\mathbf{x}, \mathbf{y}))$를 **키우고**, 생성자는 같은 값을 **줄인다**. 가짜에 0.8, 진짜에 0.3을 주면 값은 $-0.2231 + (-0.3567) = -0.5798$이다.
> - 균형에서는 판별자가 구별을 못 해 모든 쌍에 0.5를 준다. 그때 값은 $2\ln 0.5 = -1.3863$. 최적 판별자는 $d^* = p_g / (p_g + p_{\text{data}})$이고, $p_{\text{data}} = 0.3$, $p_g = 0.1$이면 0.25다.
> - [조건부 GAN](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#conditional-gan-%EC%A1%B0%EA%B1%B4%EB%B6%80-gan)은 판별자가 입력 $\mathbf{x}$도 함께 본다. 여기에 L1 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) $\lambda\,\mathbb{E}\|\mathbf{y} - g(\mathbf{x})\|_1$를 더하면 [학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5)이 안정된다. 픽셀 차이 (0.1, 0, 0.3)이면 L1 0.1333, $\lambda = 100$이면 13.3333.

## 1. 개념 설명

비유: 위조지폐범(생성자 $g$)과 감별사(판별자 $d$)의 대결이다. 감별사는 진짜와 가짜를 가려 점수를 따고, 위조범은 감별사를 속여 점수를 깎는다. [조건부 GAN](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#conditional-gan-%EC%A1%B0%EA%B1%B4%EB%B6%80-gan)에서는 감별사에게 **주문서(입력 $\mathbf{x}$)**도 보여 준다. 그래야 "진짜 같은 지폐지만 주문한 금액이 아니다"도 잡아낸다.

- **판별자 $d_\phi(\mathbf{x}, \mathbf{y})$**: 입력과 출력의 쌍을 받아 0~1 사이 값을 낸다. 강의 슬라이드는 이 값을 "가짜일 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)"로 둔다(반대로 "진짜일 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)"로 쓰는 책도 많다. 식의 모양만 뒤집힌다).
- **판별자의 점수**: [이진 분류](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#binary-classification-%EC%9D%B4%EC%A7%84-%EB%B6%84%EB%A5%98)의 [교차 엔트로피](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cross-entropy-%EA%B5%90%EC%B0%A8-%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC)에 마이너스를 붙인 것이다. 가짜 쌍 $(\mathbf{x}, g(\mathbf{x}))$에는 $\log d$를, 진짜 쌍 $(\mathbf{x}, \mathbf{y})$에는 $\log(1 - d)$를 받는다. 둘 다 0 이하이고, 완벽히 맞히면 0이다.
- **생성자의 목표**: 같은 점수를 **줄이는 것**이다. 생성자가 바꿀 수 있는 것은 가짜 쌍의 항 $\log d(\mathbf{x}, g(\mathbf{x}))$뿐이다. 판별자가 가짜를 "가짜"라고 확신하지 못하게(값을 낮추게) 만든다.
- **min-max**: $\arg\min_\theta \max_\phi$는 "판별자가 최선을 다할 때, 그 최선의 판별자에게 가장 덜 들키는 생성자"를 찾는다는 뜻이다. 아무 판별자가 아니라 **가장 좋은 판별자**를 속여야 한다.
- **L1 항**: 출력과 [정답](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5)의 픽셀별 절댓값 차이의 평균이다. 혼자 쓰면 흐릿한 평균 그림이 되지만([L2 회귀의 평균 예측](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/04_%ED%99%95%EB%A5%A0%EA%B3%BC_%EC%A0%95%EB%B3%B4/MM35_L2_%ED%9A%8C%EA%B7%80%EC%99%80_%EB%B2%94%EC%A3%BC_%EB%B6%84%ED%8F%AC.md)과 같은 이유), [GAN](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gan-generative-adversarial-network-%EC%83%9D%EC%84%B1%EC%A0%81-%EC%A0%81%EB%8C%80-%EC%8B%A0%EA%B2%BD%EB%A7%9D) 항과 함께 쓰면 큰 틀을 잡아 주어 [학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5)이 안정된다.

## 2. 필요한 이유

[Lec16 4.1](../../Lec16_%EC%A1%B0%EA%B1%B4%EB%B6%80_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8.md#41-%EC%83%9D%EC%84%B1%EC%9E%90%EC%99%80-%ED%8C%90%EB%B3%84%EC%9E%90%EC%9D%98-%EA%B2%8C%EC%9E%84)은 생성자 입장에서 판별자가 "[학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5)되는 [손실 함수](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)"라고 설명한다. [Lec16 4.2](../../Lec16_%EC%A1%B0%EA%B1%B4%EB%B6%80_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8.md#42-%ED%8C%90%EB%B3%84%EC%9E%90%EA%B0%80-%EC%9E%85%EB%A0%A5%EA%B3%BC-%EC%B6%9C%EB%A0%A5%EC%9D%84-%ED%95%A8%EA%BB%98-%EB%B3%B8%EB%8B%A4)는 판별자가 출력만 보면 입력과 맞지 않는 진짜 사진도 통과시킨다는 문제를 짚고, [Lec16 4.3](../../Lec16_%EC%A1%B0%EA%B1%B4%EB%B6%80_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8.md#43-l1-%EC%86%90%EC%8B%A4-%EB%8D%94%ED%95%98%EA%B8%B0)은 L1 항을 더한다. 식의 각 항이 어떤 숫자가 되는지 보면 "누가 무엇을 올리고 내리는지"가 분명해진다.

## 3. 숫자로 보기

**판별자의 점수** (한 쌍씩, 한 줄에 한 동작)

| 상황 | 가짜 쌍에 준 $d$ | 진짜 쌍에 준 $d$ | $\log d(\text{가짜})$ | $\log(1 - d(\text{진짜}))$ | 합 |
|---|---|---|---|---|---|
| 판별자가 잘 가림 | 0.8 | 0.3 | -0.2231 | -0.3567 | -0.5798 |
| 균형 (구별 못 함) | 0.5 | 0.5 | -0.6931 | -0.6931 | -1.3863 |

- 판별자는 합을 0에 가깝게(크게) 만들려 하고, 생성자는 첫 칸을 작게 만들려 한다.
- 생성자가 좋아져 가짜에 대한 $d$가 0.8 → 0.5 → 0.2로 내려가면 $\log d$는 -0.2231 → -0.6931 → -1.6094로 작아진다.

$$
\mathcal{L}_{\text{cGAN}}(\theta, \phi) = \mathbb{E}_{\mathbf{x}, \mathbf{y}}\big[\log d_\phi(\mathbf{x}, g_\theta(\mathbf{x})) + \log(1 - d_\phi(\mathbf{x}, \mathbf{y}))\big], \qquad g_\theta^* = \arg\min_\theta \max_\phi \mathcal{L}_{\text{cGAN}}(\theta, \phi)
$$

<details><summary>기호 풀이</summary>

| 기호 | 읽는 법 | 뜻 | 예제 값 |
| --- | --- | --- | --- |
| $\mathbf{x}$ | 엑스 | 입력 (예: 건물 도면) | |
| $\mathbf{y}$ | 와이 | 입력에 짝지어진 진짜 출력 (예: 건물 사진) | |
| $g_\theta$ | 지 세타 | 생성자, [파라미터](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#parameter-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0) $\theta$ | |
| $d_\phi$ | 디 파이 | 판별자, 쌍이 가짜일 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)을 냄 | |
| $d_\phi(\mathbf{x}, g_\theta(\mathbf{x}))$ | | 가짜 쌍에 준 "가짜 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)" | 0.8 |
| $d_\phi(\mathbf{x}, \mathbf{y})$ | | 진짜 쌍에 준 "가짜 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)" | 0.3 |
| $\mathbb{E}_{\mathbf{x}, \mathbf{y}}$ | [기댓값](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#expectation-%EA%B8%B0%EB%8C%93%EA%B0%92) | 데이터 쌍에 대한 평균 | |
| $\max_\phi$, $\min_\theta$ | 맥스, 민 | 판별자는 키우고 생성자는 줄인다 | |
| $\mathcal{L}_{\text{cGAN}}$ | 엘 씨갠 | [조건부 GAN](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#conditional-gan-%EC%A1%B0%EA%B1%B4%EB%B6%80-gan)의 값 | -0.5798 |

</details>

<details><summary>예제와 풀이</summary>

**문제**: 판별자가 생성자가 만든 쌍에 "가짜 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)" 0.6을, 진짜 쌍에 0.4를 주었다. 이 한 쌍씩에 대한 $\mathcal{L}_{\text{cGAN}}$ 값은?

**주어진 값**
- $d_\phi(\mathbf{x}, g_\theta(\mathbf{x})) = 0.6$ (가짜 쌍을 가짜라고 볼 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))
- $d_\phi(\mathbf{x}, \mathbf{y}) = 0.4$ (진짜 쌍을 가짜라고 볼 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))

**풀이**
1. 식을 그대로 쓴다: $\mathcal{L}_{\text{cGAN}} = \log d_\phi(\mathbf{x}, g_\theta(\mathbf{x})) + \log(1 - d_\phi(\mathbf{x}, \mathbf{y}))$ (쌍이 하나라 평균 $\mathbb{E}$는 그 값 자체)
2. 값을 대입한다: $\mathcal{L}_{\text{cGAN}} = \log 0.6 + \log(1 - 0.4)$
3. 괄호 안 뺄셈을 한다: $1 - 0.4 = 0.6$ 이므로 $\mathcal{L}_{\text{cGAN}} = \log 0.6 + \log 0.6$
4. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 계산한다. $\ln y$는 약 2.718인 수 $e$를 몇 제곱해야 $y$가 되는지이고, 1보다 작은 수의 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)는 음수다: $\ln 0.6 = -0.5108$
5. 두 항을 더한다: $-0.5108 + (-0.5108) = -1.0217$

**답**: **-1.0217** (반올림 때문에 $-0.5108 \times 2$의 넷째 자리가 1 다르다). 균형값 -1.3863보다 크다. 판별자가 아직 조금은 가려내고 있다는 뜻이다.

</details>

<details><summary>내 이해</summary>



</details>

**L1 항 더하기** (픽셀 3개, $\lambda = 100$)
- 생성 출력 $g(\mathbf{x}) = (0.2, 0.5, 0.9)$, [정답](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5) $\mathbf{y} = (0.3, 0.5, 0.6)$
- 칸별 절댓값 차이: $(0.1, 0, 0.3)$
- 평균: $(0.1 + 0 + 0.3)/3 = 0.1333$
- $\lambda$ 곱: $100 \times 0.1333 = 13.3333$
- 생성자가 줄이는 전체 값(판별자 항이 $\ln 0.8 = -0.2231$일 때): $-0.2231 + 13.3333 = 13.1102$

$$
g_\theta^* = \arg\min_\theta \max_\phi\ \mathcal{L}_{\text{cGAN}}(\theta, \phi) + \lambda\,\mathcal{L}_{L1}(\theta), \qquad \mathcal{L}_{L1}(\theta) = \mathbb{E}_{\mathbf{x}, \mathbf{y}}\,\|\mathbf{y} - g_\theta(\mathbf{x})\|_1
$$

<details><summary>기호 풀이</summary>

| 기호 | 읽는 법 | 뜻 | 예제 값 |
| --- | --- | --- | --- |
| $\mathcal{L}_{L1}$ | 엘 엘원 | 출력과 [정답](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5)의 L1 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) | 0.1333 |
| $\|\cdot\|_1$ | 엘원 [노름](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#norm-%EB%85%B8%EB%A6%84) | 칸별 절댓값의 합 (여기서는 픽셀 수로 나눈 평균) | |
| $\lambda$ | 람다 | 두 항의 무게를 맞추는 수 | 100 |
| $g_\theta^*$ | 지 세타 스타 | 찾고 싶은 최적 생성자 | |

</details>

<details><summary>예제와 풀이</summary>

**문제**: 생성자가 픽셀 3개를 $(0.2, 0.5, 0.9)$로 그렸고 [정답](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5)은 $(0.3, 0.5, 0.6)$이다. 판별자 항이 $\ln 0.8$이고 $\lambda = 100$일 때 생성자가 줄이려는 전체 값은? (L1은 픽셀 평균으로 잰다)

**주어진 값**
- $g_\theta(\mathbf{x}) = (0.2, 0.5, 0.9)$ (생성자가 그린 픽셀 값)
- $\mathbf{y} = (0.3, 0.5, 0.6)$ ([정답](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5) 픽셀 값)
- $\mathcal{L}_{\text{cGAN}} = \ln 0.8$ (현재 판별자 항, 생성자가 바꿀 수 있는 부분)
- $\lambda = 100$ (L1 항의 무게)

**풀이**
1. 식을 그대로 쓴다: $\mathcal{L}_{\text{cGAN}} + \lambda\,\|\mathbf{y} - g_\theta(\mathbf{x})\|_1$
2. 값을 대입한다: $\ln 0.8 + 100 \times \|(0.3, 0.5, 0.6) - (0.2, 0.5, 0.9)\|_1$
3. 칸별로 뺀다: $(0.3 - 0.2,\ 0.5 - 0.5,\ 0.6 - 0.9) = (0.1, 0, -0.3)$
4. 칸별 절댓값을 취한다 (부호를 없앰): $(0.1, 0, 0.3)$
5. 더한다: $0.1 + 0 + 0.3 = 0.4$
6. 픽셀 수로 나눠 평균을 낸다: $0.4 \div 3 = 0.1333$
7. $\lambda$를 곱한다: $100 \times 0.1333 = 13.3333$
8. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 계산한다: $\ln 0.8 = -0.2231$
9. 두 항을 더한다: $-0.2231 + 13.3333 = 13.1102$

**답**: **13.1102**. 이 단계에서는 L1 항이 훨씬 커서 생성자가 먼저 [정답](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#label-%EC%A0%95%EB%8B%B5) 근처로 끌려간다. 실제 pix2pix도 $\lambda = 100$을 썼다.

</details>

<details><summary>내 이해</summary>



</details>

## 4. 코드로 확인

```python
# [함수] import math: log 같은 수학 함수
import math
# [변수] dfake, dreal: 가짜 쌍과 진짜 쌍에 판별자가 준 "가짜 확률" (3절 표 첫 행)
dfake, dreal = 0.8, 0.3
# [변수] V: 판별자의 점수. 가짜에는 log d, 진짜에는 log(1 - d)
V = math.log(dfake) + math.log(1 - dreal)
# [연결] 3절 표 → -0.5798
print(round(V, 4))
# [연결] 균형(d = 0.5 어디서나) → -1.3863
print(round(2 * math.log(0.5), 4))
# [함수] for d in ...: 생성자가 좋아지며 가짜에 대한 d가 내려가는 세 단계
for d in [0.8, 0.5, 0.2]:
    # [쓸모] 생성자가 줄이는 항이 실제로 작아지는지 본다
    print(d, round(math.log(d), 4))
# [변수] gx: 생성 출력, y: 정답 (픽셀 3개)
gx, y = np.array([0.2, 0.5, 0.9]), np.array([0.3, 0.5, 0.6])
# [함수] np.abs(...).mean(): 칸별 절댓값 차이의 평균 = L1
# [변수] l1 = 0.1333
l1 = np.abs(y - gx).mean()
# [연결] 3절 "L1 항 더하기" → 0.1333, 13.3333, 13.1102
print(round(l1, 4), round(100 * l1, 4), round(math.log(0.8) + 100 * l1, 4))
# [변수] pdata, pg: 어떤 쌍이 진짜 데이터에서 나올 확률, 생성자에서 나올 확률
pdata, pg = 0.3, 0.1
# [연결] 최적 판별자 d* = pg / (pg + pdata) → 0.25 (개념 재확인 2)
print(pg / (pg + pdata))
```

**실행 결과**:

```text
-0.5798
-1.3863
0.8 -0.2231
0.5 -0.6931
0.2 -1.6094
0.1333 13.3333 13.1102
0.25
```

> [!WARNING]
> **⚠ 주의**
> - **부호 약속을 확인한다.** 원래 [GAN](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gan-generative-adversarial-network-%EC%83%9D%EC%84%B1%EC%A0%81-%EC%A0%81%EB%8C%80-%EC%8B%A0%EA%B2%BD%EB%A7%9D) 논문(Goodfellow 등 2014)은 $d$를 "진짜일 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)"로 두고 $\log d(\text{진짜}) + \log(1 - d(\text{가짜}))$를 판별자가 키운다. 강의 슬라이드는 $d$를 "가짜일 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)"로 둔다. 둘은 $d \to 1 - d$로 바꾼 같은 게임이다.
> - **[패치 판별자](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#patchgan-%ED%8C%A8%EC%B9%98-%ED%8C%90%EB%B3%84%EC%9E%90)**([Lec16 4.4](../../Lec16_%EC%A1%B0%EA%B1%B4%EB%B6%80_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8.md#44-%ED%8C%A8%EC%B9%98-%ED%8C%90%EB%B3%84%EC%9E%90))는 패치마다 $d$를 내고 그 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 평균 낸다. 패치 4개에 (0.9, 0.2, 0.6, 0.5)를 주면 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)는 (-0.1054, -1.6094, -0.5108, -0.6931), 평균 -0.7297이다. 패치를 1×1 픽셀로 줄이면 픽셀마다 따로 보는 것이 되어 다시 [독립 가정](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/04_%ED%99%95%EB%A5%A0%EA%B3%BC_%EC%A0%95%EB%B3%B4/MM58_%EC%A1%B0%EA%B1%B4%EB%B6%80_%ED%99%95%EB%A5%A0%EA%B3%BC_%EB%8F%85%EB%A6%BD.md)으로 돌아간다.
> - 실제 구현에서 생성자는 $\log d(\text{가짜})$를 줄이는 대신 "가짜가 진짜로 분류될 [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8) [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)"을 키우는 식(non-saturating)을 자주 쓴다. [학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5) 초반 판별자가 압도할 때 [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)가 사라지는 문제를 피하려는 것이다.

## 5. 개념 재확인

<details><summary>1. 생성자가 좋아져서 판별자가 가짜 쌍에 주는 "가짜 확률"이 0.8에서 0.3으로 내려갔다. 진짜 쌍에는 계속 0.3을 준다. 판별자의 점수는 어떻게 바뀌나?</summary>

| 단계 | 하는 일 | 식 | 계산 | 결과 |
|---|---|---|---|---|
| 1 | 전: 가짜 항 | $\ln 0.8$ | | -0.2231 |
| 2 | 전: 진짜 항 | $\ln(1 - 0.3)$ | $\ln 0.7$ | -0.3567 |
| 3 | 전: 합 | | $-0.2231 - 0.3567$ | -0.5798 |
| 4 | 후: 가짜 항 | $\ln 0.3$ | | -1.204 |
| 5 | 후: 합 | | $-1.204 - 0.3567$ | -1.5606 |

**① 쓰는 식**
$$V = \ln d(\text{가짜}) + \ln(1 - d(\text{진짜}))$$
- 위 식 ← ②(전), ③(후)에서 쓴다.
```python
# [함수] import math: 로그 계산
import math                                            # ①
# [연결] 전 -0.5798
print(round(math.log(0.8) + math.log(1 - 0.3), 4))    # ②
# [연결] 후 -1.5606 → 판별자 점수가 내려갔다 = 생성자가 이기는 중
print(round(math.log(0.3) + math.log(1 - 0.3), 4))    # ③
```

**주어진 값**
- $d(\text{가짜}) = 0.8$ (전), $0.3$ (후): 가짜 쌍에 준 가짜 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)
- $d(\text{진짜}) = 0.3$: 진짜 쌍에 준 가짜 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) (그대로)

**풀이**
1. 식을 그대로 쓴다: $V = \ln d(\text{가짜}) + \ln(1 - d(\text{진짜}))$
2. 값을 대입한다 (전): $V = \ln 0.8 + \ln(1 - 0.3)$
3. (②) 괄호 안 뺄셈을 한다: $1 - 0.3 = 0.7$
4. (②) [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 계산한다($\ln y$는 $e$를 몇 제곱해야 $y$가 되는지): $\ln 0.8 = -0.2231$, $\ln 0.7 = -0.3567$
5. (②) 더한다: $-0.2231 + (-0.3567) = -0.5798$
6. 값을 대입한다 (후): $V = \ln 0.3 + \ln 0.7$
7. (③) [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 계산한다: $\ln 0.3 = -1.204$
8. (③) 더한다: $-1.204 + (-0.3567) = -1.5606$

**답**: **-0.5798 → -1.5606**. 판별자의 점수가 내려갔다. 판별자는 다음 단계에서 다시 이 값을 올리려고 배운다. 이 밀고 당기기가 [GAN](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gan-generative-adversarial-network-%EC%83%9D%EC%84%B1%EC%A0%81-%EC%A0%81%EB%8C%80-%EC%8B%A0%EA%B2%BD%EB%A7%9D) [학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5)이다.

</details>

<details><summary>2. 어떤 쌍 $(\mathbf{x}, \mathbf{y})$가 진짜 데이터에서 나올 확률이 $p_{\text{data}} = 0.3$, 생성자에서 나올 확률이 $p_g = 0.1$이다. 최적 판별자 $d^*$는 이 쌍에 "가짜 확률"을 얼마로 주나?</summary>

| 단계 | 하는 일 | 식 | 계산 | 결과 |
|---|---|---|---|---|
| 1 | 분모 | $p_g + p_{\text{data}}$ | $0.1 + 0.3$ | 0.4 |
| 2 | 나누기 | $p_g / (p_g + p_{\text{data}})$ | $0.1 / 0.4$ | 0.25 |

**① 쓰는 식**
$$d^*(\mathbf{x}, \mathbf{y}) = \frac{p_g(\mathbf{x}, \mathbf{y})}{p_g(\mathbf{x}, \mathbf{y}) + p_{\text{data}}(\mathbf{x}, \mathbf{y})}$$
- 위 식 ← ②, ③에서 쓴다. 두 출처 중 "가짜 쪽에서 왔을 [조건부 확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#conditional-probability-%EC%A1%B0%EA%B1%B4%EB%B6%80-%ED%99%95%EB%A5%A0)"이다([베이즈 정리](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/04_%ED%99%95%EB%A5%A0%EA%B3%BC_%EC%A0%95%EB%B3%B4/MM58_%EC%A1%B0%EA%B1%B4%EB%B6%80_%ED%99%95%EB%A5%A0%EA%B3%BC_%EB%8F%85%EB%A6%BD.md)와 같은 모양).
```python
# [변수] pdata, pg: 두 출처에서 이 쌍이 나올 확률
pdata, pg = 0.3, 0.1                     # ①
# [변수] den: 분모 0.4
den = pg + pdata                         # ②
# [연결] 0.25. pg = pdata가 되면 0.5 (균형)
print(round(pg / den, 4))                # ③
```

**주어진 값**
- $p_{\text{data}} = 0.3$ (진짜 데이터에서 이 쌍이 나올 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))
- $p_g = 0.1$ (생성자에서 이 쌍이 나올 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))

**풀이**
1. 식을 그대로 쓴다: $d^* = \dfrac{p_g}{p_g + p_{\text{data}}}$
2. 값을 대입한다: $d^* = \dfrac{0.1}{0.1 + 0.3}$
3. (②) 분모를 더한다: $0.1 + 0.3 = 0.4$
4. (③) 나눗셈을 한다: $0.1 \div 0.4 = 0.25$

**답**: **0.25**. 생성자가 이 쌍을 데이터만큼(0.3) 자주 만들게 되면 $d^* = 0.3/0.6 = 0.5$가 되어 판별자는 더 가려낼 수 없다. 이것이 균형이다.

</details>

## 예제 풀기

[예제 10문제 풀기](../../%EC%98%88%EC%A0%9C/11_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8/%EC%98%88%EC%A0%9C_MM59_%EC%A1%B0%EA%B1%B4%EB%B6%80_GAN_%EB%AA%A9%EC%A0%81_%ED%95%A8%EC%88%98.md)

<details><summary>내 이해</summary>



</details>
