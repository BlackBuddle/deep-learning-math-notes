---
title: 예제 MM74 n-그램과 퍼플렉서티
type: note
tags: [MIT딥러닝, 예제]
created: 2026-10-15
cssclasses: [dl-note]
---

# 예제: n-그램 언어 모델과 퍼플렉서티

개념 노트: [MM74_n그램과_퍼플렉서티](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/13_%EC%96%B8%EC%96%B4_%EB%AA%A8%EB%8D%B8%EA%B3%BC_%EC%B6%94%EB%A1%A0/MM74_n%EA%B7%B8%EB%9E%A8%EA%B3%BC_%ED%8D%BC%ED%94%8C%EB%A0%89%EC%84%9C%ED%8B%B0.md) · [목차](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md)

풀이는 접혀 있으니 먼저 풀어 보고 펼쳐 확인한다. 문제 번호 순서는 쉬운 것에서 어려운 것, 마지막 둘은 강의 내용과 잇는 문제다. 문제 1~2와 5~8은 다음 장난감 말뭉치 5문장을 쓰는 3-그램 모델이다.

- "i like green tea", "i like green apples", "i like black tea", "i like green tea", "i like green tea"
- 센 횟수: $c(\text{i like}) = 5$, $c(\text{i like green}) = 4$, $c(\text{i like black}) = 1$, $c(\text{like green}) = 4$, $c(\text{like green tea}) = 3$, $c(\text{like green apples}) = 1$, $c(\text{like black}) = 1$, $c(\text{like black tea}) = 1$

## 기본 (직접 대입)

### 문제 1
위 말뭉치에서 3-그램 모델이 "like green" 다음에 "tea"를 낼 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) $p(\text{tea} \mid \text{like green})$은?

<details><summary>풀이와 답</summary>

**주어진 값**
- $c(\text{like green tea}) = 3$ (세 단어 묶음이 나온 횟수)
- $c(\text{like green}) = 4$ (맥락 두 단어가 나온 횟수)

**풀이**
1. 식을 그대로 쓴다: $p(w_t \mid w_{t-2}, w_{t-1}) = \dfrac{c(w_{t-2}\, w_{t-1}\, w_t)}{c(w_{t-2}\, w_{t-1})}$
2. 값을 대입한다: $p(\text{tea} \mid \text{like green}) = \dfrac{3}{4}$
3. 나눈다: $3 / 4 = 0.75$

**답**: **0.75**. 같은 방법으로 "apples"는 $1/4 = 0.25$이고 둘을 더하면 1이 된다.

</details>

### 문제 2
같은 말뭉치에서 "i like" 다음에 "green"이 올 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)과 "black"이 올 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)을 구하고, 둘을 더하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $c(\text{i like green}) = 4$, $c(\text{i like black}) = 1$ (세 단어 묶음의 횟수)
- $c(\text{i like}) = 5$ (맥락의 횟수)

**풀이**
1. 식을 그대로 쓴다: $p(w_t \mid w_{t-2}, w_{t-1}) = \dfrac{c(w_{t-2}\, w_{t-1}\, w_t)}{c(w_{t-2}\, w_{t-1})}$
2. 값을 대입한다: $p(\text{green} \mid \text{i like}) = \dfrac{4}{5}$, $p(\text{black} \mid \text{i like}) = \dfrac{1}{5}$
3. 나눈다: $4 / 5 = 0.8$, $1 / 5 = 0.2$
4. 더한다: $0.8 + 0.2 = 1$

**답**: green **0.8**, black **0.2**, 합은 **1**. 맥락이 같으면 가능한 다음 단어의 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)을 모두 더해 1이 된다.

</details>

### 문제 3
어휘가 2개뿐이고 모델이 시험 문장의 세 단어 모두에 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) 0.5를 주었다. [퍼플렉서티](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#perplexity-%ED%8D%BC%ED%94%8C%EB%A0%89%EC%84%9C%ED%8B%B0)는?

<details><summary>풀이와 답</summary>

**주어진 값**
- $N = 3$ (잰 단어 수)
- $p(w_1) = p(w_2) = p(w_3) = 0.5$ (모델이 실제 단어마다 준 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0))

**풀이**
1. 식을 그대로 쓴다: $\text{PPL} = \exp\Big(-\dfrac{1}{N}\sum_{t=1}^{N} \ln p(w_t \mid w_{<t})\Big)$
2. 값을 대입한다 (Σ를 세 항으로 펼침): $\text{PPL} = \exp\Big(-\dfrac{1}{3}\big(\ln 0.5 + \ln 0.5 + \ln 0.5\big)\Big)$
3. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 계산한다. $\ln a$는 "$e$(약 2.718)를 몇 제곱해야 $a$가 되나"이고 계산기의 ln 키로 구한다: $\ln 0.5 = -0.6931$
4. 세 항을 더한다: $-0.6931 + (-0.6931) + (-0.6931) = -2.0794$
5. $-\frac{1}{3}$을 곱한다: $-\frac{1}{3} \times (-2.0794) = 0.6931$
6. $e$를 씌운다. $\exp(a) = e^{a}$이고 계산기의 exp 키로 구한다: $e^{0.6931} = 2$

**답**: **2**. 둘 중 하나를 고르게 찍는 것과 같은 헷갈림이다.

</details>

### 문제 4
시험 문장의 두 단어에 모델이 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) 0.5와 0.125를 주었다. [퍼플렉서티](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#perplexity-%ED%8D%BC%ED%94%8C%EB%A0%89%EC%84%9C%ED%8B%B0)는?

<details><summary>풀이와 답</summary>

**주어진 값**
- $N = 2$ (잰 단어 수)
- $p(w_1) = 0.5$, $p(w_2) = 0.125$ (모델이 준 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0), $0.125 = 1/8$)

**풀이**
1. 식을 그대로 쓴다: $\text{PPL} = \exp\Big(-\dfrac{1}{N}\sum_{t=1}^{N} \ln p(w_t \mid w_{<t})\Big)$
2. 값을 대입한다 (Σ를 두 항으로 펼침): $\text{PPL} = \exp\Big(-\dfrac{1}{2}\big(\ln 0.5 + \ln 0.125\big)\Big)$
3. [로그](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#log-ln-%EB%A1%9C%EA%B7%B8)를 계산한다: $\ln 0.5 = -0.6931$, $\ln 0.125 = -2.0794$
4. 괄호 안을 더한다: $-0.6931 + (-2.0794) = -2.7726$
5. $-\frac{1}{2}$을 곱한다: $-\frac{1}{2} \times (-2.7726) = 1.3863$
6. $e$를 씌운다: $e^{1.3863} = 4$
7. 곱으로 확인한다: $(0.5 \times 0.125)^{-1/2} = 0.0625^{-1/2} = \sqrt{1 / 0.0625} = \sqrt{16} = 4$

**답**: **4**

</details>

## 응용

### 문제 5
장난감 말뭉치의 3-그램 모델로 시험 문장 "i like green tea"의 뒤 두 단어(green, tea)를 재서 [퍼플렉서티](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#perplexity-%ED%8D%BC%ED%94%8C%EB%A0%89%EC%84%9C%ED%8B%B0)를 구하라. (처음 두 단어는 맥락으로 주어진다.)

<details><summary>풀이와 답</summary>

**주어진 값**
- $N = 2$ (잰 단어 수: green, tea)
- $p(\text{green} \mid \text{i like}) = 4/5 = 0.8$ (문제 2)
- $p(\text{tea} \mid \text{like green}) = 3/4 = 0.75$ (문제 1)

**풀이**
1. 식을 그대로 쓴다: $\text{PPL} = \Big(\prod_{t=1}^{N} p(w_t \mid w_{<t})\Big)^{-1/N}$
2. 값을 대입한다 (Π를 두 항으로 펼침): $\text{PPL} = (0.8 \times 0.75)^{-1/2}$
3. 곱한다: $0.8 \times 0.75 = 0.6$ 이므로 $\text{PPL} = 0.6^{-1/2}$
4. 거듭제곱을 바꾼다 ($a^{-1/2} = \sqrt{1/a}$): $\text{PPL} = \sqrt{1 / 0.6}$
5. 뒤집는다: $1 / 0.6 = 1.6667$
6. 제곱근을 구한다 ($\sqrt{a}$는 제곱해서 $a$가 되는 수): $\sqrt{1.6667} = 1.2910$

**답**: **1.2910**. 매 단어 약 1.29지 선다를 푸는 만큼만 헷갈린다.

</details>

### 문제 6
같은 모델로 시험 문장 "i like black tea"의 뒤 두 단어(black, tea)를 재서 [퍼플렉서티](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#perplexity-%ED%8D%BC%ED%94%8C%EB%A0%89%EC%84%9C%ED%8B%B0)를 구하고, 문제 5와 비교하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $N = 2$ (잰 단어 수: black, tea)
- $p(\text{black} \mid \text{i like}) = 1/5 = 0.2$ (문제 2)
- $p(\text{tea} \mid \text{like black}) = c(\text{like black tea}) / c(\text{like black}) = 1/1 = 1$

**풀이**
1. 식을 그대로 쓴다: $\text{PPL} = \Big(\prod_{t=1}^{N} p(w_t \mid w_{<t})\Big)^{-1/N}$
2. 값을 대입한다: $\text{PPL} = (0.2 \times 1)^{-1/2}$
3. 곱한다: $0.2 \times 1 = 0.2$ 이므로 $\text{PPL} = 0.2^{-1/2}$
4. 거듭제곱을 바꾼다: $\text{PPL} = \sqrt{1 / 0.2}$
5. 뒤집는다: $1 / 0.2 = 5$
6. 제곱근을 구한다: $\sqrt{5} = 2.2361$
7. 비교한다: $2.2361 > 1.2910$

**답**: **2.2361**. 말뭉치에서 드물게 나온 "black"이 들어 있어 문제 5(1.2910)보다 더 헷갈린다. [퍼플렉서티](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#perplexity-%ED%8D%BC%ED%94%8C%EB%A0%89%EC%84%9C%ED%8B%B0)가 높을수록 모델이 그 문장을 덜 예상했다는 뜻이다.

</details>

### 문제 7
시험 문장 "i like black apples"는 말뭉치에 없다. 평활 없이는 $p(\text{apples} \mid \text{like black}) = 0$이라 [퍼플렉서티](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#perplexity-%ED%8D%BC%ED%94%8C%EB%A0%89%EC%84%9C%ED%8B%B0)가 무한대다. 더하기 1 평활($V = 6$: i, like, green, black, tea, apples)로 [퍼플렉서티](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#perplexity-%ED%8D%BC%ED%94%8C%EB%A0%89%EC%84%9C%ED%8B%B0)를 구하라. 평활 식은 $p_{+1} = \dfrac{c(\text{맥락}\ w) + 1}{c(\text{맥락}) + V}$이다.

<details><summary>풀이와 답</summary>

**주어진 값**
- $c(\text{i like black}) = 1$, $c(\text{i like}) = 5$ (black의 횟수)
- $c(\text{like black apples}) = 0$, $c(\text{like black}) = 1$ (apples의 횟수)
- $V = 6$ (어휘 크기), $N = 2$ (잰 단어 수)

**풀이**
1. 식을 그대로 쓴다: $p_{+1} = \dfrac{c(\text{맥락}\ w) + 1}{c(\text{맥락}) + V}$, $\text{PPL} = \Big(\prod_{t=1}^{N} p(w_t \mid w_{<t})\Big)^{-1/N}$
2. 값을 대입한다: $p_{+1}(\text{black}) = \dfrac{1 + 1}{5 + 6}$, $p_{+1}(\text{apples}) = \dfrac{0 + 1}{1 + 6}$
3. 분자와 분모를 더한다: $1 + 1 = 2$, $5 + 6 = 11$ 이므로 $p_{+1}(\text{black}) = 2/11 = 0.1818$
4. 같은 방법: $0 + 1 = 1$, $1 + 6 = 7$ 이므로 $p_{+1}(\text{apples}) = 1/7 = 0.1429$
5. 곱한다 (반올림 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)를 줄이려고 분수로): $\dfrac{2}{11} \times \dfrac{1}{7} = \dfrac{2}{77} = 0.02597$
6. [퍼플렉서티](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#perplexity-%ED%8D%BC%ED%94%8C%EB%A0%89%EC%84%9C%ED%8B%B0) 식에 넣는다: $\text{PPL} = \left(\dfrac{2}{77}\right)^{-1/2} = \sqrt{77 / 2}$
7. 뒤집는다: $77 / 2 = 38.5$
8. 제곱근을 구한다: $\sqrt{38.5} = 6.2048$

**답**: **6.2048**. 평활이 없으면 무한대였지만, 대신 본 적 있는 단어의 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)이 깎인다.

</details>

### 문제 8
더하기 1 평활($V = 6$)을 하면 "i like" 다음에 올 수 있는 여섯 단어의 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)이 어떻게 되는가? 합이 1인지 확인하라. ($c(\text{i like green}) = 4$, $c(\text{i like black}) = 1$, 나머지 단어 i, like, tea, apples는 0번)

<details><summary>풀이와 답</summary>

**주어진 값**
- $c(\text{i like}) = 5$ (맥락의 횟수), $V = 6$ (어휘 크기)
- 다음 단어별 횟수: green 4, black 1, i 0, like 0, tea 0, apples 0

**풀이**
1. 식을 그대로 쓴다: $p_{+1} = \dfrac{c(\text{맥락}\ w) + 1}{c(\text{맥락}) + V}$
2. 분모를 구한다: $5 + 6 = 11$
3. green의 분자: $4 + 1 = 5$ 이므로 $p_{+1}(\text{green}) = 5/11 = 0.4545$
4. black의 분자: $1 + 1 = 2$ 이므로 $p_{+1}(\text{black}) = 2/11 = 0.1818$
5. 나머지 네 단어의 분자: $0 + 1 = 1$ 이므로 각각 $1/11 = 0.0909$
6. 다 더한다: $\dfrac{5 + 2 + 1 + 1 + 1 + 1}{11} = \dfrac{11}{11} = 1$

**답**: green **0.4545**, black **0.1818**, 나머지 네 단어는 각각 **0.0909**이고 합은 **1**이다. 평활 전 green은 0.8이었다. 본 적 없는 단어에 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)을 나눠 주는 대신 본 단어의 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0)이 줄어든다.

</details>

## 강의와 연결

### 문제 9
[Lec21 2.2](../../Lec21_%EC%96%B8%EC%96%B4_%EB%AA%A8%EB%8D%B8.md#22-%ED%8D%BC%ED%94%8C%EB%A0%89%EC%84%9C%ED%8B%B0%EB%A1%9C-%EC%9E%B0-%EB%B0%9C%EC%A0%84)의 표처럼 모델이 좋아지면 [퍼플렉서티](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#perplexity-%ED%8D%BC%ED%94%8C%EB%A0%89%EC%84%9C%ED%8B%B0)가 내려간다. [퍼플렉서티](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#perplexity-%ED%8D%BC%ED%94%8C%EB%A0%89%EC%84%9C%ED%8B%B0)는 토큰당 [교차 엔트로피](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cross-entropy-%EA%B5%90%EC%B0%A8-%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC) $H$에 $e$를 씌운 값 $\text{PPL} = e^{H}$이다([Lec10 8.1](../../Lec10_%EA%B8%B0%EC%96%B5%EA%B3%BC_%EC%8B%9C%ED%80%80%EC%8A%A4.md#81-%EC%B5%9C%EB%8C%80-%EC%9A%B0%EB%8F%84-%ED%95%99%EC%8A%B5%EA%B3%BC-%EA%B5%90%EC%82%AC-%EA%B0%95%EC%9A%94)의 [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)). [교차 엔트로피](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cross-entropy-%EA%B5%90%EC%B0%A8-%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC) [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 3.0에서 2.0으로 내려가면 [퍼플렉서티](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#perplexity-%ED%8D%BC%ED%94%8C%EB%A0%89%EC%84%9C%ED%8B%B0)는 얼마에서 얼마가 되는가?

<details><summary>풀이와 답</summary>

**주어진 값**
- $H = 3.0$ 과 $H = 2.0$ (토큰당 [교차 엔트로피](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#cross-entropy-%EA%B5%90%EC%B0%A8-%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC) [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98))
- $e \approx 2.718$ (자연로그의 밑)

**풀이**
1. 식을 그대로 쓴다: $\text{PPL} = e^{H}$
2. $H = 3.0$을 대입한다: $\text{PPL} = e^{3.0}$
3. 계산기의 exp로 구한다: $e^{3.0} = 20.09$
4. $H = 2.0$을 대입한다: $\text{PPL} = e^{2.0}$
5. 계산기의 exp로 구한다: $e^{2.0} = 7.389$
6. 비를 구한다 (나눈다): $20.09 / 7.389 = 2.72$

**답**: [퍼플렉서티](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#perplexity-%ED%8D%BC%ED%94%8C%EB%A0%89%EC%84%9C%ED%8B%B0)는 **20.09에서 7.39**로 내려간다. [손실](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#loss-function-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98)이 1 줄 때마다 [퍼플렉서티](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#perplexity-%ED%8D%BC%ED%94%8C%EB%A0%89%EC%84%9C%ED%8B%B0)는 $e$배(약 2.72배) 줄어든다.

</details>

### 문제 10
[Lec21 2.3](../../Lec21_%EC%96%B8%EC%96%B4_%EB%AA%A8%EB%8D%B8.md#23-%ED%81%AC%EA%B8%B0%EB%A7%8C%EC%9C%BC%EB%A1%9C%EB%8A%94-%EB%B6%80%EC%A1%B1%ED%95%98%EB%8B%A4)에서 3,000억 개의 n-그램 횟수로 만든 대형 [언어 모델](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#language-model-%EC%96%B8%EC%96%B4-%EB%AA%A8%EB%8D%B8)이 신경망만 못했다. 이유 하나를 숫자로 보자. 어휘가 $V = 50{,}000$개일 때 가능한 3-그램의 수 $V^3$을 구하고, 말뭉치의 3-그램 자리가 $3 \times 10^{11}$개(3,000억)라면 모든 자리가 서로 다른 조합이라 해도 가능한 조합의 몇 분의 몇만 채울 수 있는지 구하라.

<details><summary>풀이와 답</summary>

**주어진 값**
- $V = 50{,}000 = 5 \times 10^4$ (어휘 크기)
- 말뭉치의 3-그램 자리 수 $= 3 \times 10^{11}$

**풀이**
1. 식을 그대로 쓴다: 가능한 3-그램 수 $= V^3$ (세 칸 모두 $V$가지)
2. 값을 대입한다: $(5 \times 10^4)^3$
3. 앞의 수를 세제곱한다: $5^3 = 125$
4. 10의 거듭제곱을 세제곱한다 ([지수](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#e-exponent-e%EC%99%80-%EC%A7%80%EC%88%98)를 곱한다): $(10^4)^3 = 10^{12}$ 이므로 $V^3 = 125 \times 10^{12}$
5. 자릿수를 맞춘다: $125 \times 10^{12} = 1.25 \times 10^{14}$
6. 비를 구한다 (나눈다): $\dfrac{3 \times 10^{11}}{1.25 \times 10^{14}}$
7. 앞의 수와 [지수](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#e-exponent-e%EC%99%80-%EC%A7%80%EC%88%98)를 따로 나눈다: $3 / 1.25 = 2.4$, $10^{11} / 10^{14} = 10^{-3}$ 이므로 $2.4 \times 10^{-3}$

**답**: 가능한 3-그램은 **$1.25 \times 10^{14}$개**이고 채울 수 있는 비율은 많아야 **0.0024**(0.24%)다. 나머지 조합은 말뭉치에 한 번도 안 나오므로 [확률](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#probability-%ED%99%95%EB%A5%A0) 0 문제가 생긴다. 신경망은 비슷한 맥락끼리 일반화해서 이 문제를 풀어 낸다.

</details>
