---
title: ReLU로 만드는 직사각형
type: note
tags: [MIT딥러닝, math]
created: 2026-10-15
cssclasses: [dl-note]
---

# ReLU로 만드는 직사각형 (ReLU rectangle, Hyperrectangle)

[목차로 돌아가기](../../00_MIT%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%AA%A9%EC%B0%A8.md) · 먼저 볼 것: [MM07_리만_합과_보편_근사](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/06_%EA%B7%BC%EC%82%AC_%EC%9D%B4%EB%A1%A0/MM07_%EB%A6%AC%EB%A7%8C_%ED%95%A9%EA%B3%BC_%EB%B3%B4%ED%8E%B8_%EA%B7%BC%EC%82%AC.md), [MM12_직사각형_근사와_차원의_저주](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/06_%EA%B7%BC%EC%82%AC_%EC%9D%B4%EB%A1%A0/MM12_%EC%A7%81%EC%82%AC%EA%B0%81%ED%98%95_%EA%B7%BC%EC%82%AC%EC%99%80_%EC%B0%A8%EC%9B%90%EC%9D%98_%EC%A0%80%EC%A3%BC.md) · 쓰는 곳: [Lec03 7.3](../../Lec03_%EA%B7%BC%EC%82%AC_%EC%9D%B4%EB%A1%A0.md#73-3%EB%8B%A8%EA%B3%84-relu%EB%A1%9C-%EC%A7%81%EC%82%AC%EA%B0%81%ED%98%95-%EB%A7%8C%EB%93%A4%EA%B8%B0), [Lec03 7.4](../../Lec03_%EA%B7%BC%EC%82%AC_%EC%9D%B4%EB%A1%A0.md#74-%EC%B4%88%EC%A7%81%EC%82%AC%EA%B0%81%ED%98%95%EA%B3%BC-%EC%A1%B0%EB%A6%BD)

📚 학습 순서 47/99 (6. 근사 이론): ← 이전: [MM12_직사각형_근사와_차원의_저주](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/06_%EA%B7%BC%EC%82%AC_%EC%9D%B4%EB%A1%A0/MM12_%EC%A7%81%EC%82%AC%EA%B0%81%ED%98%95_%EA%B7%BC%EC%82%AC%EC%99%80_%EC%B0%A8%EC%9B%90%EC%9D%98_%EC%A0%80%EC%A3%BC.md) · 다음: [MM14_꺾임_수와_깊이_분리](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/06_%EA%B7%BC%EC%82%AC_%EC%9D%B4%EB%A1%A0/MM14_%EA%BA%BE%EC%9E%84_%EC%88%98%EC%99%80_%EA%B9%8A%EC%9D%B4_%EB%B6%84%EB%A6%AC.md) →

> [!NOTE]
> **3줄 요약**
> - [MM07](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/06_%EA%B7%BC%EC%82%AC_%EC%9D%B4%EB%A1%A0/MM07_%EB%A6%AC%EB%A7%8C_%ED%95%A9%EA%B3%BC_%EB%B3%B4%ED%8E%B8_%EA%B7%BC%EC%82%AC.md)에서는 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 3개로 **삼각 봉우리**를 만들었다. 이번에는 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 4개로 **직사각형**을 만든다. $f_c(x) = \mathrm{ReLU}(cx) - \mathrm{ReLU}(cx - 1) - \mathrm{ReLU}(c(x-1) - 2) + \mathrm{ReLU}(c(x-1) - 3)$은 $c$가 클수록 [0, 1] 구간의 높이 1 직사각형에 가까워진다. 차이의 넓이는 정확히 $3/c$다.
> - $d$차원 상자(**[초직사각형](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#hyperrectangle-%EC%B4%88%EC%A7%81%EC%82%AC%EA%B0%81%ED%98%95)**)는 좌표마다 만든 직사각형을 **더한 뒤 $d - 1$을 빼고 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)**를 씌워 만든다. 모든 좌표가 구간 안일 때만 합이 $d$라서 1이 남고, 하나라도 밖이면 0이 된다.
> - 상자들에 높이 $\alpha_i$를 곱해 더하면(세 번째 [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)) 립시츠 함수 전체를 근사하는 **3층 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 망**이 된다.

## 1. 개념 설명

비유: 문을 여닫는 **스위치 두 쌍**. $x = 0$에서 "켜짐" 스위치(경사로 올라가기)와 "멈춤" 스위치(평평하게)를 누르고, $x = 1$ 근처에서 "끄기"와 "멈춤"을 한 번 더 누른다. 경사를 가파르게($c$를 크게) 할수록 문이 순식간에 열리고 닫혀 직사각형이 된다.

- **[ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 4개의 역할** ($c = 10$일 때):

| 항 | 켜지는 곳 | 하는 일 |
|---|---|---|
| $+\mathrm{ReLU}(cx)$ | $x = 0$ | [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $+c$로 올라가기 시작 |
| $-\mathrm{ReLU}(cx - 1)$ | $x = 1/c = 0.1$ | [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 0으로 → 높이 1에서 평평 |
| $-\mathrm{ReLU}(c(x-1) - 2)$ | $x = 1 + 2/c = 1.2$ | [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0) $-c$로 내려가기 시작 |
| $+\mathrm{ReLU}(c(x-1) - 3)$ | $x = 1 + 3/c = 1.3$ | [기울기](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#gradient-%EA%B8%B0%EC%9A%B8%EA%B8%B0)를 0으로 → 높이 0에서 평평 |

- 행렬로 쓰면 강의 슬라이드의 식과 같다. 첫 [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5)이 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 4개, 출력은 계수 $(+1, -1, -1, +1)$의 [가중합](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weighted-sum-%EA%B0%80%EC%A4%91%ED%95%A9)이다.
$$f_c(x) = \begin{bmatrix} +1 \\ -1 \\ -1 \\ +1 \end{bmatrix}^{\top} \mathrm{ReLU}\begin{bmatrix} cx \\ cx - 1 \\ c(x-1) - 2 \\ c(x-1) - 3 \end{bmatrix}$$

<details><summary>기호 풀이</summary>

| 기호 | 읽는 법 | 뜻 | 예제 값 |
| --- | --- | --- | --- |
| $f_c(x)$ | 에프 씨 엑스 | [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 4개로 만든 1차원 직사각형 | $f_4(1.6) = 0.6$ |
| $c$ | 씨 | 경사의 가파름 | 4 |
| $x$ | 엑스 | 입력 | 1.6 |
| $\mathrm{ReLU}$ | [렐루](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) | 음수는 0, 양수는 그대로 (칸마다 따로) | $\mathrm{ReLU}(-0.6) = 0$ |
| $\begin{bmatrix} +1 \\ -1 \\ -1 \\ +1 \end{bmatrix}^{\top}$ | 계수 벡터의 [전치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#transpose-%EC%A0%84%EC%B9%98) | 네 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 출력에 곱할 둘째 [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) | (+1, −1, −1, +1) |
| $\top$ | 트랜스포즈 | 세로 벡터를 가로로 눕힘 ([내적](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#dot-product-%EB%82%B4%EC%A0%81)) | - |

</details>

<details><summary>예제와 풀이</summary>

**문제**: 경사의 가파름이 $c = 4$인 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 직사각형 $f_c$에 입력 1.6을 넣으면 출력은?

**주어진 값**:
- $c = 4$ (경사의 가파름)
- $x = 1.6$ (입력)

**풀이**
1. 식을 그대로 쓴다: $f_c(x) = \begin{bmatrix} +1 \\ -1 \\ -1 \\ +1 \end{bmatrix}^{\top} \mathrm{ReLU}\begin{bmatrix} cx \\ cx - 1 \\ c(x-1) - 2 \\ c(x-1) - 3 \end{bmatrix}$
2. 값을 대입한다: $f_4(1.6) = \begin{bmatrix} +1 \\ -1 \\ -1 \\ +1 \end{bmatrix}^{\top} \mathrm{ReLU}\begin{bmatrix} 4 \times 1.6 \\ 4 \times 1.6 - 1 \\ 4 \times (1.6 - 1) - 2 \\ 4 \times (1.6 - 1) - 3 \end{bmatrix}$
3. 괄호 안 뺄셈을 한다: $1.6 - 1 = 0.6$
4. 곱셈을 한다: $4 \times 1.6 = 6.4$, $4 \times 0.6 = 2.4$
5. 칸마다 뺄셈을 한다: $6.4$, $6.4 - 1 = 5.4$, $2.4 - 2 = 0.4$, $2.4 - 3 = -0.6$
6. [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)의 뜻: 음수는 0으로 바꾸고, 0 이상은 그대로 둔다: $(6.4,\ 5.4,\ 0.4,\ 0)$
7. [전치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#transpose-%EC%A0%84%EC%B9%98) 곱([내적](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#dot-product-%EB%82%B4%EC%A0%81))을 항별로 펼친다: $(+1) \times 6.4 + (-1) \times 5.4 + (-1) \times 0.4 + (+1) \times 0$
8. 곱셈을 한다: $6.4 - 5.4 - 0.4 + 0$
9. 앞에서부터 뺄셈을 한다: $6.4 - 5.4 = 1$ 이므로 $1 - 0.4 + 0$
10. 뺄셈을 한다: $1 - 0.4 = 0.6$

**답**: $f_4(1.6) = 0.6$ (내려가는 경사 1.5~1.75 위)

</details>

<details><summary>내 이해</summary>



</details>

- **[초직사각형](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#hyperrectangle-%EC%B4%88%EC%A7%81%EC%82%AC%EA%B0%81%ED%98%95)([Hyperrectangle](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#hyperrectangle-%EC%B4%88%EC%A7%81%EC%82%AC%EA%B0%81%ED%98%95))**: 좌표 $x_1, \dots, x_d$ 각각에 $f_c$를 적용해 더하면, 모든 좌표가 [0, 1]이면 합이 $d$, 하나가 밖이면 $d - 1$ 이하. $d - 1$을 빼고 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)를 씌우면 상자 안만 1이 남는다.
$$h_c(x) = \mathrm{ReLU}\Big(\sum_{i=1}^{d} f_c(x_i) - (d - 1)\Big)$$

<details><summary>기호 풀이</summary>

| 기호 | 읽는 법 | 뜻 | 예제 값 |
| --- | --- | --- | --- |
| $h_c(x)$ | 에이치 씨 엑스 | $d$차원 상자([초직사각형](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#hyperrectangle-%EC%B4%88%EC%A7%81%EC%82%AC%EA%B0%81%ED%98%95)) | 0.5 |
| $d$ | 디 | 입력 차원 | 3 |
| $x_i$ | 엑스 아이 | 입력의 $i$번째 좌표 | $x_3 = 1.25$ |
| $f_c(x_i)$ | 에프 씨 엑스 아이 | 좌표 하나에 씌운 직사각형 | $f_{10}(1.25) = 0.5$ |
| $\sum_{i=1}^{d}$ | 시그마 아이는 1부터 디까지 | 좌표 $d$개의 직사각형을 더함 | 2.5 |
| $d - 1$ | 디 빼기 일 | 상자 밖을 잘라내는 문턱 | 2 |
| $\mathrm{ReLU}$ | [렐루](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) | 문턱 아래(음수)를 0으로 자름 | $\mathrm{ReLU}(0.5) = 0.5$ |

</details>

<details><summary>예제와 풀이</summary>

**문제**: 입력이 3개($d = 3$)이고 $c = 10$일 때, 점 (0.5, 0.5, 1.25)에서 상자 함수 $h_c$의 값은?

**주어진 값**:
- $d = 3$ (입력 차원)
- $c = 10$ (경사의 가파름)
- $x = (0.5, 0.5, 1.25)$ (입력 점, 셋째 좌표만 경계 근처)

**풀이**
1. 식을 그대로 쓴다: $h_c(x) = \mathrm{ReLU}\Big(\sum_{i=1}^{d} f_c(x_i) - (d - 1)\Big)$
2. 값을 대입한다: $h_{10}(0.5, 0.5, 1.25) = \mathrm{ReLU}\Big(f_{10}(0.5) + f_{10}(0.5) + f_{10}(1.25) - (3 - 1)\Big)$
3. $f_{10}$ 값을 읽는다(3.1 표): $f_{10}(0.5) = 1$ (평평한 구간), $f_{10}(1.25) = 0.5$ (내려가는 경사의 가운데)
4. 괄호 뺄셈을 한다: $3 - 1 = 2$ 이므로 $\mathrm{ReLU}(1 + 1 + 0.5 - 2)$
5. 덧셈을 한다: $1 + 1 + 0.5 = 2.5$
6. 뺄셈을 한다: $2.5 - 2 = 0.5$
7. [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)의 뜻: 음수는 0으로 바꾸고, 0 이상은 그대로 둔다: $\mathrm{ReLU}(0.5) = 0.5$
8. 비교: 셋째 좌표가 2.0이면 $f_{10}(2.0) = 0$이라 $1 + 1 + 0 - 2 = 0$, $h_c = 0$

**답**: $h_{10}(0.5, 0.5, 1.25) = 0.5$ (상자 경계의 경사 위)

</details>

<details><summary>내 이해</summary>



</details>

- **조립**: 상자를 격자점 $u_i$로 옮기고([가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)·[편향](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#bias-%ED%8E%B8%ED%96%A5) 조정) 높이 $\alpha_i$를 곱해 더한다.
$$f(x) = \sum_i \alpha_i\, h_c(x - u_i)$$

<details><summary>기호 풀이</summary>

| 기호 | 읽는 법 | 뜻 | 예제 값 |
| --- | --- | --- | --- |
| $f(x)$ | 에프 엑스 | 상자들을 쌓은 3층 망의 출력 | 1.5 |
| $\sum_i$ | 시그마 아이 | 모든 상자를 더함 | 2개 |
| $\alpha_i$ | 알파 아이 | $i$번째 상자 높이(셋째 [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)) | $\alpha_2 = 1.5$ |
| $h_c$ | 에이치 씨 | 상자 하나 ([초직사각형](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#hyperrectangle-%EC%B4%88%EC%A7%81%EC%82%AC%EA%B0%81%ED%98%95)) | 1 |
| $u_i$ | 유 아이 | $i$번째 상자의 위치(격자점) | $u_2 = 2$ |
| $x - u_i$ | 엑스 빼기 유 아이 | 상자를 $u_i$로 옮긴 효과 | 0.5 |

</details>

<details><summary>예제와 풀이</summary>

**문제**: 입력이 1개($d = 1$)이고 $c = 100$일 때, 0에 높이 0.5짜리 상자, 2에 높이 1.5짜리 상자를 둔 망에 2.5를 넣으면 출력은?

**주어진 값**:
- $d = 1$ (입력 차원, 이때 $h_c = f_c$)
- $c = 100$ (경사의 가파름)
- $x = 2.5$ (입력)
- $u_1 = 0$, $\alpha_1 = 0.5$ (상자 1의 위치와 높이)
- $u_2 = 2$, $\alpha_2 = 1.5$ (상자 2의 위치와 높이)

**풀이**
1. 식을 그대로 쓴다: $f(x) = \sum_i \alpha_i\, h_c(x - u_i)$
2. 값을 대입한다: $f(2.5) = 0.5\, h_{100}(2.5 - 0) + 1.5\, h_{100}(2.5 - 2)$
3. 괄호 안 뺄셈을 한다: $2.5 - 0 = 2.5$, $2.5 - 2 = 0.5$
4. 상자 값을 읽는다: $h_{100}(2.5) = 0$ (상자 [0, 1] 밖), $h_{100}(0.5) = 1$ (상자 안)
5. 곱셈을 한다: $0.5 \times 0 = 0$
6. 곱셈을 한다: $1.5 \times 1 = 1.5$
7. 덧셈을 한다: $0 + 1.5 = 1.5$

**답**: $f(2.5) = 1.5$ (둘째 상자의 높이)

</details>

<details><summary>내 이해</summary>



</details>

  [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)가 두 번(직사각형, [초직사각형](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#hyperrectangle-%EC%B4%88%EC%A7%81%EC%82%AC%EA%B0%81%ED%98%95)) 들어가고 마지막 [가중합](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weighted-sum-%EA%B0%80%EC%A4%91%ED%95%A9)이 선형이므로 **[가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98) 행렬 3개, [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 2번**인 3층 망이다.

![MM13_fig_relu_rect.png](../../attachments/MM13_fig_relu_rect.png)
*▲ 직접 그린 그림: 왼쪽은 $c$에 따른 $f_c(x)$, 가운데는 $f_c(x_1) + f_c(x_2)$의 십자 모양, 오른쪽은 1을 빼고 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)를 씌워 남은 정사각형 ($c = 100$)*

## 2. 필요한 이유

[MM12_직사각형_근사와_차원의_저주](../../%EC%88%98%ED%95%99%EA%B0%9C%EB%85%90/06_%EA%B7%BC%EC%82%AC_%EC%9D%B4%EB%A1%A0/MM12_%EC%A7%81%EC%82%AC%EA%B0%81%ED%98%95_%EA%B7%BC%EC%82%AC%EC%99%80_%EC%B0%A8%EC%9B%90%EC%9D%98_%EC%A0%80%EC%A3%BC.md)에서 "상자 $(L/\varepsilon)^d$개면 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) $\varepsilon$"을 보였다. 남은 일은 상자 하나를 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 망으로 만드는 것이다. [Lec03 7.3](../../Lec03_%EA%B7%BC%EC%82%AC_%EC%9D%B4%EB%A1%A0.md#73-3%EB%8B%A8%EA%B3%84-relu%EB%A1%9C-%EC%A7%81%EC%82%AC%EA%B0%81%ED%98%95-%EB%A7%8C%EB%93%A4%EA%B8%B0)에서 강사는 그래프 도구로 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)를 하나씩 더하며 이 직사각형을 직접 만들어 보이고, [7.4](../../Lec03_%EA%B7%BC%EC%82%AC_%EC%9D%B4%EB%A1%A0.md#74-%EC%B4%88%EC%A7%81%EC%82%AC%EA%B0%81%ED%98%95%EA%B3%BC-%EC%A1%B0%EB%A6%BD)에서 상자를 쌓아 정리를 완성한다.

## 3. 숫자로 보기

### 3.1 $c$에 따른 $f_c(x)$ 값

| $x$ | $c = 1$ | $c = 10$ | $c = 100$ | 목표(직사각형) |
|---|---|---|---|---|
| -0.5 | 0 | 0 | 0 | 0 |
| 0.05 | 0.05 | 0.5 | 1 | 1 |
| 0.5 | 0.5 | 1 | 1 | 1 |
| 1.0 | 1 | 1 | 1 | 1 |
| 1.15 | 1 | 1 | 0 | 0 |
| 1.25 | 1 | 0.5 | 0 | 0 |
| 2.0 | 1 | 0 | 0 | 0 |

- $c = 1$이면 계단처럼 올라가기만 하고 내려오지 않는다(내려가는 구간이 $x = 3 \sim 4$라서 표 밖).
- $c = 10$이면 0~0.1에서 올라가고 1.2~1.3에서 내려온다. $c = 100$이면 거의 직사각형이다.

### 3.2 직사각형과의 차이 넓이

| $c$ | 올라가는 쪽 빈틈 | 평평하게 더 간 부분 | 내려가는 쪽 | 합 $= 3/c$ |
|---|---|---|---|---|
| 10 | $\frac{1}{2c} = 0.05$ | $\frac{2}{c} = 0.2$ | $\frac{1}{2c} = 0.05$ | 0.3 |
| 100 | 0.005 | 0.02 | 0.005 | 0.03 |
| 1000 | 0.0005 | 0.002 | 0.0005 | 0.003 |

- $c$를 10배 키우면 차이가 10분의 1. $c \to \infty$이면 0이다.
- (강의에서 설명하지 않은 부분으로, 정리자의 해석) 강사는 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8) 상한이 왜 $\varepsilon$이 아니라 $2\varepsilon$인지 말하지 않았다. 이렇게 읽으면 맞아떨어진다. 막대(상자) 근사 $f_{\text{box}}$의 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)는 $\varepsilon$ 이하이고, $c$를 충분히 크게 하면 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 망 $f$와 $f_{\text{box}}$의 차이 넓이도 $\varepsilon$보다 작게 만들 수 있다(위 표처럼 차이가 $c$에 반비례해 줄고, 상자 수는 유한하다). 삼각 부등식 $|f - g| \le |f - f_{\text{box}}| + |f_{\text{box}} - g|$를 적분하면 전체 [오차](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#error-%EC%98%A4%EC%B0%A8)는 $\varepsilon + \varepsilon = 2\varepsilon$보다 작다.

### 3.3 2차원 상자 ($c = 10$, $d = 2$)

| 점 $(x_1, x_2)$ | $f_c(x_1) + f_c(x_2)$ | $-(d - 1) = -1$ 후 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) |
|---|---|---|
| (0.5, 0.5) 상자 안 | 1 + 1 = 2 | **1** |
| (0.5, 2.0) 한 좌표만 안 | 1 + 0 = 1 | 0 |
| (2.0, 2.0) 둘 다 밖 | 0 | 0 |
| (1.25, 0.5) 경계 경사 | 0.5 + 1 = 1.5 | 0.5 |
| (1.25, 1.25) 경계 경사 둘 | 0.5 + 0.5 = 1 | 0 |

<details><summary>기호 풀이</summary>

| 기호 | 읽는 법 | 뜻 | 예제 값 |
| --- | --- | --- | --- |
| $c$ | 씨 | 경사의 가파름. 클수록 직사각형에 가깝다 | 10 |
| $f_c(x)$ | 에프 씨 엑스 | [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 4개로 만든 1차원 직사각형 | $f_{10}(0.5) = 1$ |
| $h_c(x)$ | 에이치 씨 엑스 | $d$차원 상자([초직사각형](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#hyperrectangle-%EC%B4%88%EC%A7%81%EC%82%AC%EA%B0%81%ED%98%95)) | $h_{10}(0.5, 0.5) = 1$ |
| $d - 1$ | 디 빼기 일 | 상자 밖을 잘라내는 문턱 | 1 ($d = 2$) |
| $u_i$ | 유 아이 | $i$번째 상자의 위치(격자점) | - |
| $\alpha_i$ | 알파 아이 | $i$번째 상자의 높이(셋째 [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)) | $g(u_i)$ |

</details>

## 4. 코드로 확인

```python
# [함수] import: NumPy를 np로 불러온다. maximum, linspace, trapezoid가 여기서 온다
import numpy as np

# [함수] np.maximum(0, z): ReLU. [변수] relu: 모든 부품의 기본 단위
relu = lambda z: np.maximum(0, z)
# [변수] fc: ReLU 4개로 만든 직사각형 f_c(x). 계수 +1, -1, -1, +1 (본문 1절 표, Lec03 7.3 슬라이드)
fc = lambda v, c: relu(c * v) - relu(c * v - 1) - relu(c * (v - 1) - 2) + relu(c * (v - 1) - 3)
# [변수] c: 경사의 가파름. 1, 10, 100으로 키우며 직사각형에 가까워지는지 본다 (본문 3.1 표)
for c in (1, 10, 100):
    # [함수] round(float(...), 3): 숫자 하나로 바꿔 소수 셋째 자리까지
    # [연결] → c = 100이면 [0, 1, 1, 1, 0, 0, 0]으로 목표 직사각형과 같다
    print(c, [round(float(fc(v, c)), 3) for v in (-0.5, 0.05, 0.5, 1.0, 1.15, 1.25, 2.0)])

# [함수] np.linspace(-1, 3, 4000001): 직사각형 주변을 촘촘히 나눈 점. [변수] xx: 넓이를 잴 위치
xx = np.linspace(-1, 3, 4_000_001)
# [변수] ind: 목표 직사각형(지시 함수). 0 ≤ x ≤ 1이면 1, 아니면 0
ind = ((xx >= 0) & (xx <= 1)).astype(float)
# [변수] c: 10배씩 키운다
for c in (10, 100, 1000):
    # [함수] np.trapezoid: |f_c - 직사각형|의 넓이 = 적분 오차
    # [연결] → 0.3, 0.03, 0.003 = 3/c (본문 3.2 표)
    print(c, round(np.trapezoid(np.abs(fc(xx, c) - ind), xx), 4))

# [변수] p: 2차원 점들 (본문 3.3 표)
for p in ((0.5, 0.5), (0.5, 2.0), (2.0, 2.0), (1.25, 0.5), (1.25, 1.25)):
    # [변수] s: 좌표마다 만든 직사각형의 합 (c = 10). 상자 안이면 2
    s = fc(p[0], 10) + fc(p[1], 10)
    # [함수] relu(s - 1): d - 1 = 1을 빼고 ReLU. [연결] → 상자 안만 1이 남는다 (Lec03 7.4의 h_c)
    print(p, round(float(s), 3), round(float(relu(s - 1)), 3))
```

**실행 결과**:

```text
1 [0.0, 0.05, 0.5, 1.0, 1.0, 1.0, 1.0]
10 [0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0]
100 [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
10 0.3
100 0.03
1000 0.003
(0.5, 0.5) 2.0 1.0
(0.5, 2.0) 1.0 0.0
(2.0, 2.0) 0.0 0.0
(1.25, 0.5) 1.5 0.5
(1.25, 1.25) 1.0 0.0
```

> [!WARNING]
> **⚠ 주의**
> - **$c \to \infty$는 [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)가 무한히 커진다**는 뜻이다. 실제 [학습](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#training-%ED%95%99%EC%8A%B5)에서 [가중치](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#weight-%EA%B0%80%EC%A4%91%EC%B9%98)가 폭발하면 무언가 잘못된 신호다. 강의도 이 구성이 "증명을 위한 장치"이지 신경망이 실제로 배우는 방식은 아니라고 했다.
> - 슬라이드 식은 오른쪽 끝이 $1 + 2/c$에서 꺾여 직사각형보다 조금 길다. $c \to \infty$에서는 사라지는 차이다.
> - 뉴런 수: 상자 하나에 첫 [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) $4d$개와 둘째 [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 1개, 합 $4d + 1$개다. 강의의 정리는 $4d$로 셌다. 차수(크기의 규모)는 같다.

## 5. 개념 재확인

<details><summary>1. $c = 10$일 때 $f_c(0.03)$과 $f_c(1.28)$은?</summary>

| 단계 | 하는 일 | 계산 | 결과 |
|---|---|---|---|
| 1 | $x = 0.03$의 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 4개 | 0.3, 0, 0, 0 | - |
| 2 | 계수 곱해 합 | $0.3 - 0 - 0 + 0$ | 0.3 |
| 3 | $x = 1.28$의 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 4개 | 12.8, 11.8, 0.8, 0 | - |
| 4 | 계수 곱해 합 | $12.8 - 11.8 - 0.8 + 0$ | 0.2 |

**① 쓰는 식**
$$f_c(x) = \mathrm{ReLU}(cx) - \mathrm{ReLU}(cx - 1) - \mathrm{ReLU}(c(x-1) - 2) + \mathrm{ReLU}(c(x-1) - 3)$$
- 위 식 ← 네 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 값은 ② ($x = 0.03$)와 ④ ($x = 1.28$), 계수를 곱한 합은 ③과 ⑤에서 쓴다.
```python
# [변수] relu: ReLU. c = 10: 문제에서 준 가파름
relu, c = (lambda z: np.maximum(0, z)), 10                        # ①
# [변수] parts: 네 ReLU의 값을 목록으로 돌려주는 함수 (① 식의 네 항)
parts = lambda x: [float(relu(c * x)), float(relu(c * x - 1)), float(relu(c * (x - 1) - 2)), float(relu(c * (x - 1) - 3))]
# [연결] 단계 ② x = 0.03 → [0.3, 0.0, 0.0, 0.0] (첫 ReLU만 켜짐, 올라가는 중)
print([round(p, 3) for p in parts(0.03)])                         # ②
# [함수] a - b - e + f: 계수 +1, -1, -1, +1. [연결] 단계 ③ → 0.3
a, b, e, f = parts(0.03); print(round(a - b - e + f, 3))           # ③
# [연결] 단계 ④ x = 1.28 → [12.8, 11.8, 0.8, 0.0] (셋째까지 켜짐, 내려가는 중)
print([round(p, 3) for p in parts(1.28)])                         # ④
# [연결] 단계 ⑤ 합 → 0.2
a, b, e, f = parts(1.28); print(round(a - b - e + f, 3))           # ⑤
```

**주어진 값**
- $c = 10$ (경사의 가파름)
- $x = 0.03$, $x = 1.28$ (입력 두 개)
- 계수 $(+1, -1, -1, +1)$, [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)는 음수를 0으로 바꾼다

**② $x = 0.03$의 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 4개**
1. 식을 그대로 쓴다: $\mathrm{ReLU}(cx),\ \mathrm{ReLU}(cx - 1),\ \mathrm{ReLU}(c(x-1) - 2),\ \mathrm{ReLU}(c(x-1) - 3)$
2. 값을 대입한다: $\mathrm{ReLU}(10 \times 0.03),\ \mathrm{ReLU}(10 \times 0.03 - 1),\ \mathrm{ReLU}(10 \times (0.03 - 1) - 2),\ \mathrm{ReLU}(10 \times (0.03 - 1) - 3)$
3. 괄호 안 뺄셈을 한다: 0.03 − 1 = −0.97 이므로 $\mathrm{ReLU}(10 \times 0.03),\ \mathrm{ReLU}(10 \times 0.03 - 1),\ \mathrm{ReLU}(10 \times (-0.97) - 2),\ \mathrm{ReLU}(10 \times (-0.97) - 3)$
4. 곱셈을 한다: 10 × 0.03 = 0.3, 10 × (−0.97) = −9.7 이므로 $\mathrm{ReLU}(0.3),\ \mathrm{ReLU}(0.3 - 1),\ \mathrm{ReLU}(-9.7 - 2),\ \mathrm{ReLU}(-9.7 - 3)$
5. 뺄셈을 한다: 0.3 − 1 = −0.7, −9.7 − 2 = −11.7, −9.7 − 3 = −12.7
6. [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)를 적용한다 (음수는 0): $(0.3,\ 0,\ 0,\ 0)$

**③ 합**
1. 식을 그대로 쓴다: $f_c(x) = (+1) \times r_1 + (-1) \times r_2 + (-1) \times r_3 + (+1) \times r_4$ ($r_1 \sim r_4$는 위 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 4개의 출력)
2. 값을 대입한다: $(+1) \times 0.3 + (-1) \times 0 + (-1) \times 0 + (+1) \times 0$
3. 곱셈을 한다: $0.3 - 0 - 0 + 0$
4. 계산한다: 0.3 (0~0.1 구간의 경사로 위)

**④ $x = 1.28$의 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 4개**
1. 식을 그대로 쓴다: $\mathrm{ReLU}(cx),\ \mathrm{ReLU}(cx - 1),\ \mathrm{ReLU}(c(x-1) - 2),\ \mathrm{ReLU}(c(x-1) - 3)$
2. 값을 대입한다: $\mathrm{ReLU}(10 \times 1.28),\ \mathrm{ReLU}(10 \times 1.28 - 1),\ \mathrm{ReLU}(10 \times (1.28 - 1) - 2),\ \mathrm{ReLU}(10 \times (1.28 - 1) - 3)$
3. 괄호 안 뺄셈을 한다: 1.28 − 1 = 0.28
4. 곱셈을 한다: 10 × 1.28 = 12.8, 10 × 0.28 = 2.8 이므로 $\mathrm{ReLU}(12.8),\ \mathrm{ReLU}(12.8 - 1),\ \mathrm{ReLU}(2.8 - 2),\ \mathrm{ReLU}(2.8 - 3)$
5. 뺄셈을 한다: 12.8 − 1 = 11.8, 2.8 − 2 = 0.8, 2.8 − 3 = −0.2
6. [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)를 적용한다 (음수는 0): $(12.8,\ 11.8,\ 0.8,\ 0)$

**⑤ 합**
1. 식을 그대로 쓴다: $f_c(x) = (+1) \times r_1 + (-1) \times r_2 + (-1) \times r_3 + (+1) \times r_4$
2. 값을 대입한다: $(+1) \times 12.8 + (-1) \times 11.8 + (-1) \times 0.8 + (+1) \times 0$
3. 곱셈을 한다: $12.8 - 11.8 - 0.8 + 0$
4. 앞에서부터 뺄셈을 한다: 12.8 − 11.8 = 1, 1 − 0.8 = 0.2, 0.2 + 0 = 0.2 (1.2~1.3 구간에서 내려가는 중)

**답**: $f_{10}(0.03) = $ **0.3**, $f_{10}(1.28) = $ **0.2**.

</details>

<details><summary>2. $d = 3$, $c = 1000$일 때 점 (0.5, 0.5, 0.5)와 (0.5, 0.5, 2)의 $h_c$ 값은? 상자 하나에 드는 ReLU는 몇 개인가?</summary>

| 단계 | 하는 일 | 계산 | 결과 |
|---|---|---|---|
| 1 | 첫 점의 직사각형 합 | $1 + 1 + 1$ | 3 |
| 2 | 첫 점의 $h_c$ | $\mathrm{ReLU}(3 - 2)$ | 1 |
| 3 | 둘째 점의 합 | $1 + 1 + 0$ | 2 |
| 4 | 둘째 점의 $h_c$ | $\mathrm{ReLU}(2 - 2)$ | 0 |
| 5 | [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 수 | $4 × 3 + 1$ | 13 |

**① 쓰는 식**
$$h_c(x) = \mathrm{ReLU}\Big(\sum_{i=1}^{d} f_c(x_i) - (d - 1)\Big)$$
- 위 식 ← 합은 ②와 ④, [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)는 ③과 ⑤, 뉴런 세기는 ⑥ (첫 [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) $4d$ + 둘째 [층](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#layer-%EC%B8%B5) 1)에서 쓴다.
```python
# [변수] relu, fc: ReLU와 직사각형 f_c (본문 1절). d = 3, c = 1000: 문제에서 준 값
relu = lambda z: np.maximum(0, z)
fc = lambda v, c: relu(c * v) - relu(c * v - 1) - relu(c * (v - 1) - 2) + relu(c * (v - 1) - 3)
d, c = 3, 1000                                              # ①
# [변수] s1 = 3.0: 세 좌표 모두 상자 안
s1 = sum(float(fc(v, c)) for v in (0.5, 0.5, 0.5))          # ②
# [연결] 단계 ③ ReLU(3 - 2) → 3.0 1.0
print(s1, float(relu(s1 - (d - 1))))                        # ③
# [변수] s2 = 2.0: 셋째 좌표 2가 상자 밖
s2 = sum(float(fc(v, c)) for v in (0.5, 0.5, 2.0))          # ④
# [연결] 단계 ⑤ ReLU(2 - 2) → 2.0 0.0
print(s2, float(relu(s2 - (d - 1))))                        # ⑤
# [연결] 단계 ⑥ 첫 층 4d + 둘째 층 1 → 13
print(4 * d + 1)                                            # ⑥
```

**주어진 값**
- $d = 3$ (입력 차원), $c = 1000$ (경사의 가파름)
- 점 A = (0.5, 0.5, 0.5), 점 B = (0.5, 0.5, 2)
- $f_c(x) = \mathrm{ReLU}(cx) - \mathrm{ReLU}(cx - 1) - \mathrm{ReLU}(c(x-1) - 2) + \mathrm{ReLU}(c(x-1) - 3)$

**② 첫 점의 합**
1. 식을 그대로 쓴다: $\sum_{i=1}^{3} f_c(x_i) = f_c(0.5) + f_c(0.5) + f_c(0.5)$
2. $f_c(0.5)$의 값을 대입한다: $\mathrm{ReLU}(1000 \times 0.5) - \mathrm{ReLU}(1000 \times 0.5 - 1) - \mathrm{ReLU}(1000 \times (0.5 - 1) - 2) + \mathrm{ReLU}(1000 \times (0.5 - 1) - 3)$
3. 곱셈과 뺄셈을 한다: 1000 × 0.5 = 500, 500 − 1 = 499, 1000 × (−0.5) = −500, −500 − 2 = −502, −500 − 3 = −503 이므로 $\mathrm{ReLU}(500) - \mathrm{ReLU}(499) - \mathrm{ReLU}(-502) + \mathrm{ReLU}(-503)$
4. [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)를 적용한다 (음수는 0): $500 - 499 - 0 + 0$
5. 계산한다: 500 − 499 = 1 이므로 $f_c(0.5) = 1$
6. 세 좌표를 더한다: 1 + 1 + 1 = 3

**③ 첫 점의 $h_c$**
1. 식을 그대로 쓴다: $h_c(x) = \mathrm{ReLU}\Big(\sum_{i=1}^{d} f_c(x_i) - (d - 1)\Big)$
2. 값을 대입한다: $\mathrm{ReLU}(3 - (3 - 1))$
3. 괄호 안 뺄셈을 한다: 3 − 1 = 2 이므로 $\mathrm{ReLU}(3 - 2)$
4. 뺄셈과 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)를 한다: 3 − 2 = 1 이므로 $\mathrm{ReLU}(1) = 1$ (상자 안)

**④ 둘째 점의 합**
1. 식을 그대로 쓴다: $f_c(0.5) + f_c(0.5) + f_c(2)$ (앞의 둘은 ②에서 1)
2. $f_c(2)$의 값을 대입한다: $\mathrm{ReLU}(1000 \times 2) - \mathrm{ReLU}(1000 \times 2 - 1) - \mathrm{ReLU}(1000 \times (2 - 1) - 2) + \mathrm{ReLU}(1000 \times (2 - 1) - 3)$
3. 곱셈과 뺄셈을 한다: 1000 × 2 = 2000, 2000 − 1 = 1999, 1000 × 1 = 1000, 1000 − 2 = 998, 1000 − 3 = 997 이므로 $\mathrm{ReLU}(2000) - \mathrm{ReLU}(1999) - \mathrm{ReLU}(998) + \mathrm{ReLU}(997)$
4. [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)를 적용한다 (모두 양수라 그대로): $2000 - 1999 - 998 + 997$
5. 앞에서부터 계산한다: 2000 − 1999 = 1, 1 − 998 = −997, −997 + 997 = 0 이므로 $f_c(2) = 0$
6. 세 좌표를 더한다: 1 + 1 + 0 = 2

**⑤ 둘째 점의 $h_c$**
1. 식을 그대로 쓴다: $h_c(x) = \mathrm{ReLU}\Big(\sum_{i=1}^{d} f_c(x_i) - (d - 1)\Big)$
2. 값을 대입한다: $\mathrm{ReLU}(2 - (3 - 1))$
3. 괄호 안 뺄셈을 한다: 3 − 1 = 2 이므로 $\mathrm{ReLU}(2 - 2)$
4. 뺄셈과 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8)를 한다: 2 − 2 = 0 이므로 $\mathrm{ReLU}(0) = 0$ (하나라도 밖이면 잘린다)

**⑥ [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 수**
1. 식을 쓴다: (좌표마다 직사각형의 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 4개) × $d$ + (자르는 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) 1개) $= 4d + 1$
2. 값을 대입한다: $4 \times 3 + 1$
3. 계산한다: 4 × 3 = 12, 12 + 1 = 13

**답**: (0.5, 0.5, 0.5)는 **1**, (0.5, 0.5, 2)는 **0**, 상자 하나에 [ReLU](../../00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md#relu-%EB%A0%90%EB%A3%A8) **13개**.

</details>

## 예제 풀기

[예제 10문제 풀기](../../%EC%98%88%EC%A0%9C/06_%EA%B7%BC%EC%82%AC_%EC%9D%B4%EB%A1%A0/%EC%98%88%EC%A0%9C_MM13_ReLU%EB%A1%9C_%EB%A7%8C%EB%93%9C%EB%8A%94_%EC%A7%81%EC%82%AC%EA%B0%81%ED%98%95.md)

<details><summary>내 이해</summary>



</details>
