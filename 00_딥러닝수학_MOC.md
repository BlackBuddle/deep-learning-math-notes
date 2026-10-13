---
title: 딥러닝 수학 — Map of Content
type: moc
tags: [moc, deep-learning-math]
created: 2026-10-13
updated: 2026-10-13
---

# 딥러닝 수학 — Map of Content

딥러닝 강의 예제 노트북 10개(기초 7개, CNN 3개)를 따라가며, 각 예제를 이해하는 데 필요한 수학을 모은다.

- **정리 방식: 역추적형.** 딥러닝 개념 노트(`D`)에서 출발하고, 거기서 막히는 수학을 수학 노트(`M`)로 따로 푼다.
- 설명 방식: 말로 먼저 → 작은 숫자로 손계산 → 수식·코드 → 3줄 요약 순서로, 한 줄에 한 걸음씩 쓴다.
- 용어가 막히면: [00_딥러닝_용어사전](00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md) (용어마다 뜻·개념·관련 노트 링크)
- 새 딥러닝 개념 노트는 [99_템플릿_딥러닝개념](99_%ED%85%9C%ED%94%8C%EB%A6%BF/99_%ED%85%9C%ED%94%8C%EB%A6%BF_%EB%94%A5%EB%9F%AC%EB%8B%9D%EA%B0%9C%EB%85%90.md) 구조로 만든다.

## 폴더 구조

| 폴더 | 내용 | 파일 이름 |
|---|---|---|
| (이 폴더) | 목차, [용어 사전](00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%9A%A9%EC%96%B4%EC%82%AC%EC%A0%84.md) | `00_...` |
| `01_개념노트` | 딥러닝 개념 (예제에서 출발, 역추적형) | `D01` ~ `D09` |
| `02_수학노트` | 개념에 필요한 수학 | `M01` ~ `M10` |
| `03_예제풀이` | 예제 노트북 셀별 해설 (주석 단 코드·결과·그래프) | `E01` ~ `E10` |
| `04_Keras` | Keras 개념과 예제 모음 | `K01`, `K02` |
| `99_템플릿` | 새 개념 노트 틀 | `99_...` |
| `attachments` | 노트북에서 꺼낸 그래프 이미지 | `E##_cell#_#.png` |

> 파일 이름 앞 글자(D·M·E·K + 번호)로 종류와 순서를 알 수 있다.

## 1. 읽는 순서

1. 수학부터 보고 싶으면 → [M01_함수와_일차함수](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M01_%ED%95%A8%EC%88%98%EC%99%80_%EC%9D%BC%EC%B0%A8%ED%95%A8%EC%88%98.md) → [M02_시그마와_평균](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M02_%EC%8B%9C%EA%B7%B8%EB%A7%88%EC%99%80_%ED%8F%89%EA%B7%A0.md) → [M03_미분_기울기](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M03_%EB%AF%B8%EB%B6%84_%EA%B8%B0%EC%9A%B8%EA%B8%B0.md) → [M04_편미분](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M04_%ED%8E%B8%EB%AF%B8%EB%B6%84.md) → [M05_벡터와_행렬](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M05_%EB%B2%A1%ED%84%B0%EC%99%80_%ED%96%89%EB%A0%AC.md) → [M06_지수와_e](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M06_%EC%A7%80%EC%88%98%EC%99%80_e.md) → [M07_로그](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M07_%EB%A1%9C%EA%B7%B8.md) → [M08_확률_기초](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M08_%ED%99%95%EB%A5%A0_%EA%B8%B0%EC%B4%88.md)
2. 개념부터 보고 싶으면 → [D01_퍼셉트론](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D01_%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0.md) → [D02_경사하강법으로_직선찾기](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D02_%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95%EC%9C%BC%EB%A1%9C_%EC%A7%81%EC%84%A0%EC%B0%BE%EA%B8%B0.md) → [D03_Dense층과_정규화](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D03_Dense%EC%B8%B5%EA%B3%BC_%EC%A0%95%EA%B7%9C%ED%99%94.md) → [D04_시그모이드와_이진분류](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D04_%EC%8B%9C%EA%B7%B8%EB%AA%A8%EC%9D%B4%EB%93%9C%EC%99%80_%EC%9D%B4%EC%A7%84%EB%B6%84%EB%A5%98.md) → [D05_소프트맥스와_다중분류](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D05_%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4%EC%99%80_%EB%8B%A4%EC%A4%91%EB%B6%84%EB%A5%98.md) → [D06_이미지분류와_MNIST](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D06_%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%B6%84%EB%A5%98%EC%99%80_MNIST.md) → [D07_합성곱과_풀링](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D07_%ED%95%A9%EC%84%B1%EA%B3%B1%EA%B3%BC_%ED%92%80%EB%A7%81.md) → [D08_ReLU와_드롭아웃](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D08_ReLU%EC%99%80_%EB%93%9C%EB%A1%AD%EC%95%84%EC%9B%83.md) (막히면 체크리스트의 M 노트로)
3. 코드부터 보고 싶으면 → 아래 **예제 풀이**(`E`)를 순서대로. 셀마다 주석 단 코드·실행 결과·그래프가 있다.
4. 보충 개념 → [D09_오차와_비용함수](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D09_%EC%98%A4%EC%B0%A8%EC%99%80_%EB%B9%84%EC%9A%A9%ED%95%A8%EC%88%98.md) (오차·비용·최적값), [M09_최솟값과_볼록함수](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M09_%EC%B5%9C%EC%86%9F%EA%B0%92%EA%B3%BC_%EB%B3%BC%EB%A1%9D%ED%95%A8%EC%88%98.md) (지역/전역 최솟값), [M10_엔트로피](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M10_%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC.md) (cross-entropy의 뜻)
5. Keras만 따로 → [K01_Keras_기본개념](04_Keras/K01_Keras_%EA%B8%B0%EB%B3%B8%EA%B0%9C%EB%85%90.md) → [K02_Keras_예제모음](04_Keras/K02_Keras_%EC%98%88%EC%A0%9C%EB%AA%A8%EC%9D%8C.md)

> 한 예제를 공부할 때는 **E(코드) → D(개념) → M(막히는 수학)** 순서로 오가면 된다. 각 노트 맨 위에 서로의 링크가 있다.

## 예제 풀이 (노트북 셀별 해설)

| 노트 | 예제 노트북 | 핵심 | 결과 |
|---|---|---|---|
| [E01_퍼셉트론_예제](03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E01_%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0_%EC%98%88%EC%A0%9C.md) | EX\_01 Perceptron | 가중합 + 계단 함수 | s = 42 → 1 |
| [E02_가설최적화_예제](03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E02_%EA%B0%80%EC%84%A4%EC%B5%9C%EC%A0%81%ED%99%94_%EC%98%88%EC%A0%9C.md) | EX\_02 Optimize Hypothesis | 경사하강법으로 직선 찾기 | W 2.0, B 0.997 |
| [E03_혈압_선형회귀_예제](03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E03_%ED%98%88%EC%95%95_%EC%84%A0%ED%98%95%ED%9A%8C%EA%B7%80_%EC%98%88%EC%A0%9C.md) | EX\_03 BP Linear Regression | 실제 데이터, 작은 학습률 | 50세 → 126.2 |
| [E04_다변수_다출력_회귀_예제](03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E04_%EB%8B%A4%EB%B3%80%EC%88%98_%EB%8B%A4%EC%B6%9C%EB%A0%A5_%ED%9A%8C%EA%B7%80_%EC%98%88%EC%A0%9C.md) | EX\_04 N-variable M-output | Keras, Min-Max 정규화, Dense | loss 5.36 |
| [E05_혈압_로지스틱회귀_예제](03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E05_%ED%98%88%EC%95%95_%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1%ED%9A%8C%EA%B7%80_%EC%98%88%EC%A0%9C.md) | EX\_05 Logistic Regression | 시그모이드, binary cross-entropy | 정확도 0.7778 |
| [E06_혈압_소프트맥스분류_예제](03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E06_%ED%98%88%EC%95%95_%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4%EB%B6%84%EB%A5%98_%EC%98%88%EC%A0%9C.md) | EX\_06 Softmax | 원-핫, 소프트맥스, argmax | 정확도 0.7222 |
| [E07_MNIST_분류_예제](03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E07_MNIST_%EB%B6%84%EB%A5%98_%EC%98%88%EC%A0%9C.md) | EX\_07 MNIST | Flatten, Adam, 시험 데이터 | 시험 0.9257 |
| [E08_CNN_1Conv_MNIST_예제](03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E08_CNN_1Conv_MNIST_%EC%98%88%EC%A0%9C.md) | CNN EX\_01 | 합성곱, 맥스 풀링, validation | 시험 0.9710 |
| [E09_CNN_3Conv_MNIST_예제](03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E09_CNN_3Conv_MNIST_%EC%98%88%EC%A0%9C.md) | CNN EX\_02 | 층 쌓기, 특징 맵 | 시험 0.9889 |
| [E10_CNN_ReLU_Dropout_예제](03_%EC%98%88%EC%A0%9C%ED%92%80%EC%9D%B4/E10_CNN_ReLU_Dropout_%EC%98%88%EC%A0%9C.md) | CNN EX\_03 | ReLU, 드롭아웃 | 시험 0.9915 |

## 2. 예제 → 딥러닝 개념 → 필요한 수학 (역추적표)

| 예제 | 딥러닝 개념 | 필요한 수학 | 개념 노트 |
|---|---|---|---|
| EX\_01 Perceptron | 퍼셉트론, 가중합, 계단 함수 | 함수, Σ | [D01_퍼셉트론](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D01_%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0.md) ✅ |
| EX\_02 Optimize Hypothesis | 가설(직선), 비용 함수(MSE), 경사하강법, 학습률 | 일차함수, Σ·평균, 미분, 편미분 | [D02_경사하강법으로_직선찾기](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D02_%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95%EC%9C%BC%EB%A1%9C_%EC%A7%81%EC%84%A0%EC%B0%BE%EA%B8%B0.md) ✅ |
| EX\_03 BP Linear Regression | 실제 데이터로 선형 회귀, 학습률 조절, 예측 | EX\_02와 같음 | [D02_경사하강법으로_직선찾기](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D02_%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95%EC%9C%BC%EB%A1%9C_%EC%A7%81%EC%84%A0%EC%B0%BE%EA%B8%B0.md) ✅ |
| EX\_04 N-variable M-output | 입력·출력 여러 개, Min-Max 정규화, Dense 층, 파라미터 수 | 벡터·행렬, 비율 | [D03_Dense층과_정규화](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D03_Dense%EC%B8%B5%EA%B3%BC_%EC%A0%95%EA%B7%9C%ED%99%94.md) ✅ |
| EX\_05 Logistic Regression | 시그모이드, 이진 분류, 결정 경계, binary cross-entropy, 정확도 | 지수·e, 로그, 확률 | [D04_시그모이드와_이진분류](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D04_%EC%8B%9C%EA%B7%B8%EB%AA%A8%EC%9D%B4%EB%93%9C%EC%99%80_%EC%9D%B4%EC%A7%84%EB%B6%84%EB%A5%98.md) ✅ |
| EX\_06 Softmax Classification | 소프트맥스, 원-핫 인코딩, argmax, categorical cross-entropy | 지수, 로그, 확률(합이 1) | [D05_소프트맥스와_다중분류](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D05_%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4%EC%99%80_%EB%8B%A4%EC%A4%91%EB%B6%84%EB%A5%98.md) ✅ |
| EX\_07 MNIST | 이미지를 숫자 배열로, Flatten, 미니배치, Adam, 학습/시험 데이터 | 배열 모양(shape), 비율 | [D06_이미지분류와_MNIST](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D06_%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%B6%84%EB%A5%98%EC%99%80_MNIST.md) ✅ |
| CNN EX\_01 1 Conv | 합성곱, 필터, 패딩, 맥스 풀링, 가중치 공유 | 격자 곱하고 더하기, 최댓값 | [D07_합성곱과_풀링](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D07_%ED%95%A9%EC%84%B1%EA%B3%B1%EA%B3%BC_%ED%92%80%EB%A7%81.md) ✅ |
| CNN EX\_02 3 Conv | 층 쌓기, 특징 맵, 크기 줄어드는 계산 | 나누기·내림 | [D07_합성곱과_풀링](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D07_%ED%95%A9%EC%84%B1%EA%B3%B1%EA%B3%BC_%ED%92%80%EB%A7%81.md) ✅ |
| CNN EX\_03 ReLU·Dropout | ReLU, 기울기 소실, 드롭아웃, 과적합 | 함수(max), 기울기 곱, 확률 | [D08_ReLU와_드롭아웃](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D08_ReLU%EC%99%80_%EB%93%9C%EB%A1%AD%EC%95%84%EC%9B%83.md) ✅ |

## 3. 수학 노트 (선수 순서)

| 순서  | 노트               | 한 줄                         | 처음 쓰이는 예제    | 상태  |
| --- | ---------------- | --------------------------- | ------------ | --- |
| 1   | [M01_함수와_일차함수](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M01_%ED%95%A8%EC%88%98%EC%99%80_%EC%9D%BC%EC%B0%A8%ED%95%A8%EC%88%98.md) | 넣으면 나오는 기계, 직선 $y = wx + b$ | EX\_01, EX\_02 | ✅   |
| 2   | [M02_시그마와_평균](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M02_%EC%8B%9C%EA%B7%B8%EB%A7%88%EC%99%80_%ED%8F%89%EA%B7%A0.md)  | Σ는 for문, 평균은 더하고 나누기        | EX\_01, EX\_02 | ✅   |
| 3   | [M03_미분_기울기](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M03_%EB%AF%B8%EB%B6%84_%EA%B8%B0%EC%9A%B8%EA%B8%B0.md)   | 조금 움직이면 얼마나 변하나             | EX\_02        | ✅   |
| 4   | [M04_편미분](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M04_%ED%8E%B8%EB%AF%B8%EB%B6%84.md)      | 여러 개 중 하나만 움직이기             | EX\_02        | ✅   |
| 5   | [M05_벡터와_행렬](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M05_%EB%B2%A1%ED%84%B0%EC%99%80_%ED%96%89%EB%A0%AC.md)   | 숫자 묶음을 한 번에 계산, 행렬 곱, shape | EX\_04, EX\_07 | ✅   |
| 6   | [M06_지수와_e](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M06_%EC%A7%80%EC%88%98%EC%99%80_e.md)     | 같은 수를 여러 번 곱하기, $e^x$는 항상 양수 | EX\_05, EX\_06 | ✅   |
| 7   | [M07_로그](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M07_%EB%A1%9C%EA%B7%B8.md)        | 몇 번 곱했는지 거꾸로 묻기, 벌점 $-\ln p$ | EX\_05, EX\_06 | ✅   |
| 8   | [M08_확률_기초](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M08_%ED%99%95%EB%A5%A0_%EA%B8%B0%EC%B4%88.md)     | 0~1 사이 값, 합이 1, 곱셈 규칙       | EX\_05, EX\_06, CNN EX\_03 | ✅   |
| 9   | [M09_최솟값과_볼록함수](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M09_%EC%B5%9C%EC%86%9F%EA%B0%92%EA%B3%BC_%EB%B3%BC%EB%A1%9D%ED%95%A8%EC%88%98.md) | 지역/전역 최솟값, 볼록 함수 | EX\_02, EX\_05, CNN | ✅   |
| 10  | [M10_엔트로피](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M10_%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC.md)      | 놀람의 평균, cross-entropy의 뜻 | EX\_05, EX\_06 | ✅   |

## 4. 보충 개념 노트

| 노트 | 다루는 용어 | 관련 예제 |
|---|---|---|
| [D09_오차와_비용함수](01_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8/D09_%EC%98%A4%EC%B0%A8%EC%99%80_%EB%B9%84%EC%9A%A9%ED%95%A8%EC%88%98.md) | 오차, 평균 오차, 비용 함수, loss와 cost, 최적값, 가설 최적화 | EX\_02, EX\_03 |
| [M09_최솟값과_볼록함수](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M09_%EC%B5%9C%EC%86%9F%EA%B0%92%EA%B3%BC_%EB%B3%BC%EB%A1%9D%ED%95%A8%EC%88%98.md) | 전역/지역 최솟값(minima), 볼록 함수, 로지스틱 회귀에 MSE를 안 쓰는 이유 | EX\_02, EX\_05, CNN |
| [M10_엔트로피](02_%EC%88%98%ED%95%99%EB%85%B8%ED%8A%B8/M10_%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC.md) | 정보량, 엔트로피, cross-entropy, KL 발산 | EX\_05, EX\_06 |

## 5. Keras 정리

| 노트 | 내용 |
|---|---|
| [K01_Keras_기본개념](04_Keras/K01_Keras_%EA%B8%B0%EB%B3%B8%EA%B0%9C%EB%85%90.md) | 손으로 짠 코드와 Keras 대응표, 다섯 단계, 모델 만드는 세 방법, **출력층·loss·정답 짝 표**, 층·활성화·compile·fit·evaluate 정리 |
| [K02_Keras_예제모음](04_Keras/K02_Keras_%EC%98%88%EC%A0%9C%EB%AA%A8%EC%9D%8C.md) | 문제 유형별 최소 코드(실행 기록 있음) + 바꿔 보기 연습(중간층 켜기, Functional API, sparse 정답, EarlyStopping, 저장·불러오기) |

## 6. 예제 환경 메모

- 예제는 Colab에서 실행된 기록이다: NumPy 2.0.2, TensorFlow 2.19.0, Matplotlib 3.10.0
- 노트의 손계산 숫자는 이 PC(NumPy 2.5.4)에서 같은 코드를 다시 돌려 확인했다.
- 예제 풀이 노트(`E`)의 그래프는 노트북에 저장된 출력 이미지를 `attachments/`로 꺼낸 것이다.
