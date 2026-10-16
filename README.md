# MIT 딥러닝 한국어 강의 노트

MIT OpenCourseWare의 대학원 강의 [6.7960 Deep Learning (2024년 가을)](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/)을 한국어로 옮겨 정리한 공부 노트로, 공개된 강의 23개와 PyTorch 튜토리얼을 다룹니다.
수학이 익숙하지 않은 사람도 따라올 수 있게, 강의에 나오는 수학을 **99개의 수학 노트**로 따로 풀고 **991개의 예제**를 붙였습니다.

## 시작하기

- **[목차](00_MIT딥러닝_목차.md)**: 강의 목록, 수학 노트 목록, 13단계 **수학 학습 순서**
- **[용어 사전](00_딥러닝_용어사전.md)**: 용어 503개. 제목은 `영어 (한글)` 형식입니다(예: `Loss function (손실 함수)`). 본문에 나오는 용어는 모두 이 사전으로 연결됩니다.

처음이라면 목차의 **수학 학습 순서**를 1단계부터 따라가면서, 각 단계가 쓰이는 강의를 함께 읽는 방법을 권합니다.

## 구성

| 폴더 | 내용 |
|---|---|
| (맨 위) | 강의 노트 24개(`Lec01` ~ `Lec24`, `Tut_PyTorch`), 목차, 용어 사전 |
| [수학개념](수학개념) | 수학 노트 99개. 선수 관계에 따라 13단계 폴더(`01_수와_함수` … `13_언어_모델과_추론`)로 나눴습니다 |
| [예제](예제) | 수학 노트마다 하나씩 있는 예제 노트(문제 10개 안팎). 폴더 구조는 `수학개념`과 같습니다 |
| [attachments](attachments) | 강의 슬라이드 캡처(`Lec##_cap_…`)와 노트의 숫자로 직접 그린 그림(`…_fig_…`) |
| [obsidian-snippets](obsidian-snippets) | Obsidian에서 노트를 읽기 좋게 해 주는 CSS 스니펫 2개 |

## 강의 목록

| 강의 |
|---|
| [Lec 01. 딥러닝 소개 (Introduction to Deep Learning)](Lec01_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%86%8C%EA%B0%9C.md) |
| [Lec 02. 신경망 학습 (How to Train a Neural Net)](Lec02_%EC%8B%A0%EA%B2%BD%EB%A7%9D_%ED%95%99%EC%8A%B5.md) |
| [Lec 03. 근사 이론 (Approximation Theory)](Lec03_%EA%B7%BC%EC%82%AC_%EC%9D%B4%EB%A1%A0.md) |
| [Lec 04. 격자를 위한 아키텍처 (Architectures: Grids)](Lec04_%EA%B2%A9%EC%9E%90_%EC%95%84%ED%82%A4%ED%85%8D%EC%B2%98.md) |
| [Lec 05. 아키텍처: 그래프 (Architectures: Graphs)](Lec05_%EA%B7%B8%EB%9E%98%ED%94%84_%EC%8B%A0%EA%B2%BD%EB%A7%9D.md) |
| [Lec 06. 일반화 이론 (Generalization Theory)](Lec06_%EC%9D%BC%EB%B0%98%ED%99%94_%EC%9D%B4%EB%A1%A0.md) |
| [Lec 07. 최적화의 스케일링 규칙 (Scaling Rules for Optimization)](Lec07_%EC%B5%9C%EC%A0%81%ED%99%94%EC%9D%98_%EC%8A%A4%EC%BC%80%EC%9D%BC%EB%A7%81_%EA%B7%9C%EC%B9%99.md) |
| [Lec 08. 아키텍처: 트랜스포머 (Architectures: Transformers)](Lec08_%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8.md) |
| [Lec 09. 해커의 딥러닝 가이드 (Hacker's Guide to Deep Learning)](Lec09_%ED%95%B4%EC%BB%A4%EC%9D%98_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EA%B0%80%EC%9D%B4%EB%93%9C.md) |
| [Lec 10. 아키텍처: 기억과 시퀀스 (Architectures: Memory)](Lec10_%EA%B8%B0%EC%96%B5%EA%B3%BC_%EC%8B%9C%ED%80%80%EC%8A%A4.md) |
| [Lec 11. 표현 학습: 재구성 기반 (Representation Learning: Reconstruction-Based)](Lec11_%ED%91%9C%ED%98%84_%ED%95%99%EC%8A%B5_%EC%9E%AC%EA%B5%AC%EC%84%B1.md) |
| [Lec 12. 유사도 기반 표현 학습 (Representation Learning: Similarity-Based)](Lec12_%EC%9C%A0%EC%82%AC%EB%8F%84_%EA%B8%B0%EB%B0%98_%ED%91%9C%ED%98%84_%ED%95%99%EC%8A%B5.md) |
| [Lec 13. 표현 학습: 이론 (Representation Learning: Theory)](Lec13_%ED%91%9C%ED%98%84_%ED%95%99%EC%8A%B5_%EC%9D%B4%EB%A1%A0.md) |
| [Lec 14. 생성 모델 기초 (Generative Models: Basics)](Lec14_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8_%EA%B8%B0%EC%B4%88.md) |
| [Lec 15. 변분 오토인코더: 표현 학습과 생성 모델의 만남 (Generative Models: Representation Learning Meets Generative Modeling)](Lec15_%EB%B3%80%EB%B6%84_%EC%98%A4%ED%86%A0%EC%9D%B8%EC%BD%94%EB%8D%94.md) |
| [Lec 16. 조건부 생성 모델 (Generative Models: Conditional Models)](Lec16_%EC%A1%B0%EA%B1%B4%EB%B6%80_%EC%83%9D%EC%84%B1_%EB%AA%A8%EB%8D%B8.md) |
| [Lec 17. 분포 밖 일반화 (Generalization: Out-of-Distribution)](Lec17_%EB%B6%84%ED%8F%AC_%EB%B0%96_%EC%9D%BC%EB%B0%98%ED%99%94.md) |
| [Lec 18. 전이 학습: 모델 (Transfer Learning: Models)](Lec18_%EC%A0%84%EC%9D%B4_%ED%95%99%EC%8A%B5_%EB%AA%A8%EB%8D%B8.md) |
| [Lec 19. 전이 학습: 데이터 (Transfer Learning: Data)](Lec19_%EC%A0%84%EC%9D%B4_%ED%95%99%EC%8A%B5_%EB%8D%B0%EC%9D%B4%ED%84%B0.md) |
| [Lec 20. 스케일링 법칙 (Scaling Laws)](Lec20_%EC%8A%A4%EC%BC%80%EC%9D%BC%EB%A7%81_%EB%B2%95%EC%B9%99.md) |
| [Lec 21. 언어 모델 (Language Models)](Lec21_%EC%96%B8%EC%96%B4_%EB%AA%A8%EB%8D%B8.md) |
| [Lec 23. 계량 딥러닝 (Metrized Deep Learning)](Lec23_%EA%B3%84%EB%9F%89_%EB%94%A5%EB%9F%AC%EB%8B%9D.md) |
| [Lec 24. 딥러닝의 추론 방법 (Inference Methods for Deep Learning)](Lec24_%EC%B6%94%EB%A1%A0_%EB%B0%A9%EB%B2%95.md) |
| [PyTorch 튜토리얼 (PyTorch Tutorial)](Tut_PyTorch_%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC.md) |

22강은 OCW에 영상이 공개되지 않아 없습니다. 20강은 공개된 영상이 38분에서 끊겨 있어서 녹화된 부분까지만 정리했습니다.

## 노트 형식

**강의 노트**에는 다음이 들어 있습니다.

- 절마다 영상 시간 링크(`▶ 영상 [mm:ss]`)와, 그 절에 쓰인 수학 노트로 가는 `📐 수학 개념` 줄
- 강사가 직접 만든 슬라이드의 캡처와, 노트의 숫자로 새로 그린 그림
- 전문용어는 `한글(영어)`로 함께 표기
- 강의 이후 바뀐 사실과 예제는 `🆕 2026 업데이트` 칸에 따로 적었습니다. 강의가 한 말, 지금 상황, 출처 링크가 함께 있습니다.
- 강의 중 말실수나 틀린 숫자는 원문을 지우지 않고 괄호로 바로잡았습니다.

**수학 노트**는 모두 같은 순서로 씁니다: 3줄 요약 → 개념 설명(비유 먼저) → 필요한 이유 → 숫자로 보기 → 코드로 확인 → 주의 → 개념 재확인 → 예제 풀기.

- 수식마다 접히는 칸 세 개가 붙어 있습니다. **기호 풀이**(기호, 읽는 법, 뜻, 예제 값), **예제와 풀이**, 직접 채우는 **내 이해** 칸입니다.
- 풀이는 모두 **대입해서 풀기** 방식입니다. 식을 그대로 쓰고 값을 한꺼번에 대입한 줄을 쓴 뒤, 계산을 한 단계에 하나씩 말로 설명합니다.
- 코드는 줄마다 위에 `[함수]`, `[변수]`, `[쓸모]`, `[연결]` 주석을 달았습니다.
- 노트 맨 위의 `📚 학습 순서 n/99` 줄로 앞뒤 노트를 오갈 수 있습니다.

**예제 노트**의 문제는 기본(직접 대입) → 응용 → 강의와 연결 순서이고, 풀이는 접혀 있습니다.

## Obsidian에서 보기 좋게 (CSS 스니펫)

원본은 [Obsidian](https://obsidian.md) 노트입니다. 이 저장소를 받아 폴더째 Obsidian 금고로 열고(**Open folder as vault**) 아래 스니펫을 켜면 더 읽기 편합니다.

| 스니펫 | 하는 일 |
|---|---|
| `glossary-links.css` | 용어 사전으로 가는 링크를 본문 글자색 그대로 두고 옅은 점선 밑줄만 표시합니다. 용어마다 링크가 걸려 있어도 글이 알록달록해지지 않습니다 |
| `dl-readability.css` | 한글에 맞춘 줄 간격, `##` 제목 아래 구분선, 표 머리줄 배경과 줄무늬, 본문 속 코드 배경, 접힌 칸의 회색 처리, 긴 수식 가로 스크롤 |

1. Obsidian **설정 → 외관(Appearance) → CSS 스니펫(CSS snippets)** 오른쪽의 폴더 아이콘을 눌러 금고의 `.obsidian/snippets` 폴더를 엽니다.
2. [obsidian-snippets](obsidian-snippets) 안의 CSS 파일 2개를 그 폴더에 복사합니다.
3. 같은 화면에서 새로고침 아이콘을 누르고 두 스니펫을 켭니다.

- `dl-readability.css`는 속성에 `cssclasses: [dl-note]`가 있는 노트에만 적용됩니다. 이 저장소의 노트에는 이미 들어 있습니다.
- 용어 링크에 마우스를 올리면 사전 설명이 뜹니다. **코어 플러그인 → 페이지 미리보기(Page preview)** 가 켜져 있어야 하고 편집 보기에서는 `Ctrl`을 누른 채 올립니다.
- `glossary-links.css`의 효과는 **읽기 보기**(`Ctrl+E`)에서만 보입니다.

## GitHub에서 읽을 때

GitHub에서 바로 읽을 수 있도록 Obsidian 문법을 일반 마크다운으로 바꿔 올렸습니다.

- `[[위키링크]]` → 일반 링크, 그림 임베드 → 일반 이미지
- 접히는 칸(기호 풀이, 예제와 풀이, 개념 재확인 등) → `<details>` 접기
- 그 밖의 칸 → GitHub 알림 상자(`NOTE`, `TIP`, `WARNING`)

수식(`$...$`)은 GitHub에서 그대로 보입니다.

## 검증

- 노트에 나오는 숫자는 손계산 표, 예제 답, 개념 재확인 풀이까지 모두 파이썬으로 다시 계산해 확인했습니다. 노트 속 코드 블록은 실제로 실행했고, `실행 결과`는 그 출력을 그대로 붙였습니다.
- `🆕 2026 업데이트`는 2026년 10월 기준으로 논문, 공식 문서, 공식 블로그를 찾아 확인한 내용만 적었습니다.
- 개인 공부 노트라 틀린 곳이 있을 수 있습니다. 이상한 곳을 발견하면 이슈로 알려 주세요.

## 출처와 라이선스

- **원 강의**: MIT OpenCourseWare, *6.7960 Deep Learning, Fall 2024*. <https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/>. 원 강의는 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 라이선스입니다.
- **이 저장소**: 원 강의를 바탕으로 한국어로 옮기고 다시 구성한 2차 저작물입니다. 원 강의의 조건에 따라 같은 **CC BY-NC-SA 4.0**(출처 표기, 비영리, 동일 조건 변경 허락)으로 공개합니다. 자세한 내용은 [LICENSE.md](LICENSE.md)를 보세요.
- 슬라이드 캡처는 강의 영상에서 가져왔고 외부 사진이나 논문 그림이 중심인 슬라이드는 넣지 않았습니다. 그 밖의 그림은 노트의 숫자로 직접 그렸습니다.
- 이 저장소는 MIT나 강의자와 관계없는 개인 학습 자료입니다.
