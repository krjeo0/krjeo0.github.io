# Attention Is All You Need 정밀 독해 — Transformer 함께 읽기

> Vaswani et al., *Attention Is All You Need*, **NeurIPS 2017** (arXiv:1706.03762) 해설 자료입니다.

---

## 이 자료에 대하여

이 글은 원조 **Transformer** 논문을 **나란히 펴 놓고 읽기 위한 해설**입니다. 번역본이 아니라, 이 논문이 처음 정립한 **핵심 메커니즘**(Query/Key/Value, scaled dot-product attention, multi-head attention, positional encoding 등)과 그 직관, 그리고 이후 거의 모든 딥러닝 모델로 퍼져나간 계보를 짚는 것이 목적입니다.

- **이 시리즈의 뿌리.** 앞서 다룬 **ViT**·**Swin** 자료에서 당연하게 쓰던 어휘 — Q/K/V, multi-head, positional encoding, Pre/Post-LN — 이 **전부 이 논문에서 출발**합니다. 그 원점을 보는 것이 이 자료의 가장 큰 가치입니다.
- **"notation"의 의미.** 여기서 짚는 notation은 n, d 같은 자명한 기호가 아니라, **새로운 아이디어가 담긴 개념적 장치**(Q, K, V 등)를 말합니다. 자명한 기호는 본문에서만 풀고, 부록 A에 핵심 메커니즘을 따로 정리했습니다.
- **표기 매핑.** 수식·도판은 논문 원본 번호(예: 논문 Eq. (1), 논문 Figure 2, 논문 Table 1, 논문 §3.2)와 연결해 두었습니다.
- **도판 출처.** 아키텍처·attention 그림은 논문 원본 도판이고, 계보도와 수식 박스는 이 해설을 위해 새로 제작했습니다. (원 논문은 학술적 용도의 도판 재사용을 명시적으로 허용합니다.)

---

## 0. 한 문장 요약

> **RNN/CNN의 "순차성"과 "거리에 따라 커지는 연산"이라는 한계를 버리고, 오직 attention만으로 시퀀스의 전역 의존성을 한 번에 잇는 인코더-디코더. 병렬화가 쉽고 long-range를 곧바로 연결하며, 이후 BERT·GPT·ViT·Swin 등 거의 모든 현대 모델의 공통 조상이 된다.**

제목 *"Attention Is All You Need"* 그대로 — recurrence도 convolution도 필요 없고, **attention이면 충분하다**는 주장입니다.

---

## 1. 계보적 위치 — 모든 것의 출발점

<figure style="max-width:780px; margin:1.6em auto; text-align:center;">
  <img src="images/fig0_genealogy.png" alt="Transformer의 계보적 위치 다이어그램" style="width:100%;">
  <figcaption style="font-size:0.9em; color:#555; margin-top:0.5em;"><b>Transformer의 계보적 위치.</b> 순차성·거리 한계를 제거하고, 이후 NLP·비전 거의 모든 모델의 공통 조상이 됩니다. ★ 표시한 ViT·Swin은 이 시리즈에서 앞서 다룬 후손들입니다.</figcaption>
</figure>

**선행 연구.** 2017년 당시 시퀀스 모델링의 주류는 **RNN/LSTM/GRU 기반 인코더-디코더**였습니다. 강력하지만 **본질적으로 순차적**이라(위치 t의 은닉상태가 t−1에 의존) **병렬화가 안 되고**, 긴 시퀀스에서 느렸습니다. **Bahdanau attention (2015)** 이 RNN에 attention을 더해 정렬(alignment)을 학습하게 했지만, 어디까지나 **RNN과 함께** 쓰였습니다. **ConvS2S·ByteNet (2017)** 은 CNN으로 병렬화를 시도했으나, 두 위치를 잇는 연산량이 **거리에 따라(선형/로그) 증가**해 먼 의존성 학습이 어려웠습니다. 한편 **self-attention(intra-attention)** 은 단일 시퀀스 내부 관계를 모델링하는 도구로 일부 과제에서 쓰이고 있었습니다.

**Transformer가 던진 질문.** "recurrence와 convolution을 **완전히 빼고**, 오직 attention만으로 시퀀스를 모델링할 수 있는가?"

**이후로 이어진 것 (폭발).** 이 논문 이후 흐름이 완전히 바뀝니다. §5에서 볼 **세 가지 attention 사용처**가 그대로 세 갈래가 되어 — **BERT**(encoder-only, 마스킹 사전학습), **GPT 라인**(decoder-only, 언어모델), **T5**(encoder-decoder) — NLP를 평정합니다. 그리고 패치를 토큰으로 바꿔 비전으로 건너간 것이 바로 이 시리즈의 앞 자료인 **ViT**, 그 위에 계층·윈도우를 더한 **Swin**입니다.

> **시리즈 연결.** ViT 자료의 *[class] 토큰·Pre-LN·1D 위치 임베딩*, Swin 자료의 *relative position bias*, 둘 다의 *Q/K/V·multi-head* — 이 글을 읽고 나면 그 모든 용어가 **여기서 어떻게 정의되었는지**가 한눈에 들어옵니다.

---

## 2. 핵심 메커니즘 ① — Attention의 재정의: Query, Key, Value

이 논문이 남긴 가장 근본적인 추상화입니다. **attention 함수는 하나의 query와 (key, value) 쌍들의 집합을 output으로 매핑**합니다. output은 **value들의 가중합**이고, 각 value의 가중치는 **query와 그 key의 호환성(compatibility)** 으로 정해집니다.

**직관 — 검색(retrieval) 비유.** 도서관에서 책을 찾는다고 생각하면 쉽습니다.

- **Query(Q)**: 내가 "찾고 싶은 것"(검색어).
- **Key(K)**: 각 항목에 붙은 "색인/표제"(검색어와 비교되는 대상).
- **Value(V)**: 그 항목의 "실제 내용".

query를 **모든 key와 비교**해 유사도(가중치)를 구하고, 그 가중치로 **value들을 섞어** 답을 만듭니다. 유사한 key를 가진 value일수록 더 많이 반영됩니다.

이 **Q/K/V 추상화**가 이후 모든 attention의 공용어가 됩니다 — ViT·Swin 자료에서 등장한 Q, K, V가 정확히 이 정의입니다. 핵심은, **무엇을 Q/K/V로 삼느냐에 따라 self-attention도 되고 cross-attention도 된다**는 유연함입니다(§5).

---

## 3. 핵심 메커니즘 ② — Scaled Dot-Product Attention (논문 Figure 2 왼쪽, Eq. 1)

<figure style="max-width:760px; margin:1.5em auto; text-align:center;">
  <img src="images/fig2_attention.png" alt="Scaled Dot-Product Attention과 Multi-Head Attention 도식" style="width:100%;">
  <figcaption style="font-size:0.9em; color:#555; margin-top:0.5em;"><b>논문 Figure 2.</b> (왼쪽) Scaled Dot-Product Attention: Q·K를 MatMul → √d_k로 Scale → (선택적) Mask → SoftMax → V와 MatMul. (오른쪽) Multi-Head Attention: Q/K/V를 h번 서로 다른 Linear로 사영해 병렬 attention 후 Concat → Linear.</figcaption>
</figure>

Q를 행렬로 묶고(K, V도 마찬가지), 한 번에 계산합니다.

<div style="text-align:center; margin:1.1em 0;"><img src="images/eq_attention.png" alt="scaled dot-product attention 수식" style="max-width:620px; width:92%;"></div>

Q와 K의 내적으로 유사도를 구하고, **√d_k로 나눈 뒤** softmax로 가중치를 만들어 V에 곱합니다. 내적(dot-product) 방식은 덧셈(additive) attention과 이론적 복잡도는 비슷하지만, **고도로 최적화된 행렬곱으로 구현되어 훨씬 빠르고 메모리 효율적**입니다.

**왜 √d_k로 나누는가 (중요한 직관).** 이 작은 스케일링이 핵심 디테일입니다. q, k의 각 성분이 평균 0, 분산 1인 독립 변수라면, 둘의 내적은 다음 성질을 가집니다.

<div style="text-align:center; margin:1.0em 0;"><img src="images/eq_variance.png" alt="q와 k 내적의 평균과 분산" style="max-width:560px; width:90%;"></div>

즉 **차원 d_k가 커질수록 내적의 분산도 커져** 값이 커집니다. 내적이 너무 크면 softmax가 **포화 영역(한 값이 거의 1, 나머지는 0)** 으로 밀려 **gradient가 극도로 작아집니다.** 이를 막으려고 √d_k로 나눠 분산을 1 수준으로 되돌리는 것입니다. — 이후 거의 모든 attention 구현이 이 스케일링을 표준으로 씁니다(ViT·Swin 자료의 attention 식에 등장한 1/√d 가 바로 이것입니다).

---

## 4. 핵심 메커니즘 ③ — Multi-Head Attention (논문 Figure 2 오른쪽)

단일 attention 대신, Q/K/V를 **h번 서로 다른 학습된 선형사영**으로 더 낮은 차원(d_k, d_k, d_v)에 투영한 뒤, 각각에 attention을 **병렬로** 수행하고, 결과를 concat해 다시 사영합니다.

<div style="text-align:center; margin:1.1em 0;"><img src="images/eq_multihead.png" alt="multi-head attention 수식" style="max-width:680px; width:94%;"></div>

**왜 여러 헤드인가 (직관).** multi-head는 **"여러 표현 부분공간(subspace)에서 동시에 다른 관계를 attend"** 하기 위함입니다. 헤드가 하나뿐이면 여러 관계를 한 번에 평균내느라 정보가 뭉개집니다. 헤드를 나누면 어떤 헤드는 **장거리 의존성**을, 어떤 헤드는 **구문 구조**를, 또 어떤 헤드는 **대용어(anaphora) 해소**를 담당하는 식으로 역할이 갈립니다(§9, 논문 Figure 3~5).

논문은 **h=8**개 헤드, 각 헤드 **d_k=d_v=d_model/h=64**(d_model=512)를 씁니다. 헤드마다 차원을 1/h로 줄였기에 **총 연산량은 단일 헤드(전체 차원)와 비슷**합니다. 이 "쪼개도 비용은 그대로"가 multi-head를 공짜에 가깝게 만드는 포인트입니다.

---

## 5. 핵심 메커니즘 ④ — 세 가지 attention 사용처와 Masking

Q/K/V 추상화의 위력은 **무엇을 Q/K/V로 삼느냐**에 따라 같은 메커니즘이 다른 일을 한다는 데 있습니다. Transformer는 attention을 **세 가지 방식**으로 씁니다.

1. **인코더 self-attention**: Q, K, V가 **모두 이전 인코더 층의 출력**에서 나옵니다. 각 위치가 입력의 모든 위치에 attend합니다.
2. **디코더 self-attention (masked)**: 마찬가지로 Q, K, V가 디코더 내부에서 나오지만, **미래 위치를 보지 못하도록 마스킹**합니다. 위치 i의 예측은 i보다 앞선 위치들에만 의존해야 하기 때문입니다(auto-regressive 보존).
3. **인코더-디코더 (cross) attention**: **Q는 디코더**에서, **K·V는 인코더 출력**에서 나옵니다. 디코더의 각 위치가 입력 시퀀스 전체를 참조 — 전형적인 seq2seq attention입니다.

**Masking의 구현 (중요).** 디코더가 미래를 못 보게 하는 방법은 단순합니다 — softmax에 들어가기 직전, **금지된(미래) 연결의 점수를 −∞로 설정**합니다. 그러면 softmax 후 그 가중치가 0이 되어 정보가 흐르지 않습니다. 이 "−∞ 마스킹"이 **auto-regressive 생성**을 가능하게 합니다.

> **시리즈 연결.** 이 세 가지 사용처가 이후 세 갈래 아키텍처로 굳어집니다 — 인코더 self-attention만 쓰는 **BERT**(encoder-only), masked self-attention 중심의 **GPT**(decoder-only), 셋 다 쓰는 **T5**(encoder-decoder). 또한 Swin 자료의 *윈도우 내부 attention*도 결국 "어느 범위를 K/V로 삼을 것인가"의 변주이고, 디코더 masking의 발상은 이미지·영상 생성 모델로도 퍼집니다.

---

## 6. 핵심 메커니즘 ⑤ — Position-wise Feed-Forward Network (논문 Eq. 2)

attention 서브층 외에, 각 인코더·디코더 층에는 **위치별 FFN**이 있습니다. 각 위치(토큰)에 **독립적이고 동일하게** 적용되는 2층 MLP입니다.

<div style="text-align:center; margin:1.1em 0;"><img src="images/eq_ffn.png" alt="position-wise feed-forward network 수식" style="max-width:560px; width:90%;"></div>

ReLU를 사이에 둔 두 선형변환으로, 입출력은 d_model=512, 내부 차원은 d_ff=2048입니다. 흥미롭게도 이는 **커널 크기 1인 합성곱 두 번**과 같습니다.

**역할 분담의 직관.** Transformer 한 층을 두 일로 나눠 보면 깔끔합니다 — **attention은 토큰들 사이에 정보를 섞고(mixing across positions)**, **FFN은 각 토큰을 독립적으로 비선형 변환**합니다. "정보 교환"과 "개별 가공"이 번갈아 일어나는 구조입니다. (ViT·Swin의 블록도 정확히 이 *attention + MLP* 2단 구조를 따릅니다.)

---

## 7. 핵심 메커니즘 ⑥ — Positional Encoding (sinusoidal)

attention에는 치명적인(그러나 의도된) 성질이 있습니다 — **순서를 모릅니다.** self-attention은 토큰 순서를 바꿔도 결과가 따라 바뀔 뿐(permutation equivariant), 위치 개념이 내장돼 있지 않습니다. recurrence·convolution이 없으니 **순서 정보를 따로 주입**해야 합니다. 이를 위해 입력 임베딩에 **positional encoding**을 더합니다(같은 차원 d_model이라 그냥 합산).

논문은 서로 다른 주파수의 사인·코사인 함수를 씁니다.

<div style="text-align:center; margin:1.1em 0;"><img src="images/eq_posenc.png" alt="sinusoidal positional encoding 수식" style="max-width:620px; width:92%;"></div>

각 차원이 하나의 sinusoid에 대응하고, 파장은 2π부터 10000·2π까지 기하급수적으로 늘어납니다.

**왜 sinusoid인가 (직관).** 핵심 동기는 **상대 위치**입니다. 고정 오프셋 k에 대해 PE_{pos+k}가 PE_pos의 **선형 함수**로 표현되기 때문에, 모델이 "k칸 떨어진 위치"라는 관계를 쉽게 학습할 수 있습니다. 논문은 학습형(learned) 위치 임베딩도 실험했는데 **성능이 거의 같았고**(논문 Table 3 행 (E)), sinusoid를 택한 이유는 **훈련 때보다 긴 시퀀스로 외삽(extrapolate)** 할 여지가 있어서입니다.

> **시리즈 연결 — 위치 인코딩 계보.** ViT 자료의 *1D 학습형 위치 임베딩*, Swin 자료의 *relative position bias*, 그리고 최신 모델의 *RoPE* — 모두 "위치를 어떻게 주입할 것인가"라는, **바로 이 절에서 시작된 질문**의 후예들입니다.

---

## 8. 아키텍처 한 바퀴 (논문 Figure 1)

<figure style="max-width:520px; margin:1.5em auto; text-align:center;">
  <img src="images/fig1_architecture.png" alt="Transformer 인코더-디코더 전체 아키텍처" style="width:100%;">
  <figcaption style="font-size:0.9em; color:#555; margin-top:0.5em;"><b>논문 Figure 1.</b> 왼쪽이 인코더, 오른쪽이 디코더입니다. 각각 N개의 동일 층을 쌓습니다. 디코더는 masked self-attention과 인코더-디코더 attention을 모두 가집니다. 입력에는 positional encoding이 더해집니다.</figcaption>
</figure>

전체는 표준적인 **인코더-디코더** 구조입니다.

- **인코더**: N=6개의 동일한 층. 각 층은 두 서브층 — **multi-head self-attention** + **position-wise FFN**.
- **디코더**: N=6개의 동일한 층. 인코더의 두 서브층에 더해 **세 번째 서브층(인코더-디코더 attention)** 을 끼워 넣습니다. 그리고 self-attention은 **masked**(미래 차단)입니다.
- 각 서브층은 **잔차 연결 + LayerNorm**으로 감쌉니다.

**Post-LN 구조에 주목 (시리즈 연결).** 원조 Transformer는 서브층을 다음과 같이 감쌉니다.

<div style="text-align:center; margin:1.0em 0;"><img src="images/eq_sublayer.png" alt="Post-LN: LayerNorm(x + Sublayer(x))" style="max-width:420px; width:78%;"></div>

즉 **서브층을 통과시키고 잔차를 더한 뒤 마지막에 LayerNorm**을 적용합니다(Post-LN). 그런데 ViT·Swin을 포함한 **이후 모델 대부분은 LN을 서브층 앞에 두는 Pre-LN으로 바뀝니다** — Pre-LN이 깊은 Transformer의 학습 안정성을 높이기 때문입니다. ViT 자료에서 "Pre-LN에 주목하라"고 한 대목이, 바로 이 원조 Post-LN과의 차이를 가리킵니다.

부가 사항: 모든 서브층·임베딩의 출력 차원은 d_model=512로 일정하고, 두 임베딩 층과 pre-softmax 선형층은 **가중치를 공유**하며, 임베딩에는 √d_model을 곱합니다.

---

## 9. 왜 self-attention인가 (논문 Table 1) — 설계의 정당화

논문은 self-attention을 RNN·CNN과 **세 가지 기준**으로 비교합니다. 이 표가 Transformer의 존재 이유를 압축합니다.

1. **층당 연산 복잡도** (계산량)
2. **병렬화 가능량** = 필요한 **순차 연산(sequential operation)** 의 최소 개수 (적을수록 병렬화 유리)
3. **최대 경로 길이(maximum path length)**: 두 위치 사이 신호가 오가야 하는 거리 (짧을수록 long-range 의존성 학습이 쉬움)

| 층 종류 | 층당 복잡도 | 순차 연산 | 최대 경로 길이 |
|---|---|---|---|
| **Self-Attention** | O(n²·d) | **O(1)** | **O(1)** |
| Recurrent (RNN) | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(log_k n) |
| Self-Attention (restricted) | O(r·n·d) | O(1) | O(n/r) |

(n=시퀀스 길이, d=표현 차원, k=conv 커널 크기, r=제한 attention의 이웃 크기)

**핵심 메시지.** self-attention은 **모든 위치 쌍을 상수(O(1)) 경로로 연결**합니다 — RNN은 O(n) 경로라 먼 위치일수록 학습이 어렵습니다. 게다가 self-attention의 순차 연산은 O(1)이라 **완전히 병렬화**됩니다(RNN은 O(n) 순차 단계가 강제됨). 복잡도 면에서도 **n < d일 때 RNN보다 빠른데**, 이는 기계번역에서 흔한 상황입니다.

**대가와 보완.** self-attention은 가중 평균이라 **effective resolution이 떨어지는** 단점이 있는데(여러 위치를 평균내며 세밀함을 잃음), 이를 **multi-head**(§4)로 보완합니다. 또한 매우 긴 시퀀스에는 **이웃 크기 r로 제한한 attention**(표의 마지막 행)을 쓸 수 있다고 언급하는데 — 이 *restricted/local attention* 아이디어가 이후 **Sparse Transformer, 그리고 Swin의 윈도우 기반 attention**으로 이어집니다(Swin 자료와 직접 연결되는 대목입니다).

**부수 효과 — 해석 가능성.** attention 가중치를 들여다보면 모델이 무엇을 보는지 읽을 수 있습니다.

<figure style="max-width:720px; margin:1.5em auto; text-align:center;">
  <img src="images/fig3_attention_viz.png" alt="장거리 의존성을 따라가는 attention 시각화" style="width:100%;">
  <figcaption style="font-size:0.9em; color:#555; margin-top:0.5em;"><b>논문 Figure 3.</b> 인코더 self-attention(6층 중 5층)에서 'making'이라는 동사가 멀리 떨어진 'more difficult'에 attend하여 'making...more difficult' 구를 완성합니다. 색깔이 다른 선은 서로 다른 헤드를 나타냅니다.</figcaption>
</figure>

논문 Figure 3~5는 **개별 헤드가 서로 다른 언어적 역할**(장거리 의존성, 대용어 해소, 구문 구조 등)을 학습함을 보여줍니다. multi-head가 "여러 부분공간에서 다른 관계를 본다"는 §4의 주장을 뒷받침하는 정성적 증거입니다 — ViT 자료의 *attention distance* 분석이 이 해석성 전통을 비전으로 이은 셈입니다.

---

## 10. 실험 하이라이트

- **기계번역 (논문 Table 2).** WMT 2014 영→독에서 big 모델이 **28.4 BLEU** 로 기존 최고(앙상블 포함)를 **+2.0 BLEU** 경신, 영→불에서 **41.8 BLEU**. base 모델조차 이전 모든 모델을 능가하면서 **훈련 비용은 극히 일부**(8 P100 GPU로 12시간; big은 3.5일)였습니다.
- **구문 분석 일반화 (논문 Table 4).** 영어 constituency parsing에도 큰 튜닝 없이 잘 일반화(WSJ만으로도 강력) → 번역 전용이 아니라 **범용 시퀀스 모델**임을 시사.
- **Ablation (논문 Table 3).** 헤드 수는 너무 적어도(단일 헤드 −0.9 BLEU) 너무 많아도 나빠집니다. d_k를 줄이면 품질 하락 — 호환성 함수가 단순 내적 이상이어야 할 수 있음을 시사. 큰 모델·dropout이 도움이 되고, **학습형 위치 임베딩은 sinusoid와 거의 동일**합니다.

---

## 11. 자주 헷갈리는 포인트 / 핵심 takeaway

- **Q/K/V는 출처가 자유롭다.** 같은 입력에서 나오면 self-attention, 다른 데서 나오면 cross-attention. 이 유연함이 핵심.
- **√d_k 스케일링**은 softmax 포화(gradient 소실)를 막기 위한 것.
- **multi-head**는 "여러 부분공간에서 다른 관계를 동시에 attend"이지, 단순히 큰 attention이 아니다. 총 연산은 단일 헤드와 비슷.
- **−∞ masking**이 디코더의 auto-regressive 성질을 만든다.
- **attention은 순서를 모른다**(permutation equivariant) → positional encoding으로 순서를 주입.
- **attention = 토큰 간 정보 혼합, FFN = 토큰별 비선형 변환.** 한 층이 이 둘을 번갈아 수행.
- **Post-LN(원조) vs Pre-LN(이후 표준).** ViT·Swin은 Pre-LN. 작아 보여도 깊은 모델 안정성에 중요.

---

## 12. 이 논문이 남긴 것 — 거의 모든 것의 조상

<figure style="max-width:780px; margin:1.4em auto; text-align:center;">
  <img src="images/fig0_genealogy.png" alt="Transformer 계보도 (재게시)" style="width:100%;">
</figure>

- **NLP 세 갈래.** §5의 세 attention 사용처가 그대로 — **BERT**(encoder-only, MLM), **GPT 라인**(decoder-only, LM), **T5**(encoder-decoder, text-to-text)로 분기해 현대 NLP·LLM의 토대가 됩니다.
- **비전으로 이식.** 패치를 토큰으로 본 **ViT**, 그 위에 계층·윈도우를 더한 **Swin** — 이 시리즈의 앞 두 자료가 바로 이 논문의 직계 후손입니다.
- **효율 attention 계보.** §9에서 언급된 *restricted/local attention*이 **Sparse Transformer, Performer**, 그리고 **Swin의 windowed attention** 등 "전역 attention을 어떻게 싸게 만들 것인가"라는 거대한 연구 줄기로 이어집니다.
- **공용어의 정립.** 무엇보다 이 논문은 **Q/K/V, multi-head attention, positional encoding, scaled dot-product**라는, 이후 분야를 막론하고 통용되는 **공통 어휘**를 만들었습니다.

> **한 줄 정리.** *"Attention Is All You Need"* 는 단일 번역 모델을 넘어, 이후 거의 모든 딥러닝 시퀀스·비전 모델의 **공통 조상이자 공용어**를 정립한 논문입니다. 이 시리즈를 거꾸로 읽으면(Transformer → ViT → Swin) "attention이라는 한 아이디어가 어떻게 NLP를 평정하고, 비전으로 건너가, 다시 CNN의 inductive bias와 만나는가"라는 한 편의 계보가 완성됩니다.

---

## 부록 A. 핵심 개념(notation) 정리 — 새 메커니즘 중심

> 자명한 기호(n, d 등)가 아니라, 이 논문이 **처음 정립한 개념적 장치**만 모았습니다.

| 개념 (notation) | 의미와 역할 |
|---|---|
| **Query / Key / Value (Q, K, V)** | attention의 근본 추상화. Q와 K의 호환성으로 가중치를 만들어 V를 가중합. 검색(query)–색인(key)–내용(value) 비유. 이후 모든 attention의 공용어. |
| **Scaled Dot-Product Attention** | softmax(QKᵀ/√d_k)V. 내적으로 유사도 → √d_k로 스케일 → softmax → V 가중합. |
| **√d_k 스케일링** | 내적 분산이 d_k에 비례해 커지면 softmax가 포화(gradient↓). 분산을 1로 되돌려 학습 안정화. |
| **Multi-Head Attention** | Q/K/V를 h개로 쪼개 서로 다른 부분공간에서 병렬 attention 후 Concat·사영. 여러 관계를 동시에 포착(평균으로 뭉개지는 것 방지). |
| **Self vs Cross attention** | Q/K/V가 같은 곳에서 나오면 self, 디코더 Q + 인코더 K/V면 cross. 같은 메커니즘의 두 용법. |
| **Masked self-attention** | 미래 위치 점수를 −∞로 만들어 차단. auto-regressive 생성을 보장(디코더). |
| **Positional Encoding** | 순서 없는 attention에 위치 정보를 주입. sinusoidal(상대 위치=선형, 외삽 가능) 또는 학습형. |
| **Position-wise FFN** | 각 토큰에 독립·동일하게 적용되는 2층 MLP(=커널1 conv 2회). 토큰별 비선형 변환. |
| **Post-LN 잔차** | LayerNorm(x + Sublayer(x)). 잔차 더한 뒤 LN. (이후 모델은 Pre-LN으로 전환.) |

## 부록 B. 핵심 하이퍼파라미터 (논문 Table 3)

| 설정 | N | d_model | d_ff | h | d_k = d_v | P_drop |
|---|---|---|---|---|---|---|
| base | 6 | 512 | 2048 | 8 | 64 | 0.1 |
| big | 6 | 1024 | 4096 | 16 | 64 | 0.3 |

옵티마이저는 Adam(β₁=0.9, β₂=0.98)에 **warmup 후 inverse-sqrt 감쇠** 스케줄(warmup 4000 step), 정규화로 residual dropout과 label smoothing(0.1)을 사용합니다.

---

<p style="font-size:0.85em; color:#888; text-align:center; margin-top:2em;">본 자료는 "Attention Is All You Need"(Transformer) 논문을 함께 읽기 위한 해설입니다. 삽입된 아키텍처·attention 그림은 논문 원본 도판이며, 계보도와 수식 박스는 본 해설을 위해 새로 제작했습니다.</p>
