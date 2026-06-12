---
layout: post
title: "Attention Is All You Need"
date: 2026-06-12
authors: "Vaswani et al."
venue: "NeurIPS"
year: 2017
tags: [NLP, Transformer]
link: https://arxiv.org/abs/1706.03762
---

이 파일은 **작성 형식 예시**입니다. 실제 정리본을 쓸 때 이 파일을 복사해서 시작하세요.

## 한 줄 요약

RNN/CNN 없이 attention만으로 시퀀스를 모델링하는 Transformer 구조를 제안.

## 핵심 아이디어

Scaled dot-product attention은 다음과 같이 계산됩니다. 수식은 `$$ ... $$`로 감싸면 자동 렌더링됩니다.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

인라인 수식도 가능합니다: 차원 $d_k$로 나누는 이유는 내적 값이 커지는 것을 막기 위함.

## 메모

> 인용구는 이렇게 표시됩니다.

- 글머리표
- 코드 블록, 표, 이미지 모두 일반 마크다운 문법 그대로 사용

```python
# 코드 블록 예시
import torch
```
