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

This file is a **format example**. To write a real summary, copy this file and edit it.

## TL;DR

Proposes the Transformer, which models sequences using attention alone, with no recurrence or convolution.

## Key idea

Scaled dot-product attention is computed as follows. Wrap math in `$$ ... $$` and it renders automatically.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Inline math works too: dividing by $\sqrt{d_k}$ keeps the dot products from growing too large.

## Notes

> Block quotes look like this.

- Bullet points
- Code blocks, tables, and images all use standard Markdown

```python
# code block example
import torch
```
