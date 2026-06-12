# Paper Log — 논문 정리 블로그

GitHub Pages + Jekyll 기반. HTML은 건드릴 필요 없이 `_posts`에 md 파일만 올리면 글이 발행됩니다.

## 1. 최초 배포 (한 번만)

1. GitHub에서 새 저장소 생성 — 이름은 반드시 `본인아이디.github.io`
2. 이 폴더의 내용 전체를 저장소에 push
3. 저장소 **Settings → Pages** → Source를 `Deploy from a branch`, Branch를 `main`으로 설정
4. 1~2분 뒤 `https://본인아이디.github.io` 접속

## 2. 처음에 바꿀 것

- `_config.yml` — `nickname`을 본인 이름으로
- `assets/img/`에 본인 사진 업로드 후 `_config.yml`의 `profile_image` 경로 수정
- `about.html` — 자기소개 작성

## 3. 논문 정리본 올리기 (평소에 할 일)

`_posts/` 폴더에 아래 형식의 파일을 추가하고 push하면 끝.

- 파일명: `YYYY-MM-DD-논문제목.md` (날짜 = 읽은 날)
- 파일 맨 위 머리말:

```yaml
---
layout: post
title: "논문 제목"
date: 2026-06-12
authors: "Vaswani et al."   # 선택
venue: "NeurIPS"            # 선택
year: 2017                  # 선택
tags: [NLP, Transformer]    # 선택
link: https://arxiv.org/... # 선택, 원문 링크
---
```

머리말 아래에 일반 마크다운으로 정리본 작성. 수식은 `$...$` / `$$...$$`.

샘플: `_posts/2026-06-12-attention-is-all-you-need.md`
