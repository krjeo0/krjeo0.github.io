# Paper Log — a paper-reading catalog

Built on GitHub Pages + Jekyll. You never need to touch the HTML — just drop a
markdown file into `_posts` and it's published.

## 1. First-time deploy (once)

1. Create a new GitHub repository named exactly `yourusername.github.io`
2. Push everything in this folder to it (or drag the files in via the web UI)
3. Repo **Settings → Pages** → Source `Deploy from a branch`, Branch `main`
4. After a minute or two, open `https://yourusername.github.io`

## 2. Things to set first

- `_config.yml` — set `nickname` to your name
- Upload your photo to `assets/img/` and update `profile_image` in `_config.yml`
- `about.html` — write your bio

## 3. Adding a paper summary (the routine)

Add a file to `_posts/` and push. That's it.

- Filename: `YYYY-MM-DD-paper-title.md` (date = the day you read it)
- Front matter at the top of the file:

```yaml
---
layout: post
title: "Paper title"
date: 2026-06-12
authors: "Vaswani et al."   # optional
venue: "NeurIPS"            # optional
year: 2017                  # optional
tags: [NLP, Transformer]    # optional
link: https://arxiv.org/... # optional, link to the original
---
```

Write the summary below in plain Markdown. Math uses `$...$` / `$$...$$`.

The fields above feed the table on the home page: `venue` and `tags` become
filter options, and every column is sortable.

Sample: `_posts/2026-06-12-attention-is-all-you-need.md`
