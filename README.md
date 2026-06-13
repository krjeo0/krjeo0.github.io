# Paper Log — a paper-reading catalog + your own papers

Built on GitHub Pages + Jekyll. You never need to touch the HTML — just drop
markdown files into the right folder and they're published.

## 1. First-time deploy (once)

1. Create a new GitHub repository named exactly `yourusername.github.io`
2. Push everything in this folder to it (or drag the files in via the web UI)
3. Repo **Settings → Pages** → Source `Deploy from a branch`, Branch `main`
4. After a minute or two, open `https://yourusername.github.io`

## 2. Things to set first (in `_config.yml`)

- `nickname` — your name
- `bio` — the short line under your name
- `instagram` — your handle without the @ (leave blank to hide)
- `profile_image` — upload your photo to `assets/img/` and point to it

Also edit `about.html` to write your bio page.

## 3. The two databases (this is the important part)

There are two table pages that look identical but pull from different folders:

- **Paper Log** (home page) — papers you've read. Files go in `_posts/`.
- **My Papers** (`/my-papers/`) — papers you've written. Files go in `_my_papers/`.

The sidebar nav links to About me, Paper Log, and My Papers.

### Adding to Paper Log (papers you read)

Add a file to `_posts/` named `YYYY-MM-DD-title.md` (the date prefix is
required by Jekyll and is treated as the day you read it):

```yaml
---
layout: post
title: "Paper title"
date: 2026-06-12
authors: "Vaswani et al."   # optional
venue: "NeurIPS"            # optional
year: 2017                  # optional
tags: [NLP, Transformer]    # optional
link: https://arxiv.org/... # optional
---
```

### Adding to My Papers (papers you wrote)

Add a file to `_my_papers/` named `title.md` — **no date prefix here**, just a
plain name. Put the date inside the front matter instead:

```yaml
---
layout: post
title: "Your paper title"
date: 2026-05-01            # used for ordering
authors: "Your Name, Coauthor"
venue: "CVPR"
year: 2026
tags: [Vision, Method]
link: https://arxiv.org/...
---
```

Write the summary below the front matter in plain Markdown. Math uses
`$...$` / `$$...$$`. In both tables, `venue` and `tags` feed the filter
dropdowns and every column is sortable.

Samples to copy: `_posts/2026-06-12-attention-is-all-you-need.md` and
`_my_papers/your-first-paper.md`.

## 4. Getting found on Google (SEO)

The site is built to be search-friendly: every page has a title, description,
canonical URL, and social-share tags, plus an auto-generated sitemap and a
robots.txt.

**Required:** set `url` in `_config.yml` to your real address
(e.g. `https://yourusername.github.io`, no trailing slash). The sitemap and
canonical links use it to build absolute URLs.

After you deploy:
- Your sitemap is at `https://yourusername.github.io/sitemap.xml`
- Your robots file is at `https://yourusername.github.io/robots.txt`

**Tell Google directly (recommended, speeds things up):**
1. Go to Google Search Console and add a property using your site URL.
2. Choose the **HTML tag** verification method. Copy the token (the value
   inside `content="..."`) and paste it into `google_site_verification` in
   `_config.yml`, then redeploy. Search Console will then verify you.
3. In Search Console, open **Sitemaps** and submit `sitemap.xml`.

**Nice link previews (optional):** add a 1200x630 PNG to `assets/img/`, then set
`og_image: /assets/img/og.png` in `_config.yml`. That image shows when your
links are shared on social media or chat apps.

Note: search engines crawl on their own schedule, so it can take days to a few
weeks to appear, and Google decides what to index and how to rank it. The steps
above make your site fully crawlable and submittable, which is everything you
control from your side.
