# Mesh-Dev-Blog

## Setup

0. `cd ui`
1. Run `yarn install`
2. Run `yarn dev`

## Writing a blog post

1. Blogs live in `ui/src/content/blog/`
2. Add either a `.mdx` or `.md` file 
3. Each blog needs to start with following heading:
```
---
title: TITLE
description: BLOG DESCRIPTION
pubDate: DATE
heroImage: /path/to/blog/image
author: YOUR NAME
---
```
4. Any static files such as images or html embeddings go in `ui/public/images/blog_name` or `ui/public/html/blog_name` respectively 