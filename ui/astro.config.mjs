// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import rehypeKatex from 'rehype-katex'; // relevant
import remarkMath from 'remark-math';   // relevant

import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
    site: 'https://example.com',
    integrations: [mdx({
        remarkPlugins: [remarkMath], // relevant
        rehypePlugins: [rehypeKatex] // relevant
		}), sitemap()],
});