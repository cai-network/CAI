// SPDX-FileCopyrightText: 2025 cai Technologies Ltd
// SPDX-FileCopyrightText: 2026 CAI contributors
// SPDX-License-Identifier: Apache-2.0
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: [vitePreprocess()],

	kit: {
		paths: {
			relative: true
		},
		router: { type: 'hash' },
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			fallback: 'index.html',
			precompress: false,
			strict: true
		}),
		alias: {
			$lib: 'src/lib',
			$components: 'src/lib/components'
		}
	}
};

export default config;

