// SPDX-FileCopyrightText: 2025 cai Technologies Ltd
// SPDX-FileCopyrightText: 2026 CAI contributors
// SPDX-License-Identifier: Apache-2.0
import tailwindcss from "@tailwindcss/vite";
import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [tailwindcss(), sveltekit()],
  server: {
    proxy: {
      "/v1": "http://localhost:52415",
      "/state": "http://localhost:52415",
      "/models": "http://localhost:52415",
      "/instance": "http://localhost:52415",
    },
  },
});
