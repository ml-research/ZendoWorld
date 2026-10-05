import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build output goes directly into docs/demo/ so GitHub Pages can serve it.
// `base: "./"` keeps asset URLs relative so it works when embedded via <iframe>
// under any subpath.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "../docs/demo",
    emptyOutDir: true,
  },
});
