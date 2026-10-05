import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API base URL is injected at build time via VITE_API_URL and falls back to
// the local FastAPI service. In dev, requests to /api are proxied to :8000 so the
// browser makes same-origin calls.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
