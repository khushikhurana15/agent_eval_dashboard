import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxying /api to the FastAPI backend means the frontend code can just
// call fetch("/api/...") with no hardcoded host/port, and the browser
// never sees a cross-origin request during local dev (the backend's CORS
// middleware is still there as a fallback for anyone hitting it directly).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});