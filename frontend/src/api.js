// In local dev, requests to /api/... go through Vite's dev-server proxy
// (see vite.config.js) straight to the backend — VITE_API_BASE_URL is
// unset, so BASE_URL is "" and paths stay relative.
//
// In production the frontend and backend are deployed as separate
// services on different domains, so there's no proxy — VITE_API_BASE_URL
// must be set at build time (e.g. in Render's/Vercel's environment
// variables) to the deployed backend's full URL, e.g.
// https://agent-eval-dashboard-api.onrender.com
const BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

async function request(path, options) {
  const res = await fetch(`${BASE_URL}${path}`, options);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

export function triggerEvalRun() {
  // This call blocks until the full golden dataset has finished running
  // (several minutes) — the caller is responsible for showing a loading
  // state, since fetch() won't resolve until the backend responds.
  return request("/api/eval/run", { method: "POST" });
}

export function fetchRuns() {
  return request("/api/eval/runs");
}

export function fetchLatestRun() {
  return request("/api/eval/runs/latest");
}

export function fetchRunDetail(runId) {
  return request(`/api/eval/runs/${runId}`);
}