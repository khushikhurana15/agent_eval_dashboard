// Thin wrapper around the FastAPI endpoints. Requests go to /api/... which
// Vite's dev-server proxy (see vite.config.js) forwards to the backend at
// http://127.0.0.1:8000 — the frontend code never needs to know the host.

async function request(path, options) {
    const res = await fetch(path, options);
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