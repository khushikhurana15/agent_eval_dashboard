import { useState } from "react";
import { triggerEvalRun } from "../api.js";

export default function RunEvalButton({ onRunComplete }) {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  async function handleClick() {
    setRunning(true);
    setError(null);
    try {
      const result = await triggerEvalRun();
      onRunComplete(result.run_id);
    } catch (err) {
      setError(
        "Eval run failed to start or complete. Check the backend terminal for details."
      );
    } finally {
      setRunning(false);
    }
  }

  return (
    <div style={{ textAlign: "right" }}>
      <button className="run-button" onClick={handleClick} disabled={running}>
        {running ? (
          <>
            <span className="spinner" />
            Running eval…
          </>
        ) : (
          "Run new eval"
        )}
      </button>
      {error && <div className="run-error">{error}</div>}
    </div>
  );
}