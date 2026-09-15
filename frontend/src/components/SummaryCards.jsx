function formatPercent(value) {
    if (value === null || value === undefined) return "—";
    return `${Math.round(value * 100)}%`;
  }
  
  function formatSeconds(value) {
    if (value === null || value === undefined) return "—";
    return `${value.toFixed(1)}s`;
  }
  
  export default function SummaryCards({ run }) {
    if (!run) {
      return (
        <div className="panel">
          <div className="summary-empty">
            No eval runs yet. Click "Run new eval" to generate the first one.
          </div>
        </div>
      );
    }
  
    const hallucinationClass = run.hallucination_rate > 0 ? "fail" : "pass";
  
    return (
      <div className="panel">
        <div className="summary-grid">
          <div className="summary-cell">
            <div className="label">Accuracy</div>
            <div className="value">{formatPercent(run.accuracy)}</div>
          </div>
          <div className="summary-cell">
            <div className="label">Tool selection correct</div>
            <div className="value">
              {formatPercent(run.tool_selection_correct_rate)}
            </div>
          </div>
          <div className="summary-cell">
            <div className="label">Hallucination rate</div>
            <div className={`value ${hallucinationClass}`}>
              {formatPercent(run.hallucination_rate)}
            </div>
          </div>
          <div className="summary-cell">
            <div className="label">Avg latency</div>
            <div className="value">{formatSeconds(run.avg_latency_seconds)}</div>
          </div>
        </div>
        {run.stopped_early_reason && (
          <div className="partial-note">
            Partial run — {run.completed_questions}/{run.total_questions}{" "}
            questions completed. {run.stopped_early_reason}
          </div>
        )}
      </div>
    );
  }