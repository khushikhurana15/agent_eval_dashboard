import { useState, Fragment } from "react";

function truncate(text, max = 90) {
  if (!text) return "";
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

function ToolCell({ result }) {
    const tools = result.tools_used || [];
    const mismatch = !result.tool_correct;
  
    // Count occurrences so repeated tool calls show as "toolName ×3"
    // instead of a confusing "toolName, toolName, toolName" — multiple
    // calls to the same tool are a real, meaningful agent behavior
    // (e.g. retrying a search with a refined query), not a display bug.
    const counts = {};
    tools.forEach((t) => { counts[t] = (counts[t] || 0) + 1; });
    const display = Object.entries(counts)
      .map(([name, count]) => (count > 1 ? `${name} ×${count}` : name))
      .join(", ") || "none";
  
    return (
      <div>
        <div className={`tool-name ${mismatch ? "mismatch" : ""}`}>{display}</div>
        {mismatch && (
          <div className="tool-name" style={{ color: "#5C6270" }}>
            expected: {result.expected_tool}
          </div>
        )}
      </div>
    );
  }

function ConfidenceCell({ result }) {
  if (result.rag_confidence_distance === null || result.rag_confidence_distance === undefined) {
    return <span className="confidence-value">—</span>;
  }
  return (
    <span className="confidence-value">
      {result.rag_confidence_distance.toFixed(3)}
      {result.rag_gated ? " (gated)" : ""}
    </span>
  );
}

function TraceDetail({ result }) {
  const steps = result.reasoning_trace || [];
  return (
    <tr className="trace-row">
      <td colSpan={5}>
        <div className="trace-block">
          <h4>Reasoning trace</h4>
          {steps.length === 0 ? (
            <div className="no-trace">No tool was called — answered directly.</div>
          ) : (
            steps.map((step, i) => (
              <div className="trace-step" key={i}>
                <span className="step-tool">{step.tool_name}</span>
                {"  "}
                {JSON.stringify(step.tool_input)}
                {"\n→ "}
                {truncate(step.tool_output, 400)}
              </div>
            ))
          )}
          <h4 style={{ marginTop: 16 }}>Final answer</h4>
          <div className="trace-answer">{result.final_answer}</div>
        </div>
      </td>
    </tr>
  );
}

export default function ResultsTable({ runs, selectedRunId, onSelectRun, runDetail, loading }) {
  const [expandedId, setExpandedId] = useState(null);

  return (
    <div className="panel results-panel">
      <div className="results-header">
        <h2>Test case results</h2>
        {runs && runs.length > 0 && (
          <select
            className="run-select"
            value={selectedRunId ?? ""}
            onChange={(e) => onSelectRun(Number(e.target.value))}
          >
            {[...runs].reverse().map((run) => (
              <option key={run.id} value={run.id}>
                Run #{run.id} — {Math.round((run.accuracy ?? 0) * 100)}% accuracy
              </option>
            ))}
          </select>
        )}
      </div>

      {loading && <div className="summary-empty">Loading run detail…</div>}

      {!loading && runDetail && (
        <table className="results-table">
          <thead>
            <tr>
              <th style={{ width: "34%" }}>Question</th>
              <th style={{ width: "20%" }}>Tool used</th>
              <th style={{ width: "12%" }}>Confidence</th>
              <th style={{ width: "10%" }}>Latency</th>
              <th style={{ width: "10%" }}>Result</th>
            </tr>
          </thead>
          <tbody>
            {runDetail.results.map((result) => {
              const isOpen = expandedId === result.id;
              return (
                // NOTE: shorthand <>...</> fragments cannot take a `key`
                // prop, which is required when returning multiple root
                // elements from inside a .map(). Using the full
                // `<Fragment key={...}>` form here instead — the key lives
                // on the Fragment itself now, not on the inner <tr>.
                <Fragment key={result.id}>
                  <tr
                    className="result-row"
                    onClick={() => setExpandedId(isOpen ? null : result.id)}
                  >
                    <td className="question-cell">
                      <span className={`expand-chevron ${isOpen ? "open" : ""}`}>
                        ▸
                      </span>{" "}
                      <span className="question-text">
                        {truncate(result.question, 90)}
                      </span>
                      <div className="category-tag">{result.category}</div>
                    </td>
                    <td>
                      <ToolCell result={result} />
                    </td>
                    <td>
                      <ConfidenceCell result={result} />
                    </td>
                    <td className="confidence-value">
                      {result.latency_seconds != null
                        ? `${result.latency_seconds.toFixed(1)}s`
                        : "—"}
                    </td>
                    <td>
                      <span className={`badge ${result.passed ? "pass" : "fail"}`}>
                        {result.passed ? "Pass" : "Fail"}
                      </span>
                    </td>
                  </tr>
                  {isOpen && <TraceDetail result={result} />}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}