import { useEffect, useState, useCallback } from "react";
import { fetchRuns, fetchLatestRun, fetchRunDetail } from "./api.js";
import RunEvalButton from "./components/RunEvalButton.jsx";
import SummaryCards from "./components/SummaryCards.jsx";
import TrendChart from "./components/TrendChart.jsx";
import ResultsTable from "./components/ResultsTable.jsx";

export default function App() {
  const [runs, setRuns] = useState([]);
  const [latestRun, setLatestRun] = useState(null);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [runDetail, setRunDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const loadOverview = useCallback(async () => {
    try {
      const [allRuns, latest] = await Promise.all([
        fetchRuns(),
        fetchLatestRun().catch(() => null), // 404s if no completed runs yet
      ]);
      setRuns(allRuns);
      setLatestRun(latest);
      if (latest) {
        setSelectedRunId(latest.id);
      }
    } catch (err) {
      console.error("Failed to load eval run overview:", err);
    }
  }, []);

  useEffect(() => {
    loadOverview();
  }, [loadOverview]);

  useEffect(() => {
    if (selectedRunId == null) return;
    setDetailLoading(true);
    fetchRunDetail(selectedRunId)
      .then(setRunDetail)
      .catch((err) => console.error("Failed to load run detail:", err))
      .finally(() => setDetailLoading(false));
  }, [selectedRunId]);

  function handleRunComplete(newRunId) {
    loadOverview();
    setSelectedRunId(newRunId);
  }

  return (
    <div className="app">
      <div className="header">
        <div>
          <h1>Agent eval &amp; observability</h1>
          <p>Multi-source research agent — golden dataset evaluation</p>
        </div>
        <RunEvalButton onRunComplete={handleRunComplete} />
      </div>

      <SummaryCards run={latestRun} />
      <TrendChart runs={runs} />
      <ResultsTable
        runs={runs}
        selectedRunId={selectedRunId}
        onSelectRun={setSelectedRunId}
        runDetail={runDetail}
        loading={detailLoading}
      />
    </div>
  );
}