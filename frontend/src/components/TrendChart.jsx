import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function TrendChart({ runs }) {
  if (!runs || runs.length === 0) {
    return (
      <div className="panel chart-panel">
        <h2>Accuracy trend across runs</h2>
        <div className="chart-empty">
          Trend will appear once at least one eval run has completed.
        </div>
      </div>
    );
  }

  const data = runs.map((run) => ({
    label: `#${run.id}`,
    accuracy: Math.round((run.accuracy ?? 0) * 100),
    hallucination: Math.round((run.hallucination_rate ?? 0) * 100),
  }));

  return (
    <div className="panel chart-panel">
      <h2>Accuracy &amp; hallucination rate across runs</h2>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -4, bottom: 0 }}>
          <CartesianGrid stroke="#2A2F38" vertical={false} />
          <XAxis
            dataKey="label"
            stroke="#5C6270"
            tick={{ fontFamily: "IBM Plex Mono", fontSize: 11 }}
            axisLine={{ stroke: "#2A2F38" }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            stroke="#5C6270"
            tick={{ fontFamily: "IBM Plex Mono", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={42}
          />
          <Tooltip
            contentStyle={{
              background: "#1C2027",
              border: "1px solid #2A2F38",
              borderRadius: 6,
              fontFamily: "IBM Plex Mono",
              fontSize: 12,
            }}
            labelStyle={{ color: "#8B92A0" }}
          />
          <Line
            type="monotone"
            dataKey="accuracy"
            name="Accuracy %"
            stroke="#4CAF7D"
            strokeWidth={2}
            dot={{ r: 3, fill: "#4CAF7D" }}
          />
          <Line
            type="monotone"
            dataKey="hallucination"
            name="Hallucination %"
            stroke="#E1625B"
            strokeWidth={2}
            dot={{ r: 3, fill: "#E1625B" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}