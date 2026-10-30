import { useState } from "react";
import samples from "./samples.json";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const THRESHOLD = 0.3;

function riskLevel(score) {
  if (score < THRESHOLD) return { label: "Low risk", tone: "low" };
  if (score < 0.6) return { label: "Elevated risk", tone: "mid" };
  return { label: "High risk", tone: "high" };
}

function RiskMeter({ score }) {
  const pct = Math.min(Math.max(score, 0), 1) * 100;
  return (
    <div className="meter-wrap">
      <div className="meter">
        <div className="meter-threshold" style={{ left: `${THRESHOLD * 100}%` }} />
        <div className="meter-pointer" style={{ left: `${pct}%` }} />
      </div>
      <div className="meter-scale">
        <span>0</span>
        <span style={{ marginLeft: `${THRESHOLD * 100 - 6}%` }}>flag at {THRESHOLD}</span>
        <span>1</span>
      </div>
    </div>
  );
}

export default function App() {
  const [selected, setSelected] = useState(0);
  const [amount, setAmount] = useState(String(samples[0].features.Amount));
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);

  const sample = samples[selected];

  function pickSample(i) {
    setSelected(i);
    setAmount(String(samples[i].features.Amount));
    setResult(null);
    setError("");
  }

  async function analyze() {
    setError("");
    const amountNum = Number(amount);
    if (amount.trim() === "" || Number.isNaN(amountNum)) {
      setError("Please enter a valid amount.");
      return;
    }
    if (amountNum < 0) {
      setError("Amount cannot be negative.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: { ...sample.features, Amount: amountNum } }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(
          typeof body.detail === "string"
            ? body.detail
            : "The server could not process this transaction."
        );
      }

      const data = await response.json();
      const edited = amountNum !== sample.features.Amount;
      const entry = { ...data, name: sample.name, actual: sample.actual, edited };
      setResult(entry);
      setHistory((h) => [entry, ...h].slice(0, 5));
    } catch (e) {
      setError(
        e instanceof TypeError
          ? "Cannot reach the API. Make sure the backend is running (uvicorn api.main:app --reload)."
          : e.message
      );
    } finally {
      setLoading(false);
    }
  }

  const level = result ? riskLevel(result.risk_score) : null;

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Fraud Risk Monitor</h1>
          <p className="subtitle">Machine learning demo on a public credit-card dataset</p>
        </div>
        <span className="badge">Demo only</span>
      </header>

      <p className="notice">
        This is a portfolio demo. It is not connected to any bank or real-time payment
        system, and the transactions below are samples from a public research dataset.
      </p>

      <section className="stats">
        <div className="stat"><span className="stat-value">0.89</span><span className="stat-label">Precision (test set)</span></div>
        <div className="stat"><span className="stat-value">0.77</span><span className="stat-label">Recall (test set)</span></div>
        <div className="stat"><span className="stat-value">0.79</span><span className="stat-label">PR-AUC (test set)</span></div>
        <div className="stat"><span className="stat-value">95</span><span className="stat-label">Frauds in 56,746 test rows</span></div>
      </section>

      <main className="grid">
        <section className="card">
          <h2>1. Choose a sample transaction</h2>
          <div className="sample-buttons">
            {samples.map((s, i) => (
              <button
                key={s.name}
                className={i === selected ? "chip chip-active" : "chip"}
                onClick={() => pickSample(i)}
              >
                {s.name}
              </button>
            ))}
          </div>

          <label className="field">
            <span>Amount</span>
            <input
              type="text"
              inputMode="decimal"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
            />
          </label>

          <div className="features">
            {["V12", "V14", "V17"].map((k) => (
              <div key={k} className="feature">
                <span>{k}</span>
                <strong>{sample.features[k].toFixed(2)}</strong>
              </div>
            ))}
          </div>
          <p className="hint">
            The model receives 29 anonymized inputs (V1 to V28 and Amount). Three are
            shown here. You can edit the amount to see how the score reacts.
          </p>

          <button className="primary" onClick={analyze} disabled={loading}>
            {loading ? "Analyzing..." : "Analyze transaction"}
          </button>
          {error && <div className="error">{error}</div>}
        </section>

        <section className="card">
          <h2>2. Result</h2>
          {!result && !loading && (
            <p className="empty">Pick a sample and press Analyze to see the model's decision.</p>
          )}
          {loading && <p className="empty">Contacting the model...</p>}
          {result && !loading && (
            <>
              <div className={`verdict verdict-${level.tone}`}>
                {result.is_fraud ? "Flagged as potential fraud" : "Looks legitimate"}
              </div>
              <div className="score-row">
                <span className="score">{result.risk_score.toFixed(3)}</span>
                <span className={`level level-${level.tone}`}>{level.label}</span>
              </div>
              <RiskMeter score={result.risk_score} />
              <p className="hint">
                The risk score runs from 0 to 1 and comes from the share of trees in a
                Random Forest that vote "fraud". A transaction is flagged when the score
                is at least {result.threshold}. It is a ranking signal, not a guaranteed
                probability.
              </p>
              {!result.edited && (
                <p className="truth">
                  Dataset label for this sample: <strong>{result.actual}</strong>
                  {" "}
                  {(result.actual === "fraud") === result.is_fraud
                    ? "(model agrees)"
                    : "(model disagrees)"}
                </p>
              )}
            </>
          )}
        </section>
      </main>

      {history.length > 0 && (
        <section className="card history">
          <h2>Recent checks</h2>
          <ul>
            {history.map((h, i) => (
              <li key={i}>
                <span>{h.name}{h.edited ? " (amount edited)" : ""}</span>
                <span className="mono">{h.risk_score.toFixed(3)}</span>
                <span className={h.is_fraud ? "tag tag-flag" : "tag tag-ok"}>
                  {h.is_fraud ? "flagged" : "ok"}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <footer className="footer">
        Random Forest trained on a public, anonymized dataset. Results are for demonstration
        and do not reflect real-world fraud detection performance.
      </footer>
    </div>
  );
}