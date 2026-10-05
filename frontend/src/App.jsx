import { useEffect, useState } from "react";
import { format } from "./format.js";

// In dev the Vite proxy forwards /api -> FastAPI on :8000. In a container build
// set VITE_API_URL to the API service URL.
const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

export default function App() {
  const [cfg, setCfg] = useState(null);
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(null); // { job_id, extraction }
  const [decisions, setDecisions] = useState({}); // index -> bool
  const [published, setPublished] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/config`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setCfg)
      .catch(() => setError("API unreachable - is the service running on :8000?"));
  }, []);

  async function process() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setJob(null);
    setDecisions({});
    setPublished([]);
    try {
      const body = new FormData();
      body.append("file", file);
      const resp = await fetch(`${API_BASE}/process`, { method: "POST", body });
      if (!resp.ok) throw new Error((await resp.json())?.detail || `API returned ${resp.status}`);
      setJob(await resp.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function decide(index, approved) {
    setDecisions((d) => ({ ...d, [index]: approved }));
  }

  async function publish() {
    if (!job) return;
    setLoading(true);
    setError(null);
    try {
      const payload = {
        decisions: Object.entries(decisions).map(([index, approved]) => ({
          index: Number(index),
          approved,
        })),
        publish: true,
      };
      const resp = await fetch(`${API_BASE}/jobs/${job.job_id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) throw new Error(`API returned ${resp.status}`);
      const data = await resp.json();
      setJob({ job_id: data.job_id, extraction: data.extraction });
      setPublished(data.published);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const ext = job?.extraction;
  const approvedCount = Object.values(decisions).filter(Boolean).length;

  return (
    <main className="app">
      <header>
        <h1>TrustFeed</h1>
        <p className="subtitle">
          Foreign-language news audio -> transcript -> review-ready social clips.
          Bad actors move fast; newsrooms should move faster.
        </p>
      </header>

      {cfg && (
        <div className="metrics">
          <span>ASR <strong>Whisper {cfg.asr_model}</strong></span>
          <span>Extractor <strong>{cfg.llm_backend === "heuristic" ? "heuristic" : cfg.llm_model}</strong></span>
          <span>Max clips <strong>{cfg.max_clips}</strong></span>
        </div>
      )}

      <label htmlFor="file">Upload news audio or video (any language)</label>
      <input
        id="file"
        type="file"
        accept="audio/*,video/*"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />
      <button className="primary" onClick={process} disabled={loading || !file}>
        {loading && !ext ? "Transcribing & extracting..." : "Process"}
      </button>

      {error && <p className="error">{error}</p>}

      {ext && (
        <section className="result">
          <div className="pill-row">
            <span className="pill">Language: {ext.language}</span>
            <span className="pill">Duration: {format.time(ext.duration)}</span>
            <span className="pill">{ext.clips.length} clip candidates</span>
            <span className={`pill ${ext.status === "approved" ? "ok" : "warn"}`}>
              {ext.status === "approved" ? "approved" : "pending review"}
            </span>
          </div>

          {/* Observability: the numbers an operator cares about */}
          <p className="telemetry">
            asr {ext.asr_seconds}s | extract {ext.extract_seconds}s | backend {ext.backend}
            {" | "}JSON valid 1st try: {ext.schema_valid_first_try ? "yes" : "repaired"}
          </p>

          {ext.summary && (
            <>
              <h2>Summary</h2>
              <p className="summary">{ext.summary}</p>
            </>
          )}

          {ext.chapters?.length > 0 && (
            <>
              <h2>Chapters</h2>
              <ul className="chapters">
                {ext.chapters.map((c, i) => (
                  <li key={i}>
                    <span className="ts">{format.time(c.start)}</span>
                    <strong>{c.title}</strong>
                    {c.summary && <span className="chapter-sum"> - {c.summary}</span>}
                  </li>
                ))}
              </ul>
            </>
          )}

          <h2>Clip candidates - approve what should go out</h2>
          <p className="gate-note">
            Nothing is published until an editor approves it. This gate is the point:
            speed with a human check on trust.
          </p>
          {ext.clips.map((clip, i) => (
            <article key={i} className={`clip ${decisions[i] === true ? "approved" : decisions[i] === false ? "rejected" : ""}`}>
              <div className="clip-head">
                <span className="ts big">{clip.timespan}</span>
                <div className="clip-actions">
                  <button className={`approve ${decisions[i] === true ? "on" : ""}`} onClick={() => decide(i, true)}>
                    Approve
                  </button>
                  <button className={`reject ${decisions[i] === false ? "on" : ""}`} onClick={() => decide(i, false)}>
                    Reject
                  </button>
                </div>
              </div>
              <p className="caption">"{clip.caption}"</p>
              {clip.quote && <p className="quote">Quote: {clip.quote}</p>}
              {clip.rationale && <p className="rationale">Why: {clip.rationale}</p>}
            </article>
          ))}

          <button className="primary publish" onClick={publish} disabled={loading || approvedCount === 0}>
            Publish {approvedCount} approved clip{approvedCount === 1 ? "" : "s"}
          </button>
        </section>
      )}

      {published.length > 0 && (
        <section className="published">
          <h2>Ready to post</h2>
          {published.map((p, i) => (
            <div key={i} className="post">
              <span className="ts">{p.timespan}</span>
              <p className="caption">"{p.caption}"</p>
              <p className="channels">-> {p.channels.join(" | ")}</p>
            </div>
          ))}
        </section>
      )}

      <footer>
        Whisper (faster-whisper) ASR | local LLM extraction | human-in-the-loop
        editorial gate. Sample audio: FLEURS (CC-BY). No data leaves your machine.
      </footer>
    </main>
  );
}
