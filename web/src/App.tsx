import { useState } from "react";
import { fetchLesson } from "./api";
import { LessonWithAssets, LessonDraft, LessonSegment } from "./types";
import SegmentCard from "./components/SegmentCard";

type Data = LessonDraft | LessonWithAssets;

export default function App() {
  const [chat, setChat] = useState("teach me routing protocols");
  const [useRendered, setUseRendered] = useState(true);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    setData(null);
    try {
      const res = await fetchLesson(chat, useRendered);
      setData(res as Data);
    } catch (err: any) {
      setError(err?.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  // Only show segments that have some content (text or any asset)
  const visibleSegments: LessonSegment[] = (data?.segments ?? []).filter(
    (s) =>
      (s.text && s.text.trim().length > 0) ||
      !!s.diagram_url ||
      !!s.image_url ||
      !!s.mermaid ||
      !!s.image_prompt
  );

  return (
    <div className="container">
      <div className="header">
        <div className="title">UGTA Viewer</div>
        <span className="badge">
          {useRendered ? "rendered (PNG assets)" : "JSON-only"}
        </span>
      </div>

      <div className="panel">
        <form className="row" onSubmit={onSubmit}>
          <input
            className="input"
            value={chat}
            onChange={(e) => setChat(e.target.value)}
            placeholder="Ask: teach BFS with 3 diagrams and 4 images"
          />
          <button className="button" type="submit" disabled={loading}>
            {loading ? "Generating..." : "Generate"}
          </button>
          <label className="switch">
            <input
              type="checkbox"
              checked={useRendered}
              onChange={(e) => setUseRendered(e.target.checked)}
            />
            Use /lesson_rendered
          </label>
        </form>
        {error && (
          <div style={{ color: "var(--red)", marginTop: 10 }}>{error}</div>
        )}
      </div>

      {data && (
        <div style={{ marginTop: 16 }}>
          <h2>{data.title}</h2>
          <div className="grid">
            {visibleSegments.map((seg, idx) => (
              <SegmentCard key={idx} seg={seg} />
            ))}
          </div>
        </div>
      )}

      {!data && !loading && (
        <div style={{ marginTop: 16, color: "var(--muted)" }}>
          Try: <code>teach BFS with 3 diagrams and 4 images</code>
        </div>
      )}

      <div className="footer-space"></div>
    </div>
  );
}
