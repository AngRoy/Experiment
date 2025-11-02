import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import MermaidBlock from "./MermaidBlock";          // keep available
import { LessonSegment } from "../types";

export default function SegmentCard({
  seg,
  allowClientFallback = false,                     // default off
}: {
  seg: LessonSegment;
  allowClientFallback?: boolean;
}) {
  const hasDiagramImg = !!seg.diagram_url;
  const hasImage = !!seg.image_url;
  const canClientRender = allowClientFallback && !!seg.mermaid && !seg.diagram_url;

  return (
    <div className="card">
      {seg.section && <div className="badge">{seg.section}</div>}

      <div style={{ marginTop: 8 }}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            // hide fenced ```mermaid blocks inside markdown
            code({ className, children, ...props }) {
              const lang = (className || "").replace("language-", "");
              if ((lang || "").toLowerCase() === "mermaid") return null;
              return <code className={className} {...props}>{children}</code>;
            },
          }}
        >
{seg.text || ""}
        </ReactMarkdown>
      </div>

      {hasDiagramImg && (
        <>
          <img className="asset" src={seg.diagram_url!} alt={seg.alt_text || "diagram"} />
          <div className="meta">diagram</div>
        </>
      )}

      {canClientRender && (
        <>
          <MermaidBlock code={seg.mermaid!} />
          <div className="meta">diagram (client fallback)</div>
        </>
      )}

      {hasImage && (
        <>
          <img className="asset" src={seg.image_url!} alt={seg.alt_text || "image"} />
          <div className="meta">image</div>
        </>
      )}
    </div>
  );
}
