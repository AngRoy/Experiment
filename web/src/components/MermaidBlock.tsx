import { useEffect, useId, useRef, useState } from "react";
import mermaid from "mermaid";

mermaid.initialize({
  startOnLoad: false,
  theme: "dark",
  securityLevel: "loose"
});

export default function MermaidBlock({ code }: { code: string }) {
  const id = useId().replace(/:/g, "-");
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function render() {
      setError(null);
      try {
        const { svg } = await mermaid.render(`mmd-${id}`, code);
        if (!cancelled && ref.current) {
          ref.current.innerHTML = svg;
        }
      } catch (e: any) {
        setError(e?.message || "Mermaid render failed");
      }
    }
    render();
    return () => { cancelled = true; };
  }, [code, id]);

  if (error) return <div className="meta">Mermaid fallback error: {error}</div>;
  return <div className="mmd-wrap"><div ref={ref} /></div>;
}
