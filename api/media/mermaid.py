import os
import subprocess
import tempfile
from typing import Optional

# Resolve Mermaid CLI path:
# - use MERMAID_BIN if provided (Windows users often set mmdc.cmd)
# - otherwise try "mmdc" (must be on PATH)
def _resolve_mermaid_bin() -> str:
    env_bin = os.getenv("MERMAID_BIN")
    if env_bin and os.path.exists(env_bin):
        return env_bin
    # fallbacks
    return "mmdc"  # works if mermaid-cli is on PATH

def render_mermaid(mermaid_code: str, out_png_path: str, background: str = "transparent") -> bool:
    """
    Render a Mermaid diagram to PNG using the Mermaid CLI (mmdc).
    Returns True on success, False on failure. Does not raise.
    """
    try:
        os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
        # write a temp .mmd file
        with tempfile.NamedTemporaryFile("w", suffix=".mmd", delete=False, encoding="utf-8") as tf:
            tf.write(mermaid_code)
            tmp_in = tf.name

        bg = background or os.getenv("MERMAID_BG", "#ffe45e")  # light yellow
        cmd = [
            _resolve_mermaid_bin(),
            "-i", tmp_in,
            "-o", out_png_path,
            "--backgroundColor", bg,
            "--theme", "neutral"
        ]

        # Run and capture output; don’t raise on non-zero to allow repair fallback upstream
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok = proc.returncode == 0 and os.path.exists(out_png_path) and os.path.getsize(out_png_path) > 0

        if not ok:
            # helpful for logs; FastAPI will still handle repair steps if you implemented them
            print(f"[mermaid] render failed rc={proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")

        try:
            os.remove(tmp_in)
        except Exception:
            pass

        return ok
    except Exception as e:
        print(f"[mermaid] exception: {e}")
        return False
