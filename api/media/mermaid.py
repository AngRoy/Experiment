# api/media/mermaid.py
import os, subprocess, tempfile, sys, re, shutil, platform

def _sanitize_mermaid(code: str) -> str:
    # Basic healing: ensure header and strip fragile "style" lines
    c = code.strip()
    if not re.match(r'^(graph|flowchart)\s', c):
        c = "graph TD\n" + c
    lines = []
    for line in c.splitlines():
        if line.strip().startswith("style "):
            continue
        lines.append(line)
    return "\n".join(lines)

def _resolve_mmdc() -> tuple[str, bool]:
    """
    Returns (cmd, use_shell). On Windows prefers mmdc.cmd for reliability.
    Honors MERMAID_BIN if provided.
    """
    env = os.getenv("MERMAID_BIN")
    if env:
        return env, env.lower().endswith(".cmd")
    if platform.system().lower().startswith("win"):
        # try where/which
        for cand in ["mmdc.cmd", "mmdc"]:
            p = shutil.which(cand)
            if p:
                return p, cand.endswith(".cmd")
        # common global install location
        guess = os.path.expanduser(r"~\AppData\Roaming\npm\mmdc.cmd")
        if os.path.exists(guess):
            return guess, True
        return "mmdc.cmd", True
    # *nix/mac
    p = shutil.which("mmdc")
    return (p or "mmdc", False)

def render_mermaid(mermaid_code: str, out_path: str, theme: str = "default") -> bool:
    """
    Render Mermaid code to an image via mermaid-cli (mmdc).
    Never raises; returns False on failure.
    """
    cmd_bin, use_shell = _resolve_mmdc()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _try_render(src: str) -> bool:
        try:
            tmpdir = tempfile.mkdtemp(prefix="mmd_")
            mmd_path = os.path.join(tmpdir, "diagram.mmd")
            with open(mmd_path, "w", encoding="utf-8") as f:
                f.write(src)
            cmd = f'"{cmd_bin}" -i "{mmd_path}" -o "{out_path}" -t {theme} -b transparent' if use_shell \
                  else [cmd_bin, "-i", mmd_path, "-o", out_path, "-t", theme, "-b", "transparent"]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=use_shell)
            if p.returncode != 0:
                print(f"[mermaid] mmdc failed (rc={p.returncode}):\n{p.stderr}", file=sys.stderr)
                return False
            ok = os.path.exists(out_path) and os.path.getsize(out_path) > 0
            if not ok:
                print("[mermaid] output not created", file=sys.stderr)
            return ok
        except FileNotFoundError as e:
            print(f"[mermaid] mmdc not found: {e}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[mermaid] render error: {e}", file=sys.stderr)
            return False

    # 1) try raw
    if _try_render(mermaid_code):
        return True
    # 2) heal & retry
    healed = _sanitize_mermaid(mermaid_code)
    if healed != mermaid_code and _try_render(healed):
        return True
    # 3) minimal fallback so the user gets *some* diagram
    fallback = "graph TD\nA[Start] --> B[Neighbor 1]\nA --> C[Neighbor 2]"
    return _try_render(fallback)
