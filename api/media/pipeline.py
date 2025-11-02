import os
from typing import Dict, Any, List
from .mermaid import render_mermaid
from .images import gen_images
from .prompt_enricher import enrich_image_prompt

def render_assets_for_lesson(lesson: Dict[str, Any], out_root: str, image_concurrency: int = 5) -> Dict[str, Any]:
    """
    Enrich a LessonDraft-like dict by rendering Mermaid diagrams and generating images.
    - Diagrams: diagram_{i}.png (serial)
    - Images:   img_{i}.png (parallel, capped by image_concurrency)
    Returns the same dict with segments[i].diagram_path / image_path added.
    """
    segs: List[Dict[str, Any]] = list(lesson.get("segments", []))
    os.makedirs(out_root, exist_ok=True)
    diag_dir = os.path.join(out_root, "diagrams")
    img_dir  = os.path.join(out_root, "images")
    os.makedirs(diag_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # 1) Mermaid → PNG
    for i, seg in enumerate(segs):
        mmd = seg.get("mermaid")
        if isinstance(mmd, str) and mmd.strip():
            out_path = os.path.join(diag_dir, f"diagram_{i}.png")
            ok = render_mermaid(mmd, out_path)
            seg["diagram_path"] = out_path if ok else ""

    # 2) Image prompts → PNG (parallel)
    prompts = []
    for i, seg in enumerate(segs):
        p = seg.get("image_prompt")
        if isinstance(p, str) and p.strip():
            enriched = enrich_image_prompt(p.strip(), topic=lesson.get("title"))
            prompts.append((i, enriched))

    if prompts:
        saved = gen_images(prompts, out_dir=img_dir, concurrency=image_concurrency)
        for i, path in saved.items():
            if 0 <= i < len(segs):
                segs[i]["image_path"] = path

    out = dict(lesson)
    out["segments"] = segs
    return out
