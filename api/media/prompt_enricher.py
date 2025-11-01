# api/media/prompt_enricher.py
from typing import Optional

def enrich_image_prompt(raw: str, topic: Optional[str] = None) -> str:
    """
    Expand terse prompts into schematic, labeled, unambiguous prompts.
    Tuned for CS/algorithms educational imagery.
    """
    topic_hint = f" about {topic}" if topic else ""
    return (
        f"Create a clean 2D vector schematic{topic_hint}. "
        f"Primary goal: accurately illustrate the concept, not a photo. "
        f"Style: flat, minimal, high-contrast, no textures, no drop shadows, no photorealism. "
        f"Use a white background, thin black outlines, and a limited accent palette. "
        f"Include clear labels and arrows. "
        f"Avoid text paragraphs inside the image; use concise labels only. "
        f"Render at 1024x1024. "
        f"Content: {raw} "
        f"If relevant, show BFS levels as concentric ‘rings’ or colored layers (L0=source, L1, L2...). "
        f"Ensure nodes are distinct circles with consistent spacing; label nodes A, B, C, D... starting at the source. "
        f"Avoid people, faces, hands, and extraneous scenery."
    )
