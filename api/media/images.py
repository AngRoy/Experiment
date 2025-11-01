import os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
from google import genai
from google.genai import types
from .prompt_enricher import enrich_image_prompt

DEFAULT_IMG_MODEL = os.getenv("GEMINI_IMG_MODEL", "gemini-2.0-flash-preview-image-generation")

def _client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)

def _gen_one_image(prompt: str, model_name: Optional[str]=None) -> bytes:
    client = _client()
    model = model_name or DEFAULT_IMG_MODEL
    res = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["TEXT","IMAGE"])
    )
    parts = res.candidates[0].content.parts if res.candidates else []
    for p in parts:
        if getattr(p, "inline_data", None) and getattr(p.inline_data, "data", None):
            return p.inline_data.data
    if len(parts) > 1 and getattr(parts[1], "inline_data", None):
        return parts[1].inline_data.data
    raise RuntimeError("No image data returned by Gemini image generation")

def gen_images(prompts: List[Tuple[int, str]], out_dir: str, concurrency: int = 5,
               model_name: Optional[str]=None) -> Dict[int, str]:
    """
    Generate images in parallel (up to `concurrency`) for (index, prompt) pairs.
    Saves files under `out_dir` as img_{idx}.png. Returns {idx: path or "" on failure}.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved: Dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, 32))) as ex:
        prompts = [(i, enrich_image_prompt(p, topic="Breadth-First Search")) for (i, p) in prompts]
        futures = {ex.submit(_gen_one_image, prompt, model_name): idx for (idx, prompt) in prompts}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                img_bytes = fut.result()
                path = os.path.join(out_dir, f"img_{idx}.png")
                with open(path, "wb") as f:
                    f.write(img_bytes)
                saved[idx] = path
            except Exception as e:
                print(f"[image_gen] idx={idx} failed: {e}", file=sys.stderr)
                saved[idx] = ""
    return saved
