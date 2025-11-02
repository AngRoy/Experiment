import os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
from google import genai
from google.genai import types

DEFAULT_IMG_MODEL = os.getenv("GEMINI_IMG_MODEL", "gemini-2.0-flash-preview-image-generation")

def _client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)

def _pick_inline_image(parts) -> Optional[bytes]:
    for p in parts or []:
        inline = getattr(p, "inline_data", None)
        data = getattr(inline, "data", None) if inline else None
        if data:
            return data
    return None

def _gen_one_image(prompt: str, model_name: Optional[str] = None, tries: int = 2, backoff: float = 0.8) -> bytes:
    client = _client()
    model = model_name or DEFAULT_IMG_MODEL

    last_err = None
    for attempt in range(1, tries + 1):
        try:
            res = client.models.generate_content(
                model=model,
                contents=prompt,
                # many Gemini variants need TEXT+IMAGE to actually return the image bytes
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
            )
            parts = res.candidates[0].content.parts if res.candidates else []
            data = _pick_inline_image(parts)
            if data:
                return data
            last_err = RuntimeError("No image data in response")
        except Exception as e:
            last_err = e

        if attempt < tries:
            time.sleep(backoff * attempt)

    raise last_err or RuntimeError("image generation failed")

def gen_images(
    prompts: List[Tuple[int, str]],
    out_dir: str,
    concurrency: int = 5,
    model_name: Optional[str] = None
) -> Dict[int, str]:
    os.makedirs(out_dir, exist_ok=True)
    saved: Dict[int, str] = {}
    concurrency = max(1, min(concurrency, 32))

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
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
