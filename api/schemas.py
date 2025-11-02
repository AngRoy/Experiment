from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

# ---- Core types ----
OutputKind = Literal["text", "diagram", "image", "audio", "video"]
TextDepth = Literal["brief", "detailed", "very_detailed"]
SegmentKind = Literal["content", "diagram", "image"]

class NormalizeRequest(BaseModel):
    chat: str
    defaults: Optional[Dict[str, Any]] = Field(default_factory=dict)

class TaskSpec(BaseModel):
    topic: str
    audience: Optional[str] = "general"
    language: Optional[str] = "en"
    difficulty: Optional[str] = "intro"
    outputs: List[OutputKind] = Field(default_factory=lambda: ["text", "diagram", "image"])
    keywords: List[str] = Field(default_factory=list)
    image_ideas: List[str] = Field(default_factory=list)
    # content richness and minimum assets
    text_depth: TextDepth = "very_detailed"
    min_diagrams: int = 2   # default: 2 diagrams if user didn’t specify
    min_images: int = 2     # default: 2 images if user didn’t specify

class HelpfulNotesRequest(BaseModel):
    queries: List[str]
    kfinal: int = 10
    mmr_k: int = 20
    lambda_mmr: float = 0.6
    use_cross_encoder: bool = False

class ChunkPayload(BaseModel):
    doc_id: Optional[str] = None
    title: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    chunk_id: Optional[str] = None
    text: str

class HelpfulNotesResponse(BaseModel):
    chunks: List[ChunkPayload]
    notes: List[str]

class GenerateRequest(BaseModel):
    task_spec: TaskSpec
    helpful_notes: List[str] = Field(default_factory=list)
    model: Optional[str] = None

# ---- Lesson models (draft JSON) ----
class LessonSegment(BaseModel):
    section: Optional[str] = None
    kind: SegmentKind = "content"
    text: str
    text_format: Literal["md", "plain"] = "md"
    mermaid: Optional[str] = None
    image_prompt: Optional[str] = None
    alt_text: Optional[str] = None

class LessonDraft(BaseModel):
    title: str
    segments: List[LessonSegment]
    narration: Optional[str] = None

class FullLessonRequest(BaseModel):
    chat: str
    allow_web: bool = False
    model: Optional[str] = None

# ---- Enriched (rendered assets) ----
class EnrichedLessonSegment(LessonSegment):
    diagram_path: Optional[str] = None
    image_path: Optional[str] = None
    diagram_url: Optional[str] = None
    image_url: Optional[str] = None

class LessonWithAssets(BaseModel):
    title: str
    segments: List[EnrichedLessonSegment]
    narration: Optional[str] = None
    artifacts_root: Optional[str] = None
