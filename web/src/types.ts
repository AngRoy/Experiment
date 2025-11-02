export type SegmentKind = "content" | "diagram" | "image";

export interface LessonSegment {
  section?: string | null;
  kind: SegmentKind;
  text: string;
  text_format: "md" | "plain";
  mermaid?: string | null;
  image_prompt?: string | null;
  alt_text?: string | null;

  // rendered assets (lesson_rendered only)
  diagram_path?: string | null;
  image_path?: string | null;
  diagram_url?: string | null;
  image_url?: string | null;
}

export interface LessonDraft {
  title: string;
  segments: LessonSegment[];
  narration?: string | null;
}

export interface LessonWithAssets extends LessonDraft {
  artifacts_root?: string | null;
}
