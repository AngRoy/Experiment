const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function postJSON<T>(path: string, payload: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchLesson(chat: string, rendered = true) {
  const endpoint = rendered ? "/lesson_rendered" : "/lesson";
  return postJSON(endpoint, { chat });
}
