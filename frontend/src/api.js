// All HTTP calls to the Go API live here; pages never call fetch directly.
// Paths are relative ("/api/..."): in dev Vite forwards them, in the demo nginx does,
// so the same build works in both without knowing where the API runs.

export class ApiError extends Error {
  constructor(status, message, body = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(path, options)
  } catch (err) {
    if (err.name === 'AbortError') throw err
    throw new ApiError(0, 'API unreachable')
  }
  const text = await res.text()
  let body = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = null
    }
  }
  if (!res.ok) {
    const detail = body && typeof body.error === 'string' ? body.error : res.statusText || 'request failed'
    throw new ApiError(res.status, `HTTP ${res.status}: ${detail}`, body)
  }
  if (body === null) {
    throw new ApiError(res.status, 'expected JSON from the API')
  }
  return body
}

// GET /api/results -> { count, results: [summary] }
export function listResults({ signal } = {}) {
  return request('/api/results', { signal })
}

// GET /api/results/{id} -> summary + model + labels[] (sorted by logit) + decisions[]
export function getResult(id, { signal } = {}) {
  return request(`/api/results/${encodeURIComponent(id)}`, { signal })
}

// Image URLs go straight into <img src>; the browser fetches them itself.
export function previewUrl(id) {
  return `/api/results/${encodeURIComponent(id)}/preview`
}

// POST /api/results/{id}/decisions -> 201 { id, orthanc_instance_id, decision, comment, reviewer, decided_at }
export function postDecision(id, { decision, reviewer, comment }) {
  return request(`/api/results/${encodeURIComponent(id)}/decisions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision, reviewer, comment }),
  })
}

// GET /api/audit?after=SEQ&limit=N -> { entries: [...], next_after: SEQ | null }
export function listAudit(after, { signal, limit = 50 } = {}) {
  return request(`/api/audit?after=${after}&limit=${limit}`, { signal })
}

// GET /api/audit/verify -> 200 { ok: true, rows, last_hash } or 500 { ok: false, rows, error }.
// A 500 with ok:false IS the answer (chain broken), so return it instead of throwing.
export async function verifyAudit({ signal } = {}) {
  try {
    return await request('/api/audit/verify', { signal })
  } catch (err) {
    if (err instanceof ApiError && err.status === 500 && err.body && err.body.ok === false) return err.body
    throw err
  }
}
