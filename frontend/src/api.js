// All HTTP calls to the Go API live here; pages never call fetch directly.
// Paths are relative ("/api/..."): in dev Vite forwards them, in the demo nginx does,
// so the same build works in both without knowing where the API runs.

export class ApiError extends Error {
  constructor(status, message) {
    super(message)
    this.name = 'ApiError'
    this.status = status
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
    throw new ApiError(res.status, `HTTP ${res.status}: ${detail}`)
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
