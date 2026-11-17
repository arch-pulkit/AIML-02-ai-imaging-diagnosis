import { useEffect, useState } from 'react'
import { getResult, previewUrl } from '../api.js'
import ImageOverlay from '../components/ImageOverlay.jsx'
import LabelsTable from '../components/LabelsTable.jsx'
import { formatTime } from '../format.js'

export default function Detail({ id }) {
  const [state, setState] = useState({ status: 'loading' })
  const [selected, setSelected] = useState(null)
  const [opacity, setOpacity] = useState(0.5)

  useEffect(() => {
    const controller = new AbortController()
    getResult(id, { signal: controller.signal })
      .then((data) => setState({ status: 'ready', data }))
      .catch((err) => {
        if (err.name === 'AbortError') return
        setState({ status: 'error', message: err.message })
      })
    return () => controller.abort()
  }, [id])

  if (state.status === 'loading') return <p>Loading result…</p>
  if (state.status === 'error')
    return (
      <section>
        <p><a href="#/">← Worklist</a></p>
        <p className="error">Could not load result: {state.message}</p>
      </section>
    )

  const r = state.data
  const labels = r.labels
  // Derived, not stored: the user's choice, else the top label that has a heatmap.
  const shownLabel = selected ?? labels.find((l) => l.heatmap_url)?.label ?? null
  const shown = labels.find((l) => l.label === shownLabel)

  return (
    <section>
      <p><a href="#/">← Worklist</a></p>
      <h1>{r.image_index ?? r.orthanc_instance_id}</h1>
      <p className="muted">
        Patient {r.patient_id ?? '—'} · <span className={`pill ${r.status}`}>{r.status}</span> · received{' '}
        <span title={r.received_at}>{formatTime(r.received_at)}</span> · model {r.model.name} (
        <span className="mono">{r.model.checkpoint_sha256?.slice(0, 12)}</span>)
      </p>

      {r.status === 'error' ? (
        <p className="error">Analysis failed: {r.error}. No model output is available for this instance.</p>
      ) : (
        <div className="detail-grid">
          <div>
            <ImageOverlay
              previewUrl={previewUrl(r.orthanc_instance_id)}
              heatmapUrl={shown?.heatmap_url ?? null}
              opacity={opacity}
            />
            <div className="overlay-controls">
              <span>
                Overlay: <strong>{shownLabel ?? 'none'}</strong>
              </span>
              <label>
                Opacity{' '}
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                />{' '}
                {Math.round(opacity * 100)}%
              </label>
            </div>
            <div className="legend">
              <span>low attention</span>
              <span className="legend-bar" />
              <span>high</span>
            </div>
          </div>
          <div>
            <p className="note">
              Screening flags, not diagnoses. A heatmap shows where the model looked for that label, not where
              disease is.
            </p>
            <LabelsTable labels={labels} selected={shownLabel} onSelect={setSelected} />
          </div>
        </div>
      )}

      <h2>Decisions</h2>
      {r.decisions.length === 0 ? (
        <p className="muted">No decisions yet.</p>
      ) : (
        <ol className="decisions">
          {r.decisions.map((d, i) => (
            <li key={`${d.decided_at}-${i}`}>
              <span className={`pill ${d.decision}`}>{d.decision}</span> by <strong>{d.reviewer}</strong> at{' '}
              <span title={d.decided_at}>{formatTime(d.decided_at)}</span>
              {d.comment && <> — “{d.comment}”</>}
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}
