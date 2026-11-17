import { useEffect, useState } from 'react'
import { listResults } from '../api.js'
import { resultHref } from '../route.js'
import { formatTime } from '../format.js'

function DecisionCell({ decision }) {
  if (!decision) return <span className="pill pending">pending</span>
  return (
    <span className={`pill ${decision.decision}`} title={`${decision.reviewer} at ${decision.decided_at}`}>
      {decision.decision} · {decision.reviewer}
    </span>
  )
}

function WorklistRow({ r }) {
  const href = resultHref(r.orthanc_instance_id)
  const open = () => {
    window.location.hash = href
  }
  if (r.status === 'error') {
    return (
      <tr className="row-error" onClick={open}>
        <td title={r.received_at}>{formatTime(r.received_at)}</td>
        <td colSpan={2}>
          <a href={href} className="mono">{r.orthanc_instance_id}</a>
        </td>
        <td><span className="pill error">error</span></td>
        <td colSpan={2} className="error">{r.error}</td>
      </tr>
    )
  }
  const flagged = r.flagged ?? []
  return (
    <tr onClick={open}>
      <td title={r.received_at}>{formatTime(r.received_at)}</td>
      <td><a href={href}>{r.image_index}</a></td>
      <td>{r.patient_id}</td>
      <td><span className="pill ok">ok</span></td>
      <td className="num" title={flagged.join(', ') || 'none'}>{flagged.length}</td>
      <td><DecisionCell decision={r.latest_decision} /></td>
    </tr>
  )
}

export default function Worklist() {
  const [state, setState] = useState({ status: 'loading' })

  useEffect(() => {
    const controller = new AbortController()
    listResults({ signal: controller.signal })
      .then((data) => setState({ status: 'ready', data }))
      .catch((err) => {
        if (err.name === 'AbortError') return
        setState({ status: 'error', message: err.message })
      })
    return () => controller.abort()
  }, [])

  if (state.status === 'loading') return <p>Loading results…</p>
  if (state.status === 'error') return <p className="error">Could not load results: {state.message}</p>

  const { count, results } = state.data
  const pending = results.filter((r) => r.status === 'ok' && !r.latest_decision).length

  return (
    <section>
      <h1>Worklist</h1>
      <p className="muted">{count} results · {pending} awaiting review</p>
      <table className="worklist">
        <thead>
          <tr>
            <th>Received</th>
            <th>Image</th>
            <th>Patient</th>
            <th>Status</th>
            <th className="num">Flags</th>
            <th>Latest decision</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <WorklistRow key={r.orthanc_instance_id} r={r} />
          ))}
        </tbody>
      </table>
    </section>
  )
}
