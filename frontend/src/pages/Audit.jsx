import { useEffect, useState } from 'react'
import { listAudit, listResults, verifyAudit } from '../api.js'
import { formatTime } from '../format.js'
import { resultHref } from '../route.js'

const short = (h) => (h ? h.slice(0, 12) : '')

function VerifyBadge({ verify }) {
  if (verify.status === 'loading') return <p className="badge grey-badge">Verifying chain…</p>
  if (verify.status === 'error') return <p className="badge grey-badge">Could not verify: {verify.message}</p>
  const v = verify.data
  if (v.ok)
    return (
      <div className="badge ok-badge">
        <strong>Chain verified</strong>: {v.rows} rows, every hash recomputed by the server and linked.
        <div className="muted">
          Anchor (last hash), keep a copy outside this system to detect truncation:{' '}
          <span className="mono">{v.last_hash}</span>
        </div>
      </div>
    )
  return (
    <div className="badge broken-badge">
      <strong>Chain BROKEN</strong> (checked {v.rows} rows): <span className="mono">{v.error}</span>
    </div>
  )
}

function DetailsCell({ e }) {
  const d = e.details
  if (e.action === 'decision' && d) {
    return (
      <span>
        <span className={`pill ${d.decision}`}>{d.decision}</span> #{d.decision_id}
        {d.comment ? ` — “${d.comment}”` : ''}
      </span>
    )
  }
  return <span className="mono details-json">{d == null ? '' : JSON.stringify(d)}</span>
}

// The "link" column only compares prev_hash with the previous row's hash in the browser.
// The real check is the server's verify, which recomputes every hash from the row contents.
export default function Audit() {
  const [verify, setVerify] = useState({ status: 'loading' })
  const [log, setLog] = useState({ status: 'loading', entries: [], nextAfter: null })
  const [names, setNames] = useState({})
  const [loadingMore, setLoadingMore] = useState(false)

  useEffect(() => {
    const c = new AbortController()
    verifyAudit({ signal: c.signal })
      .then((data) => setVerify({ status: 'ready', data }))
      .catch((err) => {
        if (err.name !== 'AbortError') setVerify({ status: 'error', message: err.message })
      })
    listAudit(0, { signal: c.signal })
      .then((p) => setLog({ status: 'ready', entries: p.entries, nextAfter: p.next_after }))
      .catch((err) => {
        if (err.name !== 'AbortError') setLog({ status: 'error', entries: [], nextAfter: null, message: err.message })
      })
    listResults({ signal: c.signal })
      .then((d) => setNames(Object.fromEntries(d.results.map((r) => [r.orthanc_instance_id, r.image_index]))))
      .catch(() => {
        // names are cosmetic: rows fall back to the instance id
      })
    return () => c.abort()
  }, [])

  async function loadMore() {
    if (loadingMore || log.nextAfter == null) return
    setLoadingMore(true)
    try {
      const p = await listAudit(log.nextAfter)
      setLog((prev) => ({ ...prev, entries: [...prev.entries, ...p.entries], nextAfter: p.next_after, moreError: null }))
    } catch (err) {
      setLog((prev) => ({ ...prev, moreError: err.message }))
    } finally {
      setLoadingMore(false)
    }
  }

  return (
    <section>
      <h1>Audit log</h1>
      <VerifyBadge verify={verify} />
      {log.status === 'loading' && <p>Loading log…</p>}
      {log.status === 'error' && <p className="error">Could not load log: {log.message}</p>}
      {log.status === 'ready' && (
        <>
          <p className="muted">
            Showing {log.entries.length} entries, oldest first. Every result and decision is recorded; nothing
            can be edited or deleted.
          </p>
          <table className="audit">
            <thead>
              <tr>
                <th className="num">Seq</th>
                <th>Time</th>
                <th>Actor</th>
                <th>Action</th>
                <th>Image</th>
                <th>Details</th>
                <th>prev → hash</th>
                <th>Link</th>
              </tr>
            </thead>
            <tbody>
              {log.entries.map((e, i) => {
                const prev = i > 0 ? log.entries[i - 1] : null
                const linked = prev ? e.prev_hash === prev.hash : null
                const id = e.orthanc_instance_id
                return (
                  <tr key={e.seq}>
                    <td className="num">{e.seq}</td>
                    <td title={e.at}>{formatTime(e.at)}</td>
                    <td>{e.actor}</td>
                    <td>{e.action}</td>
                    <td>{id ? <a href={resultHref(id)}>{names[id] ?? short(id)}</a> : '—'}</td>
                    <td>
                      <DetailsCell e={e} />
                    </td>
                    <td className="mono">
                      {short(e.prev_hash)} → {short(e.hash)}
                    </td>
                    <td>
                      {linked === null ? '' : linked ? <span className="link-ok">✓</span> : <span className="link-bad">✗</span>}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          {log.moreError && <p className="error">Could not load more: {log.moreError}</p>}
          {log.nextAfter != null && (
            <button type="button" className="load-more" disabled={loadingMore} onClick={loadMore}>
              {loadingMore ? 'Loading…' : 'Load more'}
            </button>
          )}
        </>
      )}
    </section>
  )
}
