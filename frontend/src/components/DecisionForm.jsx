import { useRef, useState } from 'react'
import { postDecision } from '../api.js'

// Reviewer is self-declared (no login): a documented limitation, not authentication.
// Remembering the name in this browser is only a convenience.
const REVIEWER_KEY = 'aidiag.reviewer'

function loadReviewer() {
  try {
    return window.localStorage.getItem(REVIEWER_KEY) ?? ''
  } catch {
    return ''
  }
}

function saveReviewer(name) {
  try {
    window.localStorage.setItem(REVIEWER_KEY, name)
  } catch {
    // storage unavailable: nothing to remember, not an error
  }
}

// Client checks are only for quick, clear messages; the API is the authority and
// any rule it enforces comes back as "Not saved: HTTP 4xx: <API message>".
//
// Double-click protection: the API does not deduplicate, and on localhost a POST
// finishes (~10 ms) long before a double-click's second click (~150 ms). So the form
// stays LOCKED after a successful save until the reviewer explicitly unlocks it.
// `busy` (a ref) blocks clicks in the same instant, before React re-renders.
export default function DecisionForm({ resultId, onDecided }) {
  const [reviewer, setReviewer] = useState(loadReviewer)
  const [comment, setComment] = useState('')
  const [posting, setPosting] = useState(false)
  const [locked, setLocked] = useState(false)
  const [message, setMessage] = useState(null)
  const busy = useRef(false)

  async function submit(decision) {
    if (busy.current) return
    const name = reviewer.trim()
    const text = comment.trim()
    if (!name) {
      setMessage({ kind: 'error', text: 'Enter your name as reviewer.' })
      return
    }
    if (decision === 'reject' && !text) {
      setMessage({ kind: 'error', text: 'A comment is required to reject.' })
      return
    }
    busy.current = true
    setPosting(true)
    setMessage(null)
    try {
      const saved = await postDecision(resultId, { decision, reviewer: name, comment: text })
      saveReviewer(name)
      setComment('')
      setLocked(true)
      setMessage({ kind: 'ok', text: `Saved: ${saved.decision} by ${saved.reviewer} (decision #${saved.id}).` })
      onDecided(saved)
    } catch (err) {
      busy.current = false
      setMessage({ kind: 'error', text: `Not saved: ${err.message}` })
    } finally {
      setPosting(false)
    }
  }

  function unlock() {
    busy.current = false
    setLocked(false)
    setMessage(null)
  }

  const disabled = posting || locked

  return (
    <form className="decision-form" onSubmit={(e) => e.preventDefault()}>
      <h2>Review</h2>
      <p className="muted">
        Approve = AI output reviewed and accepted as a screening aid. Reject = you disagree (say why).
      </p>
      <label>
        Reviewer
        <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} maxLength={100} disabled={disabled} />
      </label>
      <label>
        Comment (required to reject)
        <textarea
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          rows={3}
          maxLength={2000}
          disabled={disabled}
        />
      </label>
      <div className="decision-buttons">
        <button type="button" className="approve" disabled={disabled} onClick={() => submit('approve')}>
          Approve
        </button>
        <button type="button" className="reject" disabled={disabled} onClick={() => submit('reject')}>
          Reject
        </button>
        {posting && <span className="muted">Saving…</span>}
      </div>
      {message && (
        <p className={message.kind === 'error' ? 'error' : 'saved'}>
          {message.text}
          {locked && (
            <button type="button" className="linklike" onClick={unlock}>
              Record another decision
            </button>
          )}
        </p>
      )}
    </form>
  )
}
