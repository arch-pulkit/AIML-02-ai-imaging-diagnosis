// 14 labels as sent by the API (sorted by logit, highest first).
// A label is flagged when logit >= its threshold (per-label Youden threshold on val).
export default function LabelsTable({ labels, selected, onSelect }) {
  return (
    <table className="labels">
      <thead>
        <tr>
          <th>Label</th>
          <th className="num">Logit</th>
          <th className="num">Threshold</th>
          <th>Flag</th>
          <th>Heatmap</th>
        </tr>
      </thead>
      <tbody>
        {labels.map((l) => {
          const clickable = l.heatmap_url !== null
          const cls = [l.flagged ? 'flagged' : '', l.label === selected ? 'selected' : '', clickable ? 'clickable' : '']
            .join(' ')
            .trim()
          return (
            <tr key={l.label} className={cls} onClick={clickable ? () => onSelect(l.label) : undefined}>
              <td>{l.label}</td>
              <td className="num">{l.logit.toFixed(2)}</td>
              <td className="num">{l.threshold.toFixed(2)}</td>
              <td>{l.flagged ? <span className="pill flag">flagged</span> : ''}</td>
              <td>{clickable ? (l.label === selected ? 'shown' : 'show') : '—'}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
