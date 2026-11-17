// Stored times are UTC (...Z). Show local time; keep the exact UTC value in a tooltip.
export function formatTime(iso) {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}
