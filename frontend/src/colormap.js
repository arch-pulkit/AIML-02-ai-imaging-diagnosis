// 256-entry "jet" colour table: 0 = dark blue, ~0.5 = green/yellow, 1 = dark red.
// Standard piecewise-linear jet formula (same shape as matplotlib's jet).
function clamp01(x) {
  return Math.min(1, Math.max(0, x))
}

export const JET = Array.from({ length: 256 }, (_, i) => {
  const v = i / 255
  const r = clamp01(1.5 - Math.abs(4 * v - 3))
  const g = clamp01(1.5 - Math.abs(4 * v - 2))
  const b = clamp01(1.5 - Math.abs(4 * v - 1))
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
})
