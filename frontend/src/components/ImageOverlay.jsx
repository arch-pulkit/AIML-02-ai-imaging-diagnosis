import { useEffect, useRef, useState } from 'react'
import { JET } from '../colormap.js'

// X-ray and heatmap share one square box. At model input the WHOLE image was resized
// to 224x224 (no crop), so the 224x224 heatmap is stretched over the whole box.
// The canvas keeps 224x224 pixels; CSS scales it up (smoothly) to the box size.
// Alpha = heat value: low attention stays transparent, the slider scales the rest.
export default function ImageOverlay({ previewUrl, heatmapUrl, opacity }) {
  const canvasRef = useRef(null)
  const [previewFailed, setPreviewFailed] = useState(false)
  const [failedHeatmap, setFailedHeatmap] = useState(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (!heatmapUrl) return undefined
    let cancelled = false
    const img = new Image()
    img.onload = () => {
      if (cancelled) return
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      ctx.drawImage(img, 0, 0)
      const frame = ctx.getImageData(0, 0, canvas.width, canvas.height)
      const px = frame.data
      for (let i = 0; i < px.length; i += 4) {
        const v = px[i]
        const [r, g, b] = JET[v]
        px[i] = r
        px[i + 1] = g
        px[i + 2] = b
        px[i + 3] = v
      }
      ctx.putImageData(frame, 0, 0)
    }
    img.onerror = () => {
      if (!cancelled) setFailedHeatmap(heatmapUrl)
    }
    img.src = heatmapUrl
    return () => {
      cancelled = true
    }
  }, [heatmapUrl])

  return (
    <div className="overlay-box">
      {previewFailed ? (
        <div className="overlay-msg">X-ray preview unavailable</div>
      ) : (
        <img src={previewUrl} alt="Chest X-ray" onError={() => setPreviewFailed(true)} />
      )}
      <canvas ref={canvasRef} width={224} height={224} style={{ opacity }} />
      {heatmapUrl && failedHeatmap === heatmapUrl && <div className="overlay-msg">Heatmap failed to load</div>}
    </div>
  )
}
