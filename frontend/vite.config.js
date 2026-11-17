import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev only: the browser talks to one origin; /api is forwarded to the Go API.
// The demo build uses nginx for the same job (Step 13f).
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': 'http://127.0.0.1:8080',
    },
  },
})
