import { useEffect, useState } from 'react'

// Hash routes: the part after '#' is never sent to the server and changing it
// does not reload the page. '#/results/abc' -> { page: 'detail', id: 'abc' }
export function parseHash(hash) {
  const path = hash.replace(/^#/, '') || '/'
  if (path === '/') return { page: 'worklist' }
  if (path === '/audit') return { page: 'audit' }
  const m = path.match(/^\/results\/([^/]+)$/)
  if (m) return { page: 'detail', id: decodeURIComponent(m[1]) }
  return { page: 'notfound', path }
}

export function useHashRoute() {
  const [route, setRoute] = useState(() => parseHash(window.location.hash))
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash))
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}

export function resultHref(id) {
  return `#/results/${encodeURIComponent(id)}`
}
