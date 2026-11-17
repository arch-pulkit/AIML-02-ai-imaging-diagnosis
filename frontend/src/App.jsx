import Banner from './components/Banner.jsx'
import Worklist from './pages/Worklist.jsx'
import { useHashRoute } from './route.js'
import './App.css'

function Placeholder({ title }) {
  return (
    <section>
      <h1>{title}</h1>
      <p className="muted">Coming in a later step.</p>
    </section>
  )
}

export default function App() {
  const route = useHashRoute()

  let page
  if (route.page === 'worklist') page = <Worklist />
  else if (route.page === 'detail') page = <Placeholder title={`Result ${route.id}`} />
  else if (route.page === 'audit') page = <Placeholder title="Audit log" />
  else
    page = (
      <section>
        <h1>Page not found</h1>
        <p><a href="#/">Back to worklist</a></p>
      </section>
    )

  return (
    <>
      <Banner />
      <header className="topbar">
        <a href="#/" className="brand">AI Diag Viewer</a>
        <nav>
          <a href="#/">Worklist</a>
          <a href="#/audit">Audit log</a>
        </nav>
      </header>
      <main>{page}</main>
    </>
  )
}
