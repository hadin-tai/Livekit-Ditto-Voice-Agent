import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import VoiceRoom from './components/VoiceRoom'
import DocumentManager from './components/DocumentManager'
import './styles/global.css'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<VoiceRoom />} />
        <Route path="/documents" element={<DocumentManager />} />
      </Routes>
    </Router>
  )
}

export default App
