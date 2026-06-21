'use client'

import { useState, useCallback } from 'react'
import { Upload, GitCompare, Loader2, AlertTriangle, CheckCircle, RotateCcw } from 'lucide-react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

const toBase64 = (file) => new Promise((res, rej) => {
  const r = new FileReader()
  r.onload  = () => res(r.result.split(',')[1])
  r.onerror = rej
  r.readAsDataURL(file)
})

const ROAST_META = {
  dark:   { label: 'Dark',   bg: '#292524', text: '#fff', desc: 'Bold and smoky. Low acidity, heavy body.' },
  medium: { label: 'Medium', bg: '#92400e', text: '#fff', desc: 'Balanced flavour, aroma, and acidity.' },
  light:  { label: 'Light',  bg: '#d97706', text: '#fff', desc: 'Bright acidity, complex origin character.' },
  green:  { label: 'Green',  bg: '#15803d', text: '#fff', desc: 'Unroasted — not yet ready for brewing.' },
}

async function analyzeImage(file) {
  const b64 = await toBase64(file)
  const [roastRes, defectRes] = await Promise.all([
    fetch(`${API_URL}/predict-roast`,  { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ image_base64: b64 }) }),
    fetch(`${API_URL}/predict-defect`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ image_base64: b64 }) }),
  ])
  if (!roastRes.ok || !defectRes.ok) throw new Error('Analysis failed.')
  const [roast, defect] = await Promise.all([roastRes.json(), defectRes.json()])
  return { roast, defect }
}

function DropZone({ id, label, preview, onChange, disabled }) {
  const [dragging, setDragging] = useState(false)
  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith('image/')) onChange(f)
  }, [onChange])

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onClick={() => !disabled && document.getElementById(id).click()}
      className={`relative rounded-xl border-2 border-dashed transition cursor-pointer overflow-hidden
        ${dragging ? 'border-amber-500 bg-amber-50' : 'border-stone-300 hover:border-amber-400'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      style={{ minHeight: 200 }}
    >
      <input id={id} type="file" accept="image/*" className="hidden"
        onChange={(e) => e.target.files[0] && onChange(e.target.files[0])} />
      {preview ? (
        <img src={preview} alt={label} className="w-full h-56 object-cover" />
      ) : (
        <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
          <Upload className="w-8 h-8 text-stone-400 mb-2" />
          <p className="text-sm font-medium text-stone-600">{label}</p>
          <p className="text-xs text-stone-400 mt-1">Click or drag & drop</p>
        </div>
      )}
    </div>
  )
}

function ResultCard({ label, data, accent }) {
  if (!data) return null
  const { roast, defect } = data
  const m = ROAST_META[roast.roastLevel?.toLowerCase()] || { label: roast.roastLevel, bg: '#92400e', text: '#fff', desc: '' }
  return (
    <div className="bg-white rounded-xl border border-stone-200 shadow-sm overflow-hidden">
      <div className="px-5 py-3 border-b border-stone-100 flex items-center gap-2" style={{ background: accent + '18' }}>
        <div className="w-2 h-2 rounded-full" style={{ background: accent }} />
        <span className="text-sm font-semibold text-stone-700">{label}</span>
      </div>
      <div className="p-5 space-y-5">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-stone-400 mb-2">Roast Level</p>
          <span className="inline-block px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider mb-1"
            style={{ background: m.bg, color: m.text }}>{m.label}</span>
          <p className="text-xs text-stone-500 mb-1">{m.desc}</p>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-1.5 bg-stone-100 rounded-full overflow-hidden">
              <div className="h-full bg-amber-600 rounded-full" style={{ width: `${roast.confidence}%` }} />
            </div>
            <span className="text-xs text-stone-500">{roast.confidence}%</span>
          </div>
        </div>
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-stone-400 mb-2">Defect Check</p>
          {!defect.isDefective ? (
            <div className="flex items-center gap-2 text-green-700">
              <CheckCircle className="w-4 h-4" />
              <span className="text-sm font-medium">No defects found</span>
            </div>
          ) : (
            <div className="space-y-1.5">
              {(defect.defects ?? []).map((d, i) => (
                <div key={i} className="flex justify-between items-center bg-red-50 border border-red-100 rounded-lg px-3 py-1.5">
                  <span className="text-sm capitalize text-stone-700">{d.class}</span>
                  <span className="text-xs text-red-600 font-medium">{d.confidence}%</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function ComparisonSummary({ dataA, dataB }) {
  const roastA = dataA.roast.roastLevel?.toLowerCase()
  const roastB = dataB.roast.roastLevel?.toLowerCase()
  const defA   = dataA.defect.isDefective
  const defB   = dataB.defect.isDefective
  const winner = !defA && defB ? 'Sample A' : !defB && defA ? 'Sample B' : null

  return (
    <div className="bg-amber-50 border border-amber-200 rounded-xl p-5">
      <p className="text-xs font-semibold uppercase tracking-widest text-amber-700 mb-3">Comparison Summary</p>
      <ul className="space-y-2 text-sm text-stone-700">
        <li><strong>Roast:</strong>{' '}
          {roastA === roastB
            ? `Both samples are ${roastA} roast`
            : `Sample A is ${roastA} roast, Sample B is ${roastB} roast`}
        </li>
        <li><strong>Defects:</strong>{' '}
          {!defA && !defB ? 'Neither sample shows visible defects — both look clean.'
            : defA && defB ? 'Both samples have defects detected.'
            : `${defA ? 'Sample A' : 'Sample B'} has defects; ${defA ? 'Sample B' : 'Sample A'} appears clean.`}
        </li>
        {winner && (
          <li className="pt-2 border-t border-amber-200">
            <strong>Overall:</strong> <span className="text-amber-800 font-medium">{winner}</span> looks stronger — no visible defects detected.
          </li>
        )}
      </ul>
    </div>
  )
}

export default function ComparePage() {
  const [fileA, setFileA]       = useState(null)
  const [previewA, setPreviewA] = useState(null)
  const [resultA, setResultA]   = useState(null)
  const [fileB, setFileB]       = useState(null)
  const [previewB, setPreviewB] = useState(null)
  const [resultB, setResultB]   = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')

  const handleFileA = (f) => { setFileA(f); setPreviewA(URL.createObjectURL(f)); setResultA(null) }
  const handleFileB = (f) => { setFileB(f); setPreviewB(URL.createObjectURL(f)); setResultB(null) }

  const reset = () => {
    setFileA(null); setPreviewA(null); setResultA(null)
    setFileB(null); setPreviewB(null); setResultB(null)
    setError('')
  }

  const handleAnalyze = async () => {
    if (!fileA || !fileB) return
    setLoading(true); setError(''); setResultA(null); setResultB(null)
    try {
      const [rA, rB] = await Promise.all([analyzeImage(fileA), analyzeImage(fileB)])
      setResultA(rA); setResultB(rB)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-stone-50">
      <div className="max-w-4xl mx-auto px-4 py-12">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <GitCompare className="w-8 h-8 text-amber-700" />
            <h1 className="text-3xl font-bold text-stone-900">Compare Bean Images</h1>
          </div>
          <p className="text-stone-500 text-sm">
            Upload two bean photos — we'll compare their roast level and defects side by side.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div>
            <p className="text-xs font-semibold text-stone-500 mb-1.5 uppercase tracking-wide">Sample A</p>
            <DropZone id="dz-a" label="Upload Sample A" preview={previewA} onChange={handleFileA} disabled={loading} />
          </div>
          <div>
            <p className="text-xs font-semibold text-stone-500 mb-1.5 uppercase tracking-wide">Sample B</p>
            <DropZone id="dz-b" label="Upload Sample B" preview={previewB} onChange={handleFileB} disabled={loading} />
          </div>
        </div>

        <div className="flex gap-3 mb-8">
          <button onClick={handleAnalyze} disabled={!fileA || !fileB || loading}
            className="flex items-center gap-2 bg-amber-700 text-white px-6 py-2.5 rounded-lg
                       font-semibold text-sm hover:bg-amber-800 transition disabled:opacity-40">
            {loading ? <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing...</> : 'Compare Images'}
          </button>
          {(previewA || previewB) && (
            <button onClick={reset}
              className="flex items-center gap-1.5 text-stone-500 text-sm px-3 py-2.5 rounded-lg border border-stone-300 hover:border-stone-400 transition">
              <RotateCcw className="w-3.5 h-3.5" /> Reset
            </button>
          )}
        </div>

        {error && (
          <div className="flex items-center gap-2 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6 text-sm">
            <AlertTriangle className="w-4 h-4 flex-shrink-0" />{error}
          </div>
        )}

        {resultA && resultB && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <ResultCard label="Sample A" data={resultA} accent="#92400e" />
              <ResultCard label="Sample B" data={resultB} accent="#0c447c" />
            </div>
            <ComparisonSummary dataA={resultA} dataB={resultB} />
          </div>
        )}
      </div>
    </div>
  )
}
