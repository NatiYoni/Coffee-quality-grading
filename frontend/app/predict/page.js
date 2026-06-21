'use client'

import { useState } from 'react'
import { Coffee, Plus, Trash2, Loader2, TrendingUp, Info, ChevronDown, ChevronUp } from 'lucide-react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

const EMPTY_COFFEE = () => ({
  id: Math.random().toString(36).slice(2),
  altitude: '', aroma: '', flavor: '', aftertaste: '',
  acidity: '', body: '', balance: '', uniformity: '',
  clean_cup: '', sweetness: '', moisture: '',
  category_one_defects: '0', category_two_defects: '0',
  processing_method: 'Washed', country_of_origin: 'Ethiopia',
})

const FIELDS = [
  { name: 'altitude',             label: 'Altitude (m)',          type: 'number', step: '1'    },
  { name: 'aroma',                label: 'Aroma (0-10)',          type: 'number', step: '0.1'  },
  { name: 'flavor',               label: 'Flavor (0-10)',         type: 'number', step: '0.1'  },
  { name: 'aftertaste',           label: 'Aftertaste (0-10)',     type: 'number', step: '0.1'  },
  { name: 'acidity',              label: 'Acidity (0-10)',        type: 'number', step: '0.1'  },
  { name: 'body',                 label: 'Body (0-10)',           type: 'number', step: '0.1'  },
  { name: 'balance',              label: 'Balance (0-10)',        type: 'number', step: '0.1'  },
  { name: 'uniformity',           label: 'Uniformity (0-10)',     type: 'number', step: '0.1'  },
  { name: 'clean_cup',            label: 'Clean Cup (0-10)',      type: 'number', step: '0.1'  },
  { name: 'sweetness',            label: 'Sweetness (0-10)',      type: 'number', step: '0.1'  },
  { name: 'moisture',             label: 'Moisture (%)',          type: 'number', step: '0.01' },
  { name: 'category_one_defects', label: 'Category One Defects', type: 'number', step: '1'    },
  { name: 'category_two_defects', label: 'Category Two Defects', type: 'number', step: '1'    },
]

function buildPayload(c) {
  return {
    Altitude: parseFloat(c.altitude),
    Aroma: parseFloat(c.aroma),
    Flavor: parseFloat(c.flavor),
    Aftertaste: parseFloat(c.aftertaste),
    Acidity: parseFloat(c.acidity),
    Body: parseFloat(c.body),
    Balance: parseFloat(c.balance),
    Uniformity: parseFloat(c.uniformity),
    'Clean Cup': parseFloat(c.clean_cup),
    Sweetness: parseFloat(c.sweetness),
    'Moisture Percentage': parseFloat(c.moisture),
    'Category One Defects': parseInt(c.category_one_defects),
    'Category Two Defects': parseInt(c.category_two_defects),
    Processing_Method: c.processing_method,
    Country_of_Origin: c.country_of_origin,
  }
}

function CoffeeForm({ coffee, index, total, onChange, onRemove }) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className="bg-white rounded-xl border border-stone-200 shadow-sm overflow-hidden mb-4">
      {/* Card header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-stone-100 bg-amber-50">
        <div className="flex items-center gap-2">
          <Coffee className="w-4 h-4 text-amber-700" />
          <span className="text-sm font-semibold text-stone-700">
            Coffee Sample {index + 1}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button type="button" onClick={() => setCollapsed(!collapsed)}
            className="text-stone-400 hover:text-stone-600 transition">
            {collapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
          </button>
          {total > 1 && (
            <button type="button" onClick={onRemove}
              className="text-red-400 hover:text-red-600 transition">
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Form fields */}
      {!collapsed && (
        <div className="p-5">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {FIELDS.map((f) => (
              <div key={f.name}>
                <label className="block text-xs font-medium text-stone-500 mb-1">{f.label}</label>
                <input
                  type={f.type} step={f.step}
                  value={coffee[f.name]}
                  onChange={(e) => onChange(f.name, e.target.value)}
                  className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm
                             focus:outline-none focus:ring-2 focus:ring-amber-500"
                  required
                />
              </div>
            ))}
            <div>
              <label className="block text-xs font-medium text-stone-500 mb-1">Processing Method</label>
              <select value={coffee.processing_method}
                onChange={(e) => onChange('processing_method', e.target.value)}
                className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm
                           focus:outline-none focus:ring-2 focus:ring-amber-500">
                <option>Washed</option>
                <option>Natural</option>
                <option>Honey</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-stone-500 mb-1">Country of Origin</label>
              <input type="text" value={coffee.country_of_origin}
                onChange={(e) => onChange('country_of_origin', e.target.value)}
                className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm
                           focus:outline-none focus:ring-2 focus:ring-amber-500"
                required />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function ResultCard({ index, result }) {
  const [showAI, setShowAI] = useState(false)
  return (
    <div className="bg-white rounded-xl border border-stone-200 shadow-sm overflow-hidden">
      <div className="px-5 py-3 border-b border-stone-100 bg-amber-50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Coffee className="w-4 h-4 text-amber-700" />
          <span className="text-sm font-semibold text-stone-700">Coffee Sample {index + 1}</span>
        </div>
        <span className={`text-xs font-bold px-3 py-1 rounded-full text-white
          ${result.grade === 'Specialty' ? 'bg-green-600' : 'bg-orange-500'}`}>
          {result.grade}
        </span>
      </div>

      <div className="p-5">
        {/* Score */}
        <div className="text-center mb-5">
          <p className="text-5xl font-bold text-amber-700">{result.score}</p>
          <p className="text-xs text-stone-400 mt-1">Predicted Quality Score</p>
        </div>

        {/* Flavor cluster */}
        {result.flavorCluster && (
          <div className="mb-4 p-3 bg-stone-50 rounded-lg">
            <div className="flex items-center gap-1.5 mb-1">
              <TrendingUp className="w-3.5 h-3.5 text-amber-700" />
              <span className="text-xs font-semibold text-stone-500 uppercase tracking-wide">Flavor Family</span>
            </div>
            <p className="text-sm font-medium text-stone-800">{result.flavorCluster.name}</p>
            <p className="text-xs text-stone-500">{result.flavorCluster.description}</p>
          </div>
        )}

        {/* SHAP bars */}
        <div className="mb-4">
          <p className="text-xs font-semibold text-stone-400 uppercase tracking-wide mb-2">Top Factors</p>
          <div className="space-y-2">
            {result.shap.map((f, i) => (
              <div key={i} className="flex items-center justify-between">
                <span className="text-xs text-stone-600 truncate flex-1 mr-2">{f.feature}</span>
                <span className={`text-xs font-medium ${f.direction === 'positive' ? 'text-green-600' : 'text-red-500'}`}>
                  {f.direction === 'positive' ? '+' : ''}{f.value.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Counterfactual */}
        {result.counterfactual && (
          <div className="mb-4 p-3 bg-blue-50 border border-blue-100 rounded-lg text-xs text-blue-800">
            💡 {result.counterfactual.suggestion}
          </div>
        )}

        {/* AI explanation */}
        <button onClick={() => setShowAI(!showAI)}
          className="flex items-center gap-1 text-xs text-amber-700 font-semibold hover:underline mb-2">
          <Info className="w-3.5 h-3.5" />
          {showAI ? 'Hide' : 'Show'} AI Explanation
        </button>
        {showAI && (
          <div className="bg-amber-50 p-3 rounded-lg border border-amber-100 text-xs text-stone-700 leading-relaxed">
            This coffee scored <strong>{result.score}</strong> points — classified as{' '}
            <strong>{result.grade}</strong> grade.
            {result.flavorCluster && <> It belongs to the <strong>{result.flavorCluster.name}</strong> flavor family ({result.flavorCluster.description}).</>}
            {' '}The top factors driving this score are shown above.
            {result.counterfactual && ` ${result.counterfactual.suggestion}`}
          </div>
        )}
      </div>
    </div>
  )
}

export default function PredictPage() {
  const [coffees, setCoffees] = useState([EMPTY_COFFEE()])
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')

  const addCoffee = () => setCoffees(prev => [...prev, EMPTY_COFFEE()])

  const removeCoffee = (id) => {
    setCoffees(prev => prev.filter(c => c.id !== id))
    setResults([])
  }

  const updateField = (id, field, value) => {
    setCoffees(prev => prev.map(c => c.id === id ? { ...c, [field]: value } : c))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true); setError(''); setResults([])
    try {
      const responses = await Promise.all(
        coffees.map(c =>
          fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: buildPayload(c) }),
          }).then(r => { if (!r.ok) throw new Error('Prediction failed'); return r.json() })
        )
      )
      setResults(responses)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-stone-50">
      <div className="max-w-4xl mx-auto px-4 py-12">

        {/* Header */}
        <div className="mb-8">
          <p className="text-xs font-semibold uppercase tracking-widest text-amber-700 mb-1">Quality Predictor</p>
          <h1 className="text-3xl font-bold text-stone-900">Predict coffee quality</h1>
          <p className="text-stone-500 mt-1 text-sm">
            Enter cupping details for one or more coffee samples — get a predicted quality score for each.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          {/* Coffee forms */}
          {coffees.map((c, i) => (
            <CoffeeForm
              key={c.id}
              coffee={c}
              index={i}
              total={coffees.length}
              onChange={(field, val) => updateField(c.id, field, val)}
              onRemove={() => removeCoffee(c.id)}
            />
          ))}

          {/* Add sample button */}
          <button type="button" onClick={addCoffee}
            className="flex items-center gap-2 text-amber-700 border border-amber-300 bg-amber-50
                       hover:bg-amber-100 px-4 py-2 rounded-lg text-sm font-medium transition mb-6 w-full justify-center">
            <Plus className="w-4 h-4" />
            Add another coffee sample
          </button>

          {/* Submit */}
          <button type="submit" disabled={loading}
            className="w-full bg-amber-700 text-white py-3 rounded-lg font-semibold
                       hover:bg-amber-800 transition disabled:opacity-40 flex items-center justify-center gap-2">
            {loading
              ? <><Loader2 className="w-4 h-4 animate-spin" /> Predicting...</>
              : `Predict ${coffees.length > 1 ? `${coffees.length} samples` : 'quality'}`}
          </button>
        </form>

        {error && (
          <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
            {error}
          </div>
        )}

        {/* Results */}
        {results.length > 0 && (
          <div className="mt-8">
            <p className="text-xs font-semibold uppercase tracking-widest text-stone-400 mb-4">
              Results — {results.length} sample{results.length > 1 ? 's' : ''}
            </p>
            <div className={`grid gap-4 ${results.length > 1 ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1 max-w-md'}`}>
              {results.map((r, i) => (
                <ResultCard key={i} index={i} result={r} />
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  )
}