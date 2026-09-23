'use client'

import { useEffect, useId, useRef, useState } from 'react'
import { Info, Loader2 } from 'lucide-react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

function validExplanation(data) {
  if (data?.method !== 'validated_evidence_plan' || data.provider !== 'gemini' ||
      typeof data.model !== 'string' || !Number.isFinite(data.evidence?.score) ||
      !Array.isArray(data.evidence?.drivers) || !data.evidence.drivers.length ||
      !Array.isArray(data.sentences) || data.sentences.length < 2 ||
      !Array.isArray(data.notes) || !data.notes.every(note => typeof note === 'string')) return false
  const ids = new Set(['prediction'])
  for (const driver of data.evidence.drivers) {
    if (typeof driver.feature !== 'string' || !Number.isFinite(driver.value) ||
        driver.direction !== (driver.value > 0 ? 'positive' : driver.value < 0 ? 'negative' : 'neutral')) return false
    ids.add(`shap:${driver.feature}`)
  }
  const sentenceIds = new Set(data.sentences.map(sentence => sentence?.evidence_id))
  return sentenceIds.size === data.sentences.length && sentenceIds.has('prediction') &&
    data.sentences.every(sentence => typeof sentence?.text === 'string' && ids.has(sentence.evidence_id))
}

function matchesPrediction(result, evidence) {
  return evidence?.score === result.score && evidence?.grade === result.grade &&
    evidence.drivers?.length === result.shap.length &&
    result.shap.every(item =>
      evidence.drivers.some(driver =>
        driver.feature === item.feature && driver.value === item.value
      )
    )
}

export default function Explanation({ result, features }) {
  const [explanation, setExplanation] = useState(null)
  const [expanded, setExpanded] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const request = useRef(null)
  const panelId = useId()

  useEffect(() => () => request.current?.abort(), [])

  async function explain() {
    if (explanation) {
      setExpanded(value => !value)
      return
    }
    const controller = new AbortController()
    request.current = controller
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_URL}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features }),
        signal: controller.signal,
      })
      const data = await response.json()
      if (!response.ok) {
        const message = data?.detail?.message || data?.error
        throw new Error(typeof message === 'string' ? message : 'Explanation unavailable. Try again.')
      }
      if (!validExplanation(data)) {
        throw new Error('The explanation response was invalid. Try again.')
      }
      if (!matchesPrediction(result, data.evidence)) {
        throw new Error('The prediction changed. Predict the sample again before requesting an explanation.')
      }
      setExplanation(data)
      setExpanded(true)
    } catch (err) {
      if (!controller.signal.aborted) {
        setError(err instanceof SyntaxError
          ? 'The explanation service returned an unreadable response. Try again.'
          : err.message || 'Could not reach the explanation service. Try again.')
      }
    } finally {
      if (!controller.signal.aborted) setLoading(false)
    }
  }

  return (
    <section className="border-t border-stone-200 pt-4" aria-label="Prediction explanation">
      <button
        type="button"
        onClick={explain}
        disabled={loading}
        aria-expanded={expanded}
        aria-controls={panelId}
        className="flex items-center gap-2 rounded-md text-sm text-amber-800 font-semibold
                   hover:underline focus-visible:outline focus-visible:outline-2
                   focus-visible:outline-offset-4 focus-visible:outline-amber-700 disabled:opacity-60"
      >
        {loading
          ? <Loader2 className="w-4 h-4 motion-safe:animate-spin" aria-hidden="true" />
          : <Info className="w-4 h-4" aria-hidden="true" />}
        {loading ? 'Checking explanation…' : explanation
          ? `${expanded ? 'Hide' : 'Show'} explanation`
          : error ? 'Try explanation again' : 'Explain this prediction'}
      </button>
      {!explanation && (
        <p className="mt-2 text-xs leading-relaxed text-stone-600">
          Sends this prediction and its feature attributions to Google Gemini.
          Use public or synthetic samples only; Google may use free-tier data to improve its products.
        </p>
      )}
      {loading && <p role="status" className="mt-2 text-xs text-stone-600">Your quality prediction is unchanged.</p>}
      {error && <p role="alert" className="mt-3 text-sm text-red-700">{error}</p>}
      <div id={panelId} hidden={!expanded || !explanation}>
        {explanation && (
          <div className="mt-4 space-y-3 text-sm leading-relaxed text-stone-800">
            {explanation.sentences.map((sentence, index) => (
              <p key={sentence.evidence_id}>
                {sentence.text}{' '}
                <a
                  href={`#${panelId}-evidence-${index}`}
                  aria-label={`Supporting evidence for statement ${index + 1}`}
                  className="text-amber-800 underline underline-offset-2 focus-visible:outline"
                >[{index + 1}]</a>
              </p>
            ))}
            <div className="rounded-lg bg-stone-50 p-3">
              <h3 className="font-semibold text-stone-800">Supporting evidence</h3>
              <ol className="mt-2 space-y-1 text-xs text-stone-700">
                {explanation.sentences.map((sentence, index) => {
                  const driver = explanation.evidence.drivers.find(
                    item => `shap:${item.feature}` === sentence.evidence_id
                  )
                  return (
                    <li key={sentence.evidence_id} id={`${panelId}-evidence-${index}`}>
                      [{index + 1}] {sentence.evidence_id === 'prediction'
                        ? `Model score: ${explanation.evidence.score}; grade: ${explanation.evidence.grade}`
                        : `${driver.feature}: SHAP ${driver.value.toPrecision(6)} (${driver.direction})`}
                    </li>
                  )
                })}
              </ol>
            </div>
            <p className="text-xs text-stone-600">
              Gemini selected and ordered the points; the server checked the evidence and rendered the wording.
              {' '}Model: {explanation.model}.
            </p>
            {explanation.notes.map(note => <p key={note} className="text-xs text-stone-600">{note}</p>)}
          </div>
        )}
      </div>
    </section>
  )
}
