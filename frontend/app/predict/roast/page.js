'use client'

import { useState } from 'react'
import { Camera, Upload } from 'lucide-react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

export default function RoastPage() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImage(file)
      setPreview(URL.createObjectURL(file))
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!image) return

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const reader = new FileReader()
      reader.onload = async () => {
        const base64 = reader.result.split(',')[1]
        const res = await fetch(`${API_URL}/predict-roast`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_base64: base64 }),
        })
        if (!res.ok) throw new Error('Prediction failed')
        const data = await res.json()
        setResult(data)
        setLoading(false)
      }
      reader.readAsDataURL(image)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-stone-50 py-12">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-3 mb-8">
          <Camera className="w-10 h-10 text-amber-700" />
          <h1 className="text-4xl font-bold text-gray-900">Roast Classifier</h1>
        </div>

        <form onSubmit={handleSubmit} className="bg-white p-8 rounded-lg shadow-md mb-8">
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">Upload Bean Image</label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-amber-500 transition">
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-3" />
              <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" id="file-upload" />
              <label htmlFor="file-upload" className="cursor-pointer text-amber-700 font-semibold hover:underline">
                Click to upload an image
              </label>
              <p className="text-gray-500 text-sm mt-1">PNG, JPG up to 10MB</p>
            </div>
          </div>

          {preview && (
            <div className="mb-6">
              <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg shadow" />
            </div>
          )}

          <button type="submit" disabled={loading || !image} className="w-full bg-amber-700 text-white py-3 rounded-lg font-semibold hover:bg-amber-800 transition disabled:opacity-50">
            {loading ? 'Analyzing...' : 'Classify Roast Level'}
          </button>
        </form>

        {error && <p className="text-red-600 mb-4">{error}</p>}

        {result && (
          <div className="bg-white p-8 rounded-lg shadow-md text-center">
            <h2 className="text-2xl font-bold mb-4">Roast Level</h2>
            <p className="text-5xl font-bold text-amber-700 capitalize mb-2">{result.roastLevel}</p>
            <p className="text-gray-600">Confidence: {result.confidence}%</p>
          </div>
        )}
      </div>
    </div>
  )
}