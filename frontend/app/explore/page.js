'use client'

import { useState, useEffect } from 'react'
import { MapPin, TrendingUp } from 'lucide-react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

export default function ExplorePage() {
  const [regions, setRegions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const fetchRegions = async () => {
      try {
        const res = await fetch(`${API_URL}/regions`)
        if (!res.ok) throw new Error('Failed to fetch regions')
        const data = await res.json()
        setRegions(data)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    fetchRegions()
  }, [])

  return (
    <div className="min-h-screen bg-stone-50 py-12">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-3 mb-8">
          <MapPin className="w-10 h-10 text-amber-700" />
          <h1 className="text-4xl font-bold text-gray-900">Explore Ethiopian Regions</h1>
        </div>

        <p className="text-gray-600 mb-8">
          Discover coffee quality patterns across Ethiopian growing regions. Data is aggregated from model predictions on regional samples.
        </p>

        {error && <p className="text-red-600 mb-4">{error}</p>}

        {loading ? (
          <p className="text-gray-500">Loading regional data...</p>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {regions.map((region, i) => (
              <div key={i} className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition">
                <h3 className="text-xl font-semibold mb-2">{region.Region || region.region || 'Unknown Region'}</h3>
                <div className="flex items-center gap-2 mb-3">
                  <TrendingUp className="w-5 h-5 text-amber-700" />
                  <span className="text-2xl font-bold text-amber-700">
                    {region.avgScore ? region.avgScore.toFixed(1) : 'N/A'}
                  </span>
                  <span className="text-gray-500">/ 100</span>
                </div>
                <p className="text-gray-600 text-sm">
                  {region.sampleCount ? `${region.sampleCount} samples` : 'Sample count unavailable'}
                </p>
              </div>
            ))}
          </div>
        )}

        {!loading && regions.length === 0 && !error && (
          <div className="bg-white p-8 rounded-lg shadow-md text-center">
            <MapPin className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">No Regional Data Yet</h3>
            <p className="text-gray-600">
              Regional quality data will appear here after the model is trained and Ethiopian coffee samples are processed.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}