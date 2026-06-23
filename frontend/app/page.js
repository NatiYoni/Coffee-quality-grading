import Link from 'next/link'
import { Coffee, MapPin, BarChart3, Camera, GitCompare, Sparkles } from 'lucide-react'

export default function Home() {
  return (
    <div className="min-h-screen bg-stone-50">
      {/* Hero Section */}
      <section className="bg-gradient-to-b from-amber-700 to-amber-800 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <Coffee className="w-20 h-20 mx-auto mb-6" />
            <h1 className="text-5xl font-bold mb-4">
              Ethiopian Coffee Quality Predictor
            </h1>
            <p className="text-xl text-amber-100 max-w-3xl mx-auto">
              Predict coffee quality scores using machine learning, explore flavor
              profiles, and discover what makes Ethiopian coffee exceptional.
            </p>
            <div className="mt-8 flex gap-4 justify-center">
              <Link
                href="/predict"
                className="bg-white text-amber-700 px-6 py-3 rounded-lg font-semibold hover:bg-amber-50 transition-colors"
              >
                Start Predicting
              </Link>
              <Link
                href="/explore"
                className="border-2 border-white text-white px-6 py-3 rounded-lg font-semibold hover:bg-amber-700 transition-colors"
              >
                Explore Regions
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12 text-gray-900">
            How It Works
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white p-6 rounded-lg shadow-md">
              <BarChart3 className="w-12 h-12 text-amber-700 mb-4" />
              <h3 className="text-xl font-semibold mb-2">1. Input Data</h3>
              <p className="text-gray-600">
                Enter coffee attributes like altitude, aroma, flavor, and processing
                method, or upload a photo for roast/defect analysis.
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-md">
              <Sparkles className="w-12 h-12 text-amber-700 mb-4" />
              <h3 className="text-xl font-semibold mb-2">2. ML Prediction</h3>
              <p className="text-gray-600">
                Our XGBoost model predicts quality scores and explains which factors
                matter most using SHAP values.
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-md">
              <MapPin className="w-12 h-12 text-amber-700 mb-4" />
              <h3 className="text-xl font-semibold mb-2">3. Get Insights</h3>
              <p className="text-gray-600">
                Receive a quality grade, flavor family, improvement suggestions, and
                regional comparisons.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12 text-gray-900">
            Features
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Link href="/predict" className="block p-6 border border-gray-200 rounded-lg hover:shadow-lg transition-shadow contain-content">
              <BarChart3 className="w-10 h-10 text-amber-700 mb-3" />
              <h3 className="text-lg font-semibold mb-2">Quality Predictor</h3>
              <p className="text-gray-600">Predict coffee quality scores from sensory and origin data.</p>
            </Link>
            <Link href="/predict/roast" className="block p-6 border border-gray-200 rounded-lg hover:shadow-lg transition-shadow contain-content">
              <Camera className="w-10 h-10 text-amber-700 mb-3" />
              <h3 className="text-lg font-semibold mb-2">Roast Classifier</h3>
              <p className="text-gray-600">Upload a bean photo to identify roast level (green, light, medium, dark).</p>
            </Link>
            <Link href="/predict/defect" className="block p-6 border border-gray-200 rounded-lg hover:shadow-lg transition-shadow contain-content">
              <Camera className="w-10 h-10 text-amber-700 mb-3" />
              <h3 className="text-lg font-semibold mb-2">Defect Detector</h3>
              <p className="text-gray-600">Detect bean defects using computer vision and Roboflow.</p>
            </Link>
            <Link href="/compare" className="block p-6 border border-gray-200 rounded-lg hover:shadow-lg transition-shadow contain-content">
              <GitCompare className="w-10 h-10 text-amber-700 mb-3" />
              <h3 className="text-lg font-semibold mb-2">Compare Coffees</h3>
              <p className="text-gray-600">Side-by-side comparison of two coffee samples with key differences.</p>
            </Link>
            <Link href="/explore" className="block p-6 border border-gray-200 rounded-lg hover:shadow-lg transition-shadow contain-content">
              <MapPin className="w-10 h-10 text-amber-700 mb-3" />
              <h3 className="text-lg font-semibold mb-2">Explore Regions</h3>
              <p className="text-gray-600">Interactive map of Ethiopian coffee regions with quality rankings.</p>
            </Link>
            <div className="p-6 border border-gray-200 rounded-lg">
              <Sparkles className="w-10 h-10 text-amber-700 mb-3" />
              <h3 className="text-lg font-semibold mb-2">AI Explanations</h3>
              <p className="text-gray-600">Get plain-language explanations of predictions, no data science background needed.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Audience */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12 text-gray-900">
            Built For
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-24 h-24 bg-amber-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Coffee className="w-12 h-12 text-amber-700" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Farmers</h3>
              <p className="text-gray-600">Understand how your processing and growing conditions affect quality.</p>
            </div>
            <div className="text-center">
              <div className="w-24 h-24 bg-amber-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <BarChart3 className="w-12 h-12 text-amber-700" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Cooperatives</h3>
              <p className="text-gray-600">Compare batches and identify top-performing regions and methods.</p>
            </div>
            <div className="text-center">
              <div className="w-24 h-24 bg-amber-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <MapPin className="w-12 h-12 text-amber-700" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Buyers</h3>
              <p className="text-gray-600">Make informed decisions with data-driven quality assessments.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p>Ethiopian Coffee Quality Predictor — ML School Project</p>
        </div>
      </footer>
    </div>
  )
}