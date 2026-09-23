const { test, expect } = require('@playwright/test')

const prediction = {
  score: 84.2,
  grade: 'Specialty',
  shap: [
    { feature: 'Flavor', value: 0.9, direction: 'positive' },
    { feature: 'Balance', value: 0.6, direction: 'positive' },
    { feature: 'Altitude', value: -0.3, direction: 'negative' },
    { feature: 'Aroma', value: 0.1, direction: 'positive' },
  ],
  flavorCluster: { id: 2, name: 'Balanced & Sweet', description: 'High balance and sweetness' },
  counterfactual: null,
}

const explanation = {
  provider: 'gemini',
  model: 'test-model',
  method: 'validated_evidence_plan',
  evidence: { score: prediction.score, grade: prediction.grade, drivers: prediction.shap },
  sentences: [
    { text: "The model predicts 84.2 points and assigns a 'Specialty' label.", evidence_id: 'prediction' },
    { text: 'Flavor made a positive contribution of 0.9 points relative to the model baseline.', evidence_id: 'shap:Flavor' },
    { text: 'Altitude shifted this prediction downward by 0.3 points relative to the model baseline.', evidence_id: 'shap:Altitude' },
  ],
  notes: ['Model attributions do not establish the effects of farming changes.'],
}

async function jsonResponse(route, status, body) {
  await route.fulfill({
    status, contentType: 'application/json', body: JSON.stringify(body),
    headers: { 'Access-Control-Allow-Origin': '*' },
  })
}

async function predict(page) {
  const values = ['1500', '8.2', '8.1', '8', '8', '7.9', '8', '10', '10', '10', '11.5', '0', '0']
  const fields = page.locator('input[type="number"]')
  await expect(fields).toHaveCount(13)
  for (let i = 0; i < values.length; i++) await fields.nth(i).fill(values[i])
  await page.getByRole('button', { name: 'Predict quality', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Explain this prediction' })).toBeVisible()
}

test.beforeEach(async ({ page }) => {
  await page.route('**/api/predict', route => jsonResponse(route, 200, prediction))
  await page.goto('/predict')
})

test('explanations are opt-in, numeric-only, evidence-linked and cached per result', async ({ page }, testInfo) => {
  const requests = []
  await page.route('**/api/explain', async route => {
    requests.push(route.request().postDataJSON())
    await jsonResponse(route, 200, explanation)
  })
  await predict(page)
  expect(requests).toHaveLength(0)
  await expect(page.getByText(/Use public or synthetic samples only/)).toBeVisible()
  await page.getByRole('button', { name: 'Explain this prediction' }).click()
  await expect(page.getByRole('heading', { name: 'Supporting evidence' })).toBeVisible()
  await expect(page.getByText(explanation.sentences[1].text)).toBeVisible()
  await expect(page.getByRole('link', { name: 'Supporting evidence for statement 2' })).toBeVisible()
  expect(Object.keys(requests[0].features)).toHaveLength(13)
  expect(Object.values(requests[0].features).every(value => typeof value === 'number')).toBe(true)
  await page.getByRole('button', { name: 'Hide explanation' }).click()
  await page.getByRole('button', { name: 'Show explanation' }).click()
  expect(requests).toHaveLength(1)
  await page.getByRole('region', { name: 'Prediction explanation' }).screenshot({
    path: testInfo.outputPath('explanation-desktop.png'),
  })
})

for (const [status, code, message] of [
  [429, 'quota_exceeded', 'Gemini quota reached. Wait before trying again.'],
  [503, 'not_configured', 'Explanations are not configured.'],
  [502, 'reversed_direction', 'The generated explanation failed evidence checks.'],
]) {
  test(`${code} keeps the prediction visible and permits an explicit retry`, async ({ page }) => {
    let requests = 0
    await page.route('**/api/explain', route => {
      requests++
      return jsonResponse(route, requests === 1 ? status : 200,
        requests === 1 ? { detail: { code, message } } : explanation)
    })
    await predict(page)
    await page.getByRole('button', { name: 'Explain this prediction' }).click()
    await expect(page.getByRole('region', { name: 'Prediction explanation' }).getByRole('alert')).toHaveText(message)
    await expect(page.getByText('84.2', { exact: true })).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Supporting evidence' })).not.toBeVisible()
    await page.getByRole('button', { name: 'Try explanation again' }).click()
    await expect(page.getByRole('heading', { name: 'Supporting evidence' })).toBeVisible()
    expect(requests).toBe(2)
  })
}

test('explanation for a different prediction is not shown', async ({ page }) => {
  await page.route('**/api/explain', route => jsonResponse(route, 200, {
    ...explanation, evidence: { ...explanation.evidence, score: 99 },
  }))
  await predict(page)
  await page.getByRole('button', { name: 'Explain this prediction' }).click()
  await expect(page.getByRole('region', { name: 'Prediction explanation' }).getByRole('alert')).toContainText('The prediction changed')
  await expect(page.getByText(explanation.sentences[0].text)).not.toBeVisible()
})

test('an unknown evidence reference is not silently presented as score evidence', async ({ page }) => {
  await page.route('**/api/explain', route => jsonResponse(route, 200, {
    ...explanation, sentences: [
      explanation.sentences[0],
      { text: 'Invented statement', evidence_id: 'unknown-source' },
    ],
  }))
  await predict(page)
  await page.getByRole('button', { name: 'Explain this prediction' }).click()
  await expect(page.getByRole('region', { name: 'Prediction explanation' }).getByRole('alert')).toContainText('response was invalid')
  await expect(page.getByText('Invented statement')).not.toBeVisible()
})

test('changing sample inputs clears the old explanation', async ({ page }) => {
  await page.route('**/api/explain', route => jsonResponse(route, 200, explanation))
  await predict(page)
  await page.getByRole('button', { name: 'Explain this prediction' }).click()
  await expect(page.getByRole('heading', { name: 'Supporting evidence' })).toBeVisible()
  await page.locator('input[type="number"]').first().fill('2000')
  await expect(page.getByRole('region', { name: 'Prediction explanation' })).not.toBeVisible()
})

test('explanation stays usable at a narrow mobile viewport', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 360, height: 800 })
  await page.route('**/api/explain', route => jsonResponse(route, 200, explanation))
  await predict(page)
  await page.getByRole('button', { name: 'Explain this prediction' }).click()
  await expect(page.getByRole('heading', { name: 'Supporting evidence' })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await page.getByRole('region', { name: 'Prediction explanation' }).screenshot({
    path: testInfo.outputPath('explanation-mobile.png'),
  })
})
