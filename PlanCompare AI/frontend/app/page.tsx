'use client';

import { FormEvent, useState } from 'react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';

type ComparisonResponse = {
  message?: string;
  llm_report?: unknown;
  bounding_boxes?: unknown[];
  files?: Record<string, string>;
};

function formatComparisonOutput(data: ComparisonResponse) {
  return JSON.stringify(
    {
      report: data.llm_report ?? 'No report returned.',
      changed_regions: data.bounding_boxes?.length ?? 0,
      files: data.files ?? {},
      message: data.message ?? '',
    },
    null,
    2,
  );
}

export default function HomePage() {
  const [previousDrawing, setPreviousDrawing] = useState<File | null>(null);
  const [revisedDrawing, setRevisedDrawing] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [detailedResult, setDetailedResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setResult(null);
    setDetailedResult(null);

    if (!previousDrawing || !revisedDrawing) {
      setError('Please upload both PDF drawings before comparing.');
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append('old_file', previousDrawing);
    formData.append('new_file', revisedDrawing);

    try {
      const response = await fetch(`${API_BASE_URL}/compare-drawings`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json().catch(() => null);
        throw new Error(data?.detail || 'Comparison failed.');
      }

      const data = (await response.json()) as ComparisonResponse;
      setResult(data.message || 'Comparison completed successfully.');
      setDetailedResult(formatComparisonOutput(data));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="page-shell">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">PlanCompare</p>
          <h1>Drawing comparison for built environments</h1>
          <p className="subtext">
            Upload previous and revised PDFs, then compare critical changes with one clean, construction-style workflow.
          </p>
        </div>
      </section>

      <section className="form-panel">
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="upload-grid">
            <label className="upload-card">
              <span>Previous drawing</span>
              <input
                type="file"
                accept="application/pdf"
                onChange={(e) => setPreviousDrawing(e.target.files?.[0] ?? null)}
              />
              <p className="file-note">{previousDrawing?.name ?? 'PDF only, max 30 MB'}</p>
            </label>

            <label className="upload-card">
              <span>Revised drawing</span>
              <input
                type="file"
                accept="application/pdf"
                onChange={(e) => setRevisedDrawing(e.target.files?.[0] ?? null)}
              />
              <p className="file-note">{revisedDrawing?.name ?? 'PDF only, max 30 MB'}</p>
            </label>
          </div>

          <button className="submit-button" type="submit" disabled={isLoading}>
            {isLoading ? 'Comparing...' : 'Compare drawings'}
          </button>

          {error && <div className="message error">{error}</div>}
          {result && <div className="message success">{result}</div>}
          {detailedResult && (
            <section className="result-box" aria-label="Detailed comparison output">
              <div className="result-header">
                <h2>Detailed output</h2>
                <span>Generated report</span>
              </div>
              <textarea value={detailedResult} readOnly />
            </section>
          )}
        </form>
      </section>
    </main>
  );
}
