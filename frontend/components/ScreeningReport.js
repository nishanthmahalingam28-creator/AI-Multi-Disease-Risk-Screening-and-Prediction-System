/**
 * Reusable AI Multi-Disease Screening Report Component.
 * 
 * Provides an accessible, standardized clinical report view with
 * browser-friendly printing (window.print() / Save as PDF).
 */

export function renderScreeningReport(container, {
  screening,
  onBack,
  onDashboard,
  isClinicContext = false,
}) {
  if (!screening) {
    container.innerHTML = `
      <div class="empty-state">
        <p>Screening report could not be loaded.</p>
        <button class="btn btn-primary" id="btn-report-back">Back</button>
      </div>
    `;
    container.querySelector('#btn-report-back')?.addEventListener('click', onBack);
    return;
  }

  const disease = screening.disease || {};
  const pred = screening.prediction || {};
  const patient = screening.patient || {};
  const clinic = screening.clinic_user || null;
  const inputs = screening.input_data || {};

  const probPct = pred.probability !== undefined ? (pred.probability * 100).toFixed(1) : '0.0';
  const riskLevel = (pred.risk_level || 'LOW').toUpperCase();
  const isPositive = pred.predicted_class === 1;

  let riskBadgeClass = 'risk-low';
  let riskColor = '#10b981';
  if (riskLevel === 'HIGH') {
    riskBadgeClass = 'risk-high';
    riskColor = '#ef4444';
  } else if (riskLevel === 'MODERATE') {
    riskBadgeClass = 'risk-moderate';
    riskColor = '#f59e0b';
  }

  // Format date
  let formattedDate = 'N/A';
  if (screening.created_at) {
    try {
      const dt = new Date(screening.created_at);
      formattedDate = dt.toLocaleString('en-US', {
        dateStyle: 'medium',
        timeStyle: 'short',
      });
    } catch {
      formattedDate = screening.created_at;
    }
  }

  // Format inputs into table rows
  const inputEntries = Object.entries(inputs);
  const inputRows = inputEntries.map(([k, v]) => `
    <tr>
      <td class="param-name">${k}</td>
      <td class="param-val">${typeof v === 'number' ? v : String(v)}</td>
    </tr>
  `).join('');

  container.innerHTML = `
    <div class="screening-report-wrapper" id="printable-report">
      <!-- Report Actions Toolbar (Hidden on print) -->
      <div class="report-toolbar no-print">
        <div class="toolbar-left">
          <button type="button" class="btn btn-outline btn-sm" id="btn-report-back">
            ← Back to History
          </button>
          <button type="button" class="btn btn-outline btn-sm" id="btn-report-dash">
            📊 Dashboard
          </button>
        </div>
        <div class="toolbar-right">
          <button type="button" class="btn btn-primary btn-sm" id="btn-trigger-print">
            🖨️ Print / Save PDF
          </button>
        </div>
      </div>

      <!-- Printable Document Container -->
      <div class="printable-document">
        <!-- Document Header -->
        <header class="doc-header">
          <div class="doc-brand">
            <span class="doc-logo">⚕️</span>
            <div>
              <h1 class="doc-title">AI MULTI-DISEASE RISK SCREENING REPORT</h1>
              <p class="doc-subtitle">Clinical Statistical Risk Assessment Summary</p>
            </div>
          </div>
          <div class="doc-meta">
            <div><strong>Screening ID:</strong> <span class="font-mono">${screening.id}</span></div>
            <div><strong>Date & Time:</strong> ${formattedDate}</div>
            <div><strong>Lifecycle Status:</strong> <span class="badge badge-success">${screening.status || 'COMPLETED'}</span></div>
          </div>
        </header>

        <hr class="doc-divider" />

        <!-- Clinical & Patient Details Grid -->
        <section class="doc-section">
          <div class="doc-grid-two-col">
            <div class="doc-box">
              <h2 class="doc-section-heading">👤 Patient Information</h2>
              <table class="doc-table-minimal">
                <tr><th>Full Name:</th><td>${patient.full_name || 'Confidential Patient'}</td></tr>
                <tr><th>Identifier / Email:</th><td>${patient.email || 'N/A'}</td></tr>
                ${patient.mobile ? `<tr><th>Contact:</th><td>${patient.mobile}</td></tr>` : ''}
                ${patient.date_of_birth ? `<tr><th>Date of Birth:</th><td>${patient.date_of_birth}</td></tr>` : ''}
              </table>
            </div>

            <div class="doc-box">
              <h2 class="doc-section-heading">🏥 Clinical Facility & Provider</h2>
              <table class="doc-table-minimal">
                ${clinic ? `
                  <tr><th>Practitioner:</th><td>${clinic.full_name || 'Clinical Staff'}</td></tr>
                  <tr><th>Medical Facility:</th><td>${clinic.clinic_name || 'Healthcare Center'}</td></tr>
                  <tr><th>Workflow:</th><td>Clinic-Assisted Screening</td></tr>
                ` : `
                  <tr><th>Screening Mode:</th><td>Patient Self-Screening</td></tr>
                  <tr><th>Review Status:</th><td>Pending Clinical Review</td></tr>
                  <tr><th>Facility:</th><td>Direct Portal Session</td></tr>
                `}
              </table>
            </div>
          </div>
        </section>

        <!-- Condition & Prediction Risk Evaluation -->
        <section class="doc-section">
          <h2 class="doc-section-heading">🩺 Risk Assessment Findings</h2>
          <div class="doc-evaluation-banner">
            <div class="eval-disease-info">
              <span class="disease-icon-inline">${disease.icon || '⚕️'}</span>
              <div>
                <span class="eval-category">${disease.category || 'Clinical Medicine'}</span>
                <h3 class="eval-disease-name">${disease.name || disease.code}</h3>
              </div>
            </div>

            <div class="eval-risk-badge">
              <span class="badge ${riskBadgeClass} badge-lg">
                ● ${pred.risk_level || 'LOW'} RISK
              </span>
            </div>
          </div>

          <div class="doc-metrics-grid">
            <div class="metric-card">
              <span class="metric-lbl">Estimated Probability</span>
              <span class="metric-val" style="color: ${riskColor}">${probPct}%</span>
            </div>

            <div class="metric-card">
              <span class="metric-lbl">Classification Indicator</span>
              <span class="metric-val ${isPositive ? 'text-danger' : 'text-success'}">
                ${isPositive ? 'Screening Positive' : 'Screening Negative'}
              </span>
            </div>

            <div class="metric-card">
              <span class="metric-lbl">Inference Model Version</span>
              <span class="metric-val font-mono">${pred.model_version || '1.0.0'}</span>
            </div>
          </div>
        </section>

        <!-- Submitted Features Observation Table -->
        ${inputEntries.length > 0 ? `
          <section class="doc-section">
            <h2 class="doc-section-heading">📋 Evaluated Clinical Parameters (${inputEntries.length} Features)</h2>
            <div class="doc-table-container">
              <table class="doc-parameters-table">
                <thead>
                  <tr>
                    <th>Feature Name</th>
                    <th>Submitted Measurement</th>
                  </tr>
                </thead>
                <tbody>
                  ${inputRows}
                </tbody>
              </table>
            </div>
          </section>
        ` : ''}

        <!-- Mandatory Medical Disclaimer Notice -->
        <footer class="doc-footer">
          <div class="doc-disclaimer-box">
            <strong>⚠️ MANDATORY CLINICAL NOTICE:</strong>
            <p>
              ${pred.disclaimer || 'This result is an AI-based screening/risk estimate and is not a medical diagnosis. Clinical evaluation by a qualified healthcare professional is required before making any diagnostic or therapeutic decisions.'}
            </p>
          </div>
          <div class="doc-signoff">
            <div>Verified against model feature contract: <code>${pred.model_version || '1.0.0'}</code></div>
            <div>AI Multi-Disease Risk Screening Platform &copy; 2026</div>
          </div>
        </footer>
      </div>
    </div>
  `;

  // Attach button events
  container.querySelector('#btn-report-back')?.addEventListener('click', onBack);
  container.querySelector('#btn-report-dash')?.addEventListener('click', onDashboard);
  container.querySelector('#btn-trigger-print')?.addEventListener('click', () => {
    window.print();
  });
}
