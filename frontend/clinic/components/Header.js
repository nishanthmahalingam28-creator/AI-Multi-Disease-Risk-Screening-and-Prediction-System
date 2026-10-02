/**
 * Clinic Header Component.
 * 
 * Displays provider greeting, institutional clinic affiliation, and system status.
 */

export function renderClinicHeader(container, { title, subtitle, clinicUser }) {
  const providerName = clinicUser ? clinicUser.fullName : 'Clinical Practitioner';
  const clinicName = clinicUser ? clinicUser.clinicName : 'Authorized Clinic';

  container.innerHTML = `
    <div class="page-header clinic-header">
      <div class="header-main">
        <h1 class="header-title">${title || `Provider Portal — ${clinicName}`}</h1>
        <p class="header-subtitle">${subtitle || `Practitioner: ${providerName} | Authorized Clinical Screening Console`}</p>
      </div>
      <div class="header-meta">
        <span class="badge badge-success">● Provider Active</span>
        <span class="badge badge-pulse">${clinicName}</span>
      </div>
    </div>
  `;
}
