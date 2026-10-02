/**
 * Dashboard & Page Header Component.
 * 
 * Renders personalized greeting, purpose introduction, and breadcrumbs.
 */

export function renderHeader(container, { title, subtitle, userName }) {
  const greeting = userName ? `Welcome, ${userName}` : 'Welcome to the Screening System';

  container.innerHTML = `
    <div class="page-header">
      <div class="header-main">
        <h1 class="header-title">${title || greeting}</h1>
        <p class="header-subtitle">${subtitle || 'Select an active disease screening model below to assess clinical statistical risk factors.'}</p>
      </div>
      <div class="header-meta">
        <span class="badge badge-pulse">● System Online</span>
        <span class="badge badge-neutral">Supabase & REST API</span>
      </div>
    </div>
  `;
}
