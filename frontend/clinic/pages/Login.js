/**
 * Clinic User Login Page.
 * 
 * Supports secure practitioner authentication, input validation,
 * loading states, demo clinic credentials, and redirect to Clinic Dashboard.
 */

import { clinicAuthService } from '../services/auth.js';
import { clinicStore } from '../hooks/store.js';

export function renderClinicLoginPage(container) {
  container.innerHTML = `
    <div class="auth-page">
      <div class="auth-card">
        <div class="auth-header">
          <span class="auth-logo">🏥</span>
          <h1 class="auth-title">Clinical Provider Sign In</h1>
          <p class="auth-subtitle">Authorized Healthcare Practitioner & Clinic Portal</p>
        </div>

        <form id="clinic-login-form" class="auth-form" novalidate>
          <div class="form-group">
            <label for="clinic-login-email" class="form-label">Institutional Clinic Email <span class="required">*</span></label>
            <input 
              type="email" 
              id="clinic-login-email" 
              name="email" 
              class="form-control" 
              placeholder="practitioner@clinic.org" 
              autocomplete="email" 
              required 
            />
          </div>

          <div class="form-group">
            <label for="clinic-login-password" class="form-label">Password <span class="required">*</span></label>
            <input 
              type="password" 
              id="clinic-login-password" 
              name="password" 
              class="form-control" 
              placeholder="••••••••" 
              autocomplete="current-password" 
              required 
            />
          </div>

          <div class="auth-actions">
            <button type="submit" class="btn btn-primary btn-block btn-lg" id="btn-submit-clinic-login">
              Sign In to Clinic Console
            </button>
          </div>
        </form>

        <div class="demo-credentials-box">
          <small><strong>Quick Demo Provider Access:</strong><br />Email: <code>clinic@generalhospital.org</code><br />Password: <code>Clinic@123</code></small>
        </div>

        <div class="auth-footer">
          <p>Looking for patient self-screening? <a href="./index.html" class="btn-link">Go to Patient Portal</a></p>
        </div>
      </div>
    </div>
  `;

  const form = container.querySelector('#clinic-login-form');
  const btnSubmit = container.querySelector('#btn-submit-clinic-login');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    clinicStore.clearAlert();

    const email = form.email.value.trim();
    const password = form.password.value;

    if (!email) {
      clinicStore.setAlert('error', 'Please enter your institutional clinic email address.');
      return;
    }

    if (!password) {
      clinicStore.setAlert('error', 'Please enter your password.');
      return;
    }

    try {
      btnSubmit.disabled = true;
      btnSubmit.textContent = 'Authenticating Clinic Session...';

      await clinicAuthService.login({ email, password });
      clinicStore.setAlert('success', 'Clinic provider session established.');
      clinicStore.navigate('dashboard');
    } catch (err) {
      clinicStore.setAlert('error', err.message || 'Clinic authentication failed.');
    } finally {
      btnSubmit.disabled = false;
      btnSubmit.textContent = 'Sign In to Clinic Console';
    }
  });
}
