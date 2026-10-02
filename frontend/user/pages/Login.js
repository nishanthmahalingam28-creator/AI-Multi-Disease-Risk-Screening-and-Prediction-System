/**
 * User Login Page.
 * 
 * Supports secure authentication with email & password, loading state,
 * input validation, and redirect to Dashboard.
 */

import { authService } from '../services/auth.js';
import { store } from '../hooks/store.js';

export function renderLoginPage(container) {
  container.innerHTML = `
    <div class="auth-page">
      <div class="auth-card">
        <div class="auth-header">
          <span class="auth-logo">⚕️</span>
          <h1 class="auth-title">Patient & User Sign In</h1>
          <p class="auth-subtitle">Access your AI multi-disease risk screening portal</p>
        </div>

        <form id="login-form" class="auth-form" novalidate>
          <div class="form-group">
            <label for="login-email" class="form-label">Email Address <span class="required">*</span></label>
            <input 
              type="email" 
              id="login-email" 
              name="email" 
              class="form-control" 
              placeholder="you@example.com" 
              autocomplete="email" 
              required 
            />
          </div>

          <div class="form-group">
            <label for="login-password" class="form-label">Password <span class="required">*</span></label>
            <input 
              type="password" 
              id="login-password" 
              name="password" 
              class="form-control" 
              placeholder="••••••••" 
              autocomplete="current-password" 
              required 
            />
          </div>

          <div class="auth-actions">
            <button type="submit" class="btn btn-primary btn-block btn-lg" id="btn-submit-login">
              Sign In to Portal
            </button>
          </div>
        </form>

        <div class="auth-footer">
          <p>Don't have an account yet? <button type="button" class="btn-link" id="link-to-signup">Create an Account</button></p>
        </div>

        <div class="demo-credentials-box">
          <small><strong>Quick Demo Access:</strong> Email: <code>patient@example.com</code> | Password: <code>Patient@123</code></small>
        </div>
      </div>
    </div>
  `;

  const form = container.querySelector('#login-form');
  const btnSubmit = container.querySelector('#btn-submit-login');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    store.clearAlert();

    const email = form.email.value.trim();
    const password = form.password.value;

    if (!email) {
      store.setAlert('error', 'Please enter your email address.');
      return;
    }

    if (!password) {
      store.setAlert('error', 'Please enter your password.');
      return;
    }

    try {
      btnSubmit.disabled = true;
      btnSubmit.textContent = 'Authenticating...';

      await authService.login({ email, password });
      store.setAlert('success', 'Welcome back! Authentication successful.');
      store.navigate('dashboard');
    } catch (err) {
      store.setAlert('error', err.message || 'Login failed. Please check your credentials.');
    } finally {
      btnSubmit.disabled = false;
      btnSubmit.textContent = 'Sign In to Portal';
    }
  });

  container.querySelector('#link-to-signup')?.addEventListener('click', () => {
    store.navigate('signup');
  });
}
