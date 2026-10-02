/**
 * User Registration / Signup Page.
 * 
 * Supports new user creation with full name, email, password confirmation,
 * mobile, and date of birth.
 */

import { authService } from '../services/auth.js';
import { store } from '../hooks/store.js';

export function renderSignupPage(container) {
  container.innerHTML = `
    <div class="auth-page">
      <div class="auth-card auth-card-wide">
        <div class="auth-header">
          <span class="auth-logo">📝</span>
          <h1 class="auth-title">Create Patient Account</h1>
          <p class="auth-subtitle">Register to begin proactive AI multi-disease health screenings</p>
        </div>

        <form id="signup-form" class="auth-form" novalidate>
          <div class="form-grid">
            <div class="form-group full-width">
              <label for="signup-name" class="form-label">Full Name <span class="required">*</span></label>
              <input 
                type="text" 
                id="signup-name" 
                name="fullName" 
                class="form-control" 
                placeholder="e.g. Jane Doe" 
                required 
              />
            </div>

            <div class="form-group full-width">
              <label for="signup-email" class="form-label">Email Address <span class="required">*</span></label>
              <input 
                type="email" 
                id="signup-email" 
                name="email" 
                class="form-control" 
                placeholder="you@example.com" 
                autocomplete="email" 
                required 
              />
              <small class="form-hint">Used as your unique screening identification identity</small>
            </div>

            <div class="form-group">
              <label for="signup-mobile" class="form-label">Mobile Number</label>
              <input 
                type="tel" 
                id="signup-mobile" 
                name="mobile" 
                class="form-control" 
                placeholder="+1 555-0199" 
              />
            </div>

            <div class="form-group">
              <label for="signup-dob" class="form-label">Date of Birth</label>
              <input 
                type="date" 
                id="signup-dob" 
                name="dateOfBirth" 
                class="form-control" 
              />
            </div>

            <div class="form-group">
              <label for="signup-password" class="form-label">Password <span class="required">*</span></label>
              <input 
                type="password" 
                id="signup-password" 
                name="password" 
                class="form-control" 
                placeholder="At least 6 characters" 
                autocomplete="new-password" 
                required 
              />
            </div>

            <div class="form-group">
              <label for="signup-confirm-password" class="form-label">Confirm Password <span class="required">*</span></label>
              <input 
                type="password" 
                id="signup-confirm-password" 
                name="confirmPassword" 
                class="form-control" 
                placeholder="Re-type password" 
                autocomplete="new-password" 
                required 
              />
            </div>
          </div>

          <div class="auth-actions">
            <button type="submit" class="btn btn-primary btn-block btn-lg" id="btn-submit-signup">
              Complete Registration & Open Dashboard
            </button>
          </div>
        </form>

        <div class="auth-footer">
          <p>Already have an account? <button type="button" class="btn-link" id="link-to-login">Sign In Instead</button></p>
        </div>
      </div>
    </div>
  `;

  const form = container.querySelector('#signup-form');
  const btnSubmit = container.querySelector('#btn-submit-signup');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    store.clearAlert();

    const fullName = form.fullName.value.trim();
    const email = form.email.value.trim();
    const mobile = form.mobile.value.trim();
    const dateOfBirth = form.dateOfBirth.value;
    const password = form.password.value;
    const confirmPassword = form.confirmPassword.value;

    if (!fullName) {
      store.setAlert('error', 'Please enter your full name.');
      return;
    }

    if (!email) {
      store.setAlert('error', 'Please enter your email address.');
      return;
    }

    if (!password) {
      store.setAlert('error', 'Please create a password.');
      return;
    }

    if (password.length < 6) {
      store.setAlert('error', 'Password must be at least 6 characters long.');
      return;
    }

    if (password !== confirmPassword) {
      store.setAlert('error', 'Passwords do not match. Please verify both password fields.');
      return;
    }

    try {
      btnSubmit.disabled = true;
      btnSubmit.textContent = 'Creating Account...';

      await authService.signup({
        fullName,
        email,
        password,
        confirmPassword,
        mobile,
        dateOfBirth,
      });

      store.setAlert('success', `Welcome, ${fullName}! Your account has been created.`);
      store.navigate('dashboard');
    } catch (err) {
      store.setAlert('error', err.message || 'Registration failed.');
    } finally {
      btnSubmit.disabled = false;
      btnSubmit.textContent = 'Complete Registration & Open Dashboard';
    }
  });

  container.querySelector('#link-to-login')?.addEventListener('click', () => {
    store.navigate('login');
  });
}
