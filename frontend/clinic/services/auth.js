/**
 * Clinic User Authentication Service.
 * 
 * Supports authorized clinic provider authentication, session management,
 * and clinic profile updates. Strictly separated from patient user authentication.
 */

import { CLINIC_ROLE } from '../types/constants.js';

const CLINIC_SESSION_KEY = 'ai_screening_clinic_session';
const CLINIC_USERS_KEY = 'ai_screening_clinic_users';

export class ClinicAuthService {
  constructor() {
    this.currentClinicUser = null;
    this.sessionToken = null;
    this.listeners = [];
    this.restoreSession();
  }

  /**
   * Subscribe to clinic auth state changes.
   */
  onAuthStateChange(callback) {
    if (typeof callback === 'function') {
      this.listeners.push(callback);
      callback(this.currentClinicUser);
    }
    return () => {
      this.listeners = this.listeners.filter(cb => cb !== callback);
    };
  }

  _notifyListeners() {
    for (const listener of this.listeners) {
      try {
        listener(this.currentClinicUser);
      } catch (err) {
        console.error('Clinic auth listener error:', err);
      }
    }
  }

  /**
   * Restore existing clinic session from storage.
   */
  restoreSession() {
    try {
      if (typeof window === 'undefined') return null;
      const raw = localStorage.getItem(CLINIC_SESSION_KEY);
      if (raw) {
        const session = JSON.parse(raw);
        if (session && session.user && session.user.role === CLINIC_ROLE && session.expiresAt > Date.now()) {
          this.currentClinicUser = session.user;
          this.sessionToken = session.token;
          return this.currentClinicUser;
        } else {
          this.logout();
        }
      }
    } catch (e) {
      console.warn('Failed to restore clinic session:', e);
      this.currentClinicUser = null;
      this.sessionToken = null;
    }
    return null;
  }

  /**
   * Retrieve all locally registered clinic users.
   */
  _getRegisteredClinicUsers() {
    try {
      const raw = localStorage.getItem(CLINIC_USERS_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  _saveRegisteredClinicUsers(users) {
    localStorage.setItem(CLINIC_USERS_KEY, JSON.stringify(users));
  }

  /**
   * Validate email format using standard regex.
   */
  validateEmail(email) {
    if (!email || typeof email !== 'string') return false;
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email.trim());
  }

  /**
   * Authenticate clinic user.
   */
  async login({ email, password }) {
    if (!this.validateEmail(email)) {
      throw new Error('Please provide a valid institutional clinic email address.');
    }

    if (!password) {
      throw new Error('Password is required for clinical provider authentication.');
    }

    const cleanEmail = email.trim().toLowerCase();
    const clinicUsers = this._getRegisteredClinicUsers();

    let user = clinicUsers.find(u => u.email === cleanEmail);

    if (!user) {
      // Default demo clinic account for local testing
      if (cleanEmail === 'clinic@generalhospital.org' && password === 'Clinic@123') {
        user = {
          id: 'cln_mitchell_001',
          fullName: 'Dr. Sarah Mitchell, MD',
          email: 'clinic@generalhospital.org',
          mobile: '+1-555-0344',
          clinicName: 'Metro Health Pulmonary & Neurology Clinic',
          role: CLINIC_ROLE,
          createdAt: new Date().toISOString(),
          isActive: true,
          passwordHash: btoa('Clinic@123'),
        };
        clinicUsers.push(user);
        this._saveRegisteredClinicUsers(clinicUsers);
      } else {
        throw new Error('Invalid clinic credentials. Please check your institutional email and password.');
      }
    }

    if (user.passwordHash !== btoa(password)) {
      throw new Error('Invalid clinic credentials. Please check your institutional email and password.');
    }

    // Role check: Ensure this is a clinic user, not an ordinary patient
    if (user.role !== CLINIC_ROLE) {
      throw new Error('Access denied: Unauthorized role. This portal requires clinical provider credentials.');
    }

    const sessionUser = {
      id: user.id,
      fullName: user.fullName,
      email: user.email,
      mobile: user.mobile || '',
      clinicName: user.clinicName || 'Authorized Clinic',
      role: CLINIC_ROLE,
      createdAt: user.createdAt,
      isActive: user.isActive,
    };

    this._setSession(sessionUser);
    return sessionUser;
  }

  /**
   * Set active authenticated clinic session.
   */
  _setSession(user) {
    this.currentClinicUser = user;
    this.sessionToken = 'cln_token_' + Math.random().toString(36).substring(2) + Date.now().toString(36);

    const sessionPayload = {
      user: this.currentClinicUser,
      token: this.sessionToken,
      expiresAt: Date.now() + 12 * 60 * 60 * 1000, // 12 hours for clinical shift
    };

    localStorage.setItem(CLINIC_SESSION_KEY, JSON.stringify(sessionPayload));
    this._notifyListeners();
  }

  /**
   * Update clinic profile information.
   */
  async updateProfile({ fullName, mobile, clinicName }) {
    if (!this.currentClinicUser) {
      throw new Error('No active clinic session found.');
    }

    if (!fullName || fullName.trim().length < 2) {
      throw new Error('Practitioner full name must be at least 2 characters.');
    }

    if (!clinicName || clinicName.trim().length < 2) {
      throw new Error('Clinic / Hospital facility name must be at least 2 characters.');
    }

    const clinicUsers = this._getRegisteredClinicUsers();
    const idx = clinicUsers.findIndex(u => u.id === this.currentClinicUser.id || u.email === this.currentClinicUser.email);

    const updatedUser = {
      ...this.currentClinicUser,
      fullName: fullName.trim(),
      mobile: mobile ? mobile.trim() : '',
      clinicName: clinicName.trim(),
      updatedAt: new Date().toISOString(),
    };

    if (idx !== -1) {
      clinicUsers[idx] = { ...clinicUsers[idx], ...updatedUser };
      this._saveRegisteredClinicUsers(clinicUsers);
    }

    this._setSession(updatedUser);
    return updatedUser;
  }

  /**
   * Log out clinic user.
   */
  logout() {
    this.currentClinicUser = null;
    this.sessionToken = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem(CLINIC_SESSION_KEY);
    }
    this._notifyListeners();
  }

  /**
   * Verify if current session belongs to an authorized clinic user.
   */
  isAuthenticated() {
    return !!this.currentClinicUser && this.currentClinicUser.role === CLINIC_ROLE;
  }

  /**
   * Get active clinic user profile.
   */
  getClinicUser() {
    return this.currentClinicUser;
  }
}

export const clinicAuthService = new ClinicAuthService();
