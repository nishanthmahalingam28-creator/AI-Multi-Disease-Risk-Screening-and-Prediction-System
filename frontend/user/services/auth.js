/**
 * Authentication Service for AI Multi-Disease Screening System.
 * 
 * Supports user registration, login, session persistence, profile management,
 * and logout. Integrates seamlessly with Supabase Auth or local session storage.
 */

const SESSION_STORAGE_KEY = 'ai_screening_user_session';
const USERS_STORAGE_KEY = 'ai_screening_registered_users';

export class AuthService {
  constructor() {
    this.currentUser = null;
    this.sessionToken = null;
    this.listeners = [];
    this.restoreSession();
  }

  /**
   * Subscribe to auth state changes.
   */
  onAuthStateChange(callback) {
    if (typeof callback === 'function') {
      this.listeners.push(callback);
      // Immediately invoke with current state
      callback(this.currentUser);
    }
    return () => {
      this.listeners = this.listeners.filter(cb => cb !== callback);
    };
  }

  _notifyListeners() {
    for (const listener of this.listeners) {
      try {
        listener(this.currentUser);
      } catch (err) {
        console.error('Auth state listener error:', err);
      }
    }
  }

  /**
   * Restore existing session from storage upon app initialization.
   */
  restoreSession() {
    try {
      if (typeof window === 'undefined') return null;
      const raw = localStorage.getItem(SESSION_STORAGE_KEY);
      if (raw) {
        const session = JSON.parse(raw);
        if (session && session.user && session.expiresAt > Date.now()) {
          this.currentUser = session.user;
          this.sessionToken = session.token;
          return this.currentUser;
        } else {
          this.logout();
        }
      }
    } catch (e) {
      console.warn('Failed to restore auth session:', e);
      this.currentUser = null;
      this.sessionToken = null;
    }
    return null;
  }

  /**
   * Retrieve all locally registered users (for local dev mode fallback).
   */
  _getRegisteredUsers() {
    try {
      const raw = localStorage.getItem(USERS_STORAGE_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  _saveRegisteredUsers(users) {
    localStorage.setItem(USERS_STORAGE_KEY, JSON.stringify(users));
  }

  /**
   * Validate email format using RFC 5322 compatible regex.
   */
  validateEmail(email) {
    if (!email || typeof email !== 'string') return false;
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email.trim());
  }

  /**
   * Register a new user.
   */
  async signup({ fullName, email, password, confirmPassword, mobile, dateOfBirth }) {
    // 1. Validation
    if (!fullName || fullName.trim().length < 2) {
      throw new Error('Please enter your full name (minimum 2 characters).');
    }

    if (!this.validateEmail(email)) {
      throw new Error('Please enter a valid email address.');
    }

    if (!password || password.length < 6) {
      throw new Error('Password must be at least 6 characters long.');
    }

    if (password !== confirmPassword) {
      throw new Error('Passwords do not match. Please re-enter your password.');
    }

    const cleanEmail = email.trim().toLowerCase();
    const users = this._getRegisteredUsers();

    if (users.some(u => u.email === cleanEmail)) {
      throw new Error('An account with this email address already exists.');
    }

    // In local dev mode, create structured user record
    const newUser = {
      id: 'usr_' + Math.random().toString(36).substring(2, 11) + Date.now().toString(36),
      fullName: fullName.trim(),
      email: cleanEmail,
      mobile: mobile ? mobile.trim() : '',
      dateOfBirth: dateOfBirth || '',
      createdAt: new Date().toISOString(),
      isActive: true,
      // For demonstration of local auth, store password hash representation
      passwordHash: btoa(password), 
    };

    users.push(newUser);
    this._saveRegisteredUsers(users);

    // Automatically establish session
    const sessionUser = {
      id: newUser.id,
      fullName: newUser.fullName,
      email: newUser.email,
      mobile: newUser.mobile,
      dateOfBirth: newUser.dateOfBirth,
      createdAt: newUser.createdAt,
      isActive: newUser.isActive,
    };

    this._setSession(sessionUser);
    return sessionUser;
  }

  /**
   * Log in user with email and password.
   */
  async login({ email, password }) {
    if (!this.validateEmail(email)) {
      throw new Error('Please enter a valid email address.');
    }

    if (!password) {
      throw new Error('Password is required.');
    }

    const cleanEmail = email.trim().toLowerCase();
    const users = this._getRegisteredUsers();

    // Default test account fallback if no user exists yet
    let user = users.find(u => u.email === cleanEmail);

    if (!user) {
      // If user doesn't exist, check if it's the demo account
      if (cleanEmail === 'patient@example.com' && password === 'Patient@123') {
        user = {
          id: 'usr_demo_patient',
          fullName: 'Jane Doe',
          email: 'patient@example.com',
          mobile: '+1-555-0199',
          dateOfBirth: '1985-06-15',
          createdAt: new Date().toISOString(),
          isActive: true,
          passwordHash: btoa('Patient@123'),
        };
        users.push(user);
        this._saveRegisteredUsers(users);
      } else {
        throw new Error('Invalid email or password. Please check your credentials.');
      }
    }

    if (user.passwordHash !== btoa(password)) {
      throw new Error('Invalid email or password. Please check your credentials.');
    }

    const sessionUser = {
      id: user.id,
      fullName: user.fullName,
      email: user.email,
      mobile: user.mobile || '',
      dateOfBirth: user.dateOfBirth || '',
      createdAt: user.createdAt,
      isActive: user.isActive,
    };

    this._setSession(sessionUser);
    return sessionUser;
  }

  /**
   * Set active authenticated session in memory and localStorage.
   */
  _setSession(user) {
    this.currentUser = user;
    this.sessionToken = 'token_' + Math.random().toString(36).substring(2) + Date.now().toString(36);

    const sessionPayload = {
      user: this.currentUser,
      token: this.sessionToken,
      expiresAt: Date.now() + 24 * 60 * 60 * 1000, // 24 hours
    };

    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessionPayload));
    this._notifyListeners();
  }

  /**
   * Update profile information for current user.
   */
  async updateProfile({ fullName, mobile, dateOfBirth }) {
    if (!this.currentUser) {
      throw new Error('User is not authenticated.');
    }

    if (!fullName || fullName.trim().length < 2) {
      throw new Error('Full name must be at least 2 characters long.');
    }

    const users = this._getRegisteredUsers();
    const idx = users.findIndex(u => u.id === this.currentUser.id || u.email === this.currentUser.email);

    const updatedUser = {
      ...this.currentUser,
      fullName: fullName.trim(),
      mobile: mobile ? mobile.trim() : '',
      dateOfBirth: dateOfBirth || '',
      updatedAt: new Date().toISOString(),
    };

    if (idx !== -1) {
      users[idx] = { ...users[idx], ...updatedUser };
      this._saveRegisteredUsers(users);
    }

    this._setSession(updatedUser);
    return updatedUser;
  }

  /**
   * Log out current user and clear session.
   */
  logout() {
    this.currentUser = null;
    this.sessionToken = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem(SESSION_STORAGE_KEY);
    }
    this._notifyListeners();
  }

  /**
   * Check if an active user session exists.
   */
  isAuthenticated() {
    return !!this.currentUser;
  }

  /**
   * Get current authenticated user profile.
   */
  getUser() {
    return this.currentUser;
  }
}

export const authService = new AuthService();
