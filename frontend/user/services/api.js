/**
 * Centralized API Service Layer for AI Multi-Disease Screening Frontend.
 * 
 * Communicates with the Flask REST API backend endpoints:
 * - GET  /api/health
 * - GET  /api/models
 * - POST /api/predict/lung-cancer
 * - POST /api/predict/asthma
 * - POST /api/predict/parkinsons
 */

export class ApiService {
  constructor() {
    // Configurable base URL with sensible local default
    const envUrl = (typeof window !== 'undefined' && window.__ENV__?.API_BASE_URL) || null;
    const storedUrl = (typeof window !== 'undefined' && localStorage.getItem('API_BASE_URL')) || null;
    this.baseUrl = envUrl || storedUrl || 'http://127.0.0.1:5000/api';
  }

  /**
   * Set API base URL dynamically.
   */
  setBaseUrl(url) {
    if (url && typeof url === 'string') {
      this.baseUrl = url.replace(/\/+$/, '');
      if (typeof window !== 'undefined') {
        localStorage.setItem('API_BASE_URL', this.baseUrl);
      }
    }
  }

  /**
   * Internal request helper with structured error handling.
   */
  async _request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const defaultHeaders = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };

    const config = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...(options.headers || {}),
      },
    };

    try {
      const response = await fetch(url, config);
      let data = null;

      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        data = await response.json();
      } else {
        const text = await response.text();
        data = { message: text };
      }

      if (!response.ok) {
        const errorMessage = data?.error?.message || data?.message || `Request failed with status ${response.status}`;
        const error = new Error(errorMessage);
        error.status = response.status;
        error.code = data?.error?.code || 'API_ERROR';
        error.details = data;
        throw error;
      }

      return data;
    } catch (err) {
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        const networkError = new Error('Unable to reach the prediction server. Please verify the backend is running at ' + this.baseUrl);
        networkError.code = 'NETWORK_ERROR';
        throw networkError;
      }
      throw err;
    }
  }

  /**
   * Check backend health and operational status.
   */
  async checkHealth() {
    return this._request('/health', { method: 'GET' });
  }

  /**
   * Fetch currently available models from the backend.
   */
  async getAvailableModels() {
    return this._request('/models', { method: 'GET' });
  }

  /**
   * Submit Lung Cancer screening prediction request.
   * @param {Object} payload Feature inputs matching LungCancerInputSchema
   */
  async predictLungCancer(payload) {
    return this._request('/predict/lung-cancer', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  /**
   * Submit Asthma screening prediction request.
   * @param {Object} payload Feature inputs matching AsthmaInputSchema
   */
  async predictAsthma(payload) {
    return this._request('/predict/asthma', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  /**
   * Submit Parkinson's screening prediction request.
   * @param {Object} payload Feature inputs matching ParkinsonsInputSchema (22 acoustic features)
   */
  async predictParkinsons(payload) {
    return this._request('/predict/parkinsons', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  /**
   * Generic disease prediction dispatcher.
   */
  async predict(diseaseCode, payload) {
    switch (diseaseCode) {
      case 'lung_cancer':
        return this.predictLungCancer(payload);
      case 'asthma':
        return this.predictAsthma(payload);
      case 'parkinsons':
        return this.predictParkinsons(payload);
      default:
        throw new Error(`Prediction model for '${diseaseCode}' is currently under development and not yet available.`);
    }
  }

  /**
   * Persist a completed screening to the backend database.
   */
  async saveScreening({ diseaseCode, inputData, predictionResult, user = null }) {
    const headers = {};
    if (user && user.id) {
      headers['X-User-Id'] = user.id;
    }
    if (user && user.email) {
      headers['X-User-Email'] = user.email;
    }

    const payload = {
      disease_code: diseaseCode,
      input_data: inputData,
      prediction_result: predictionResult.prediction || predictionResult,
      user_id: user ? user.id : undefined,
    };

    return this._request('/screenings', {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });
  }

  /**
   * Retrieve current user's historical screenings with optional pagination and filters.
   */
  async getUserHistory({ user = null, page = 1, pageSize = 20, disease = null, riskLevel = null } = {}) {
    const headers = {};
    if (user && user.id) {
      headers['X-User-Id'] = user.id;
    }
    if (user && user.email) {
      headers['X-User-Email'] = user.email;
    }

    const params = new URLSearchParams();
    if (page) params.set('page', page);
    if (pageSize) params.set('page_size', pageSize);
    if (disease) params.set('disease', disease);
    if (riskLevel) params.set('risk_level', riskLevel);

    const query = params.toString() ? `?${params.toString()}` : '';
    return this._request(`/history/user${query}`, {
      method: 'GET',
      headers,
    });
  }

  /**
   * Fetch single screening detailed record for report rendering.
   */
  async getScreeningDetail(screeningId, user = null) {
    const headers = {};
    if (user && user.id) {
      headers['X-User-Id'] = user.id;
    }
    if (user && user.email) {
      headers['X-User-Email'] = user.email;
    }

    return this._request(`/screenings/${screeningId}`, {
      method: 'GET',
      headers,
    });
  }
}

export const apiService = new ApiService();

