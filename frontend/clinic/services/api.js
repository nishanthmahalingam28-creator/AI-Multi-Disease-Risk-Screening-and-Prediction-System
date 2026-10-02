/**
 * Clinic API Service Layer.
 * 
 * Provides authorized API communication with Flask REST backend endpoints:
 * - GET  /api/health
 * - GET  /api/models
 * - POST /api/predict/lung-cancer
 * - POST /api/predict/asthma
 * - POST /api/predict/parkinsons
 */

export class ClinicApiService {
  constructor() {
    const envUrl = (typeof window !== 'undefined' && window.__ENV__?.API_BASE_URL) || null;
    const storedUrl = (typeof window !== 'undefined' && localStorage.getItem('API_BASE_URL')) || null;
    this.baseUrl = envUrl || storedUrl || 'http://127.0.0.1:5000/api';
  }

  setBaseUrl(url) {
    if (url && typeof url === 'string') {
      this.baseUrl = url.replace(/\/+$/, '');
      if (typeof window !== 'undefined') {
        localStorage.setItem('API_BASE_URL', this.baseUrl);
      }
    }
  }

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
        error.code = data?.error?.code || 'CLINIC_API_ERROR';
        error.details = data;
        throw error;
      }

      return data;
    } catch (err) {
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        const networkError = new Error('Clinical screening backend is unreachable at ' + this.baseUrl);
        networkError.code = 'NETWORK_ERROR';
        throw networkError;
      }
      throw err;
    }
  }

  async checkHealth() {
    return this._request('/health', { method: 'GET' });
  }

  async getAvailableModels() {
    return this._request('/models', { method: 'GET' });
  }

  async predictLungCancer(payload) {
    return this._request('/predict/lung-cancer', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async predictAsthma(payload) {
    return this._request('/predict/asthma', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async predictParkinsons(payload) {
    return this._request('/predict/parkinsons', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async predict(diseaseCode, payload) {
    switch (diseaseCode) {
      case 'lung_cancer':
        return this.predictLungCancer(payload);
      case 'asthma':
        return this.predictAsthma(payload);
      case 'parkinsons':
        return this.predictParkinsons(payload);
      default:
        throw new Error(`Screening model for '${diseaseCode}' is in clinical trial development and not yet available.`);
    }
  }

  /**
   * Persist a completed clinic consultation screening.
   */
  async saveScreening({ diseaseCode, inputData, predictionResult, clinicUser = null, patientInfo = null }) {
    const headers = {};
    if (clinicUser && clinicUser.id) {
      headers['X-Clinic-User-Id'] = clinicUser.id;
    }
    if (clinicUser && clinicUser.email) {
      headers['X-Clinic-User-Email'] = clinicUser.email;
    }

    const payload = {
      disease_code: diseaseCode,
      input_data: inputData,
      prediction_result: predictionResult.prediction || predictionResult,
      clinic_user_id: clinicUser ? clinicUser.id : undefined,
      patient_info: patientInfo || undefined,
    };

    return this._request('/screenings', {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });
  }

  /**
   * Retrieve clinic screening history associated with this practitioner/facility.
   */
  async getClinicHistory({ clinicUser = null, page = 1, pageSize = 20, disease = null, riskLevel = null } = {}) {
    const headers = {};
    if (clinicUser && clinicUser.id) {
      headers['X-Clinic-User-Id'] = clinicUser.id;
    }
    if (clinicUser && clinicUser.email) {
      headers['X-Clinic-User-Email'] = clinicUser.email;
    }

    const params = new URLSearchParams();
    if (page) params.set('page', page);
    if (pageSize) params.set('page_size', pageSize);
    if (disease) params.set('disease', disease);
    if (riskLevel) params.set('risk_level', riskLevel);

    const query = params.toString() ? `?${params.toString()}` : '';
    return this._request(`/history/clinic${query}`, {
      method: 'GET',
      headers,
    });
  }

  /**
   * Fetch single clinical screening detailed report.
   */
  async getScreeningDetail(screeningId, clinicUser = null) {
    const headers = {};
    if (clinicUser && clinicUser.id) {
      headers['X-Clinic-User-Id'] = clinicUser.id;
    }
    if (clinicUser && clinicUser.email) {
      headers['X-Clinic-User-Email'] = clinicUser.email;
    }

    return this._request(`/screenings/${screeningId}`, {
      method: 'GET',
      headers,
    });
  }
}

export const clinicApiService = new ClinicApiService();

