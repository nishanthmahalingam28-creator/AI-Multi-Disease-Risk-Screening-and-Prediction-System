/**
 * Patient Demographic Intake Form for Clinic Assisted Screening.
 * 
 * Captures patient identity conforming directly to the existing database
 * `users` entity structure (full_name, email, mobile, date_of_birth).
 * Separates patient personal identity from model prediction feature inputs.
 */

export function renderPatientForm(container, initialData = {}) {
  container.innerHTML = `
    <div class="patient-intake-card">
      <div class="form-section-title">
        <span>👤 Patient Identification (Users Schema Entity)</span>
      </div>
      <p class="form-hint" style="margin-bottom: 14px;">
        Record the patient record details to associate this clinic-assisted screening session.
      </p>

      <div class="form-grid">
        <div class="form-group">
          <label for="patient-fullname" class="form-label">Patient Full Name <span class="required">*</span></label>
          <input 
            type="text" 
            id="patient-fullname" 
            name="patientFullName" 
            class="form-control" 
            placeholder="e.g. Robert Johnson" 
            value="${initialData.fullName || ''}" 
            required 
          />
        </div>

        <div class="form-group">
          <label for="patient-email" class="form-label">Patient Email / MRN Identifier <span class="required">*</span></label>
          <input 
            type="email" 
            id="patient-email" 
            name="patientEmail" 
            class="form-control" 
            placeholder="patient.mrn@example.com" 
            value="${initialData.email || ''}" 
            required 
          />
        </div>

        <div class="form-group">
          <label for="patient-mobile" class="form-label">Mobile Contact Number</label>
          <input 
            type="tel" 
            id="patient-mobile" 
            name="patientMobile" 
            class="form-control" 
            placeholder="+1 555-0812" 
            value="${initialData.mobile || ''}" 
          />
        </div>

        <div class="form-group">
          <label for="patient-dob" class="form-label">Date of Birth</label>
          <input 
            type="date" 
            id="patient-dob" 
            name="patientDob" 
            class="form-control" 
            value="${initialData.dateOfBirth || ''}" 
          />
        </div>
      </div>
    </div>
  `;
}

export function validateAndExtractPatientInfo(container) {
  const nameInput = container.querySelector('#patient-fullname');
  const emailInput = container.querySelector('#patient-email');
  const mobileInput = container.querySelector('#patient-mobile');
  const dobInput = container.querySelector('#patient-dob');

  const fullName = nameInput ? nameInput.value.trim() : '';
  const email = emailInput ? emailInput.value.trim() : '';
  const mobile = mobileInput ? mobileInput.value.trim() : '';
  const dateOfBirth = dobInput ? dobInput.value : '';

  if (!fullName || fullName.length < 2) {
    throw new Error('Please enter the patient’s full name (minimum 2 characters).');
  }

  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    throw new Error('Please enter a valid patient email address or MRN identifier.');
  }

  return {
    fullName,
    email,
    mobile,
    dateOfBirth,
  };
}
