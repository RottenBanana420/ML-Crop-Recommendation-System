// Form Validation and Prediction

// Validation ranges (must match backend config)
const VALIDATION_RANGES = {
    N: { min: 0, max: 140, unit: 'kg/ha' },
    P: { min: 5, max: 145, unit: 'kg/ha' },
    K: { min: 5, max: 205, unit: 'kg/ha' },
    temperature: { min: 8.8, max: 43.7, unit: '°C' },
    humidity: { min: 14.3, max: 99.9, unit: '%' },
    ph: { min: 3.5, max: 9.9, unit: 'pH' },
    rainfall: { min: 20.2, max: 298.6, unit: 'mm' }
};

// Validate single field
function validateField(fieldName, value) {
    const range = VALIDATION_RANGES[fieldName];
    if (!range) return null;

    const numValue = parseFloat(value);

    if (isNaN(numValue)) {
        return `${fieldName} must be a number`;
    }

    if (numValue < range.min || numValue > range.max) {
        return `${fieldName} must be between ${range.min} and ${range.max} ${range.unit}`;
    }

    return null;
}

// Update field UI based on validation
function updateFieldUI(fieldName, error) {
    const input = document.getElementById(fieldName);
    const errorSpan = document.getElementById(`${fieldName}-error`);

    if (error) {
        input.classList.add('invalid');
        input.classList.remove('valid');
        errorSpan.textContent = error;
        errorSpan.style.display = 'block';
    } else {
        input.classList.remove('invalid');
        input.classList.add('valid');
        errorSpan.textContent = '';
        errorSpan.style.display = 'none';
    }
}

// Validate all fields
function validateForm() {
    let isValid = true;
    const errors = {};

    for (const fieldName in VALIDATION_RANGES) {
        const input = document.getElementById(fieldName);
        if (input) {
            const error = validateField(fieldName, input.value);
            if (error) {
                isValid = false;
                errors[fieldName] = error;
            }
            updateFieldUI(fieldName, error);
        }
    }

    return { isValid, errors };
}

// Submit prediction
async function submitPrediction(formData) {
    try {
        const response = await fetch('/predict/api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || 'Prediction failed');
        }

        return data;
    } catch (error) {
        console.error('Prediction error:', error);
        throw error;
    }
}

// Display results
function displayResults(result) {
    const resultsSection = document.getElementById('results-section');
    const cropName = document.getElementById('crop-name');
    const confidence = document.getElementById('confidence');
    const alternativesList = document.getElementById('alternatives-list');

    // Show results section
    resultsSection.classList.remove('hidden');

    // Display main prediction
    cropName.textContent = result.crop;
    confidence.textContent = `${result.confidence_percent}%`;

    // Display alternatives
    alternativesList.innerHTML = '';
    if (result.alternatives && result.alternatives.length > 0) {
        result.alternatives.forEach(alt => {
            const altCard = document.createElement('div');
            altCard.className = 'card';
            altCard.innerHTML = `
                <h4 style="color: var(--primary-green); margin-bottom: var(--space-2);">${alt.crop}</h4>
                <p style="margin: 0; color: var(--text-secondary);">${alt.confidence_percent}% confidence</p>
            `;
            alternativesList.appendChild(altCard);
        });
    }

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Initialize form
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    if (!form) return;

    // Add real-time validation
    for (const fieldName in VALIDATION_RANGES) {
        const input = document.getElementById(fieldName);
        if (input) {
            input.addEventListener('input', debounce(() => {
                const error = validateField(fieldName, input.value);
                updateFieldUI(fieldName, error);
            }, 300));
        }
    }

    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Validate form
        const validation = validateForm();
        if (!validation.isValid) {
            showToast('Please fix the errors in the form', 'error');
            return;
        }

        // Collect form data
        const formData = {};
        for (const fieldName in VALIDATION_RANGES) {
            const input = document.getElementById(fieldName);
            if (input) {
                formData[fieldName] = parseFloat(input.value);
            }
        }

        // Disable submit button
        const submitBtn = document.getElementById('submit-btn');
        const originalText = submitBtn.textContent;
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner"></span> Predicting...';

        try {
            // Submit prediction
            const response = await submitPrediction(formData);

            if (response.success) {
                displayResults(response.result);
                showToast('Prediction generated successfully!', 'success');
            } else {
                throw new Error(response.message || 'Prediction failed');
            }
        } catch (error) {
            showToast(error.message, 'error');
            console.error('Prediction error:', error);
        } finally {
            // Re-enable submit button
            submitBtn.disabled = false;
            submitBtn.textContent = originalText;
        }
    });
});
