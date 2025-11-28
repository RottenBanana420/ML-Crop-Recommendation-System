// Main JavaScript - Core functionality

// Toast notification system
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span style="font-size: 1.5rem;">${type === 'success' ? '✓' : '✗'}</span>
            <div>
                <strong>${type === 'success' ? 'Success' : 'Error'}</strong>
                <p style="margin: 0; font-size: 0.875rem;">${message}</p>
            </div>
        </div>
    `;

    document.body.appendChild(toast);

    // Remove after 5 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// Utility: Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Crop Recommendation System initialized');
});
