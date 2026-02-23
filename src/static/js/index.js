// Show filename when video is selected
document.getElementById('video').addEventListener('change', function (e) {
    const fileName = e.target.files[0]?.name;
    const label = document.getElementById('fileLabel');
    if (fileName) {
        label.textContent = `📁 ${fileName}`;
        label.style.color = '#667eea';
        label.style.fontWeight = '600';
    }
});

// Update FPS badge and slider fill colour as the user drags
function updateFps(value) {
    value = parseInt(value);

    const badge = document.getElementById('fpsValue');
    const slider = document.getElementById('fps');

    if (badge) {
        badge.textContent = `${value} fps`;
        badge.className = 'setting-badge';
        if (value > 12) badge.classList.add('hot');
        else if (value > 5) badge.classList.add('warn');
    }

    // Update the CSS gradient so the filled portion tracks the thumb
    if (slider) {
        const pct = ((value - 1) / (24 - 1)) * 100;
        slider.style.setProperty('--pct', pct + '%');
    }
}

// Initialise slider fill on page load
window.addEventListener('DOMContentLoaded', () => updateFps(document.getElementById('fps').value));