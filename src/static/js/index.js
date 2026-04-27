/* ──────────────────────────────────────────
   File label
   ────────────────────────────────────────── */
document.getElementById('video').addEventListener('change', function (e) {
    const fileName = e.target.files[0]?.name;
    const label    = document.getElementById('fileLabel');
    const labelTxt = document.getElementById('fileLabelText');
    if (fileName) {
        labelTxt.textContent = fileName;
        label.classList.add('selected');
    }
});


/* ──────────────────────────────────────────
   FPS slider
   ────────────────────────────────────────── */
function updateFps(value) {
    value = parseInt(value);

    const badge  = document.getElementById('fpsValue');
    const slider = document.getElementById('fps');

    if (badge) {
        badge.textContent = `${value} fps`;
        badge.className   = 'setting-badge';
        if      (value > 16) badge.classList.add('hot');
        else if (value > 8)  badge.classList.add('warn');
    }

    if (slider) {
        const pct = ((value - 1) / (24 - 1)) * 100;
        slider.style.setProperty('--pct', pct + '%');
    }
}

window.addEventListener('DOMContentLoaded', () => {
    const fps = document.getElementById('fps');
    if (fps) updateFps(fps.value);
});


/* ──────────────────────────────────────────
   Weight total tracker
   ────────────────────────────────────────── */
const WEIGHT_IDS = ['w_face', 'w_emotion', 'w_pose', 'w_iqa'];

function updateWeightTotal() {
    let total = 0;

    for (const id of WEIGHT_IDS) {
        const el = document.getElementById(id);
        if (el) {
            const v = parseFloat(el.value);
            if (!isNaN(v)) total += v;
        }
    }

    const display = document.getElementById('weightTotal');
    const status  = document.getElementById('weightStatus');

    if (!display || !status) return;

    display.textContent = total.toFixed(2);

    // visual feedback
    if (Math.abs(total - 1.0) < 0.001) {
        status.textContent = '✓';
        status.className   = 'weight-status';
    } else if (total > 1.0) {
        status.textContent = '↑';
        status.className   = 'weight-status error';
    } else {
        status.textContent = '↓';
        status.className   = 'weight-status warn';
    }
}

/* Initialise on load */
window.addEventListener('DOMContentLoaded', updateWeightTotal);