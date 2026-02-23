let currentImageIndex = 0;
let displayedFrames = [];
let displayedImages = [];

function setView(view) {
    const gallery = document.getElementById('gallery');
    const buttons = document.querySelectorAll('.view-btn');
    
    buttons.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    if (view === 'list') {
        gallery.classList.add('list-view');
    } else {
        gallery.classList.remove('list-view');
    }
}

function openModal(index) {
    currentImageIndex = index;
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const modalInfo = document.getElementById('modalInfo');
    
    const imageFilename = displayedImages[index];
    const frame = displayedFrames[index];
    
    const imageUrl = `/keyframes/${imageFilename}`;
    modalImage.src = imageUrl;
    
    const infoText = `
        Frame ${index + 1} of ${displayedImages.length}
        | Score: ${frame.final_score.toFixed(3)}
        | IQA: ${frame.w_iqa.toFixed(3)}
        | Face: ${frame.face_signal.toFixed(3)}
        | Emotion: ${frame.emotion_intensity.toFixed(3)}
        | Pose: ${frame.pose_signal.toFixed(3)}
    `;
    
    modalInfo.innerHTML = `
        <div class="modal-info-text">${infoText}</div>
        <div class="modal-actions">
            <button class="btn-enhance-modal" onclick="enhanceKeyframe('${imageFilename}', event)">
                Enhance Keyframe
            </button>
        </div>
    `;
    
    modal.style.display = 'block';
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    document.getElementById('imageModal').style.display = 'none';
    document.body.style.overflow = 'auto';
}

function changeImage(direction) {
    currentImageIndex = (currentImageIndex + direction + displayedImages.length) % displayedImages.length;
    openModal(currentImageIndex);
}

// Keyboard navigation
document.addEventListener('keydown', function(event) {
    const modal = document.getElementById('imageModal');
    if (modal.style.display === 'block') {
        if (event.key === 'ArrowLeft') changeImage(-1);
        if (event.key === 'ArrowRight') changeImage(1);
        if (event.key === 'Escape') closeModal();
    }
});

function downloadAll() {
    alert('Download all functionality would require server-side zip creation.');
}

// ============================================================================
// Open thumbnail editor with metadata
// ============================================================================
async function enhanceKeyframe(filename, event) {
    if (event) event.stopPropagation(); // Prevent modal from closing
    
    // Close the image modal first
    closeModal();
    
    // Try to load metadata for this keyframe
    // (You can expand this to fetch from your JSON API)
    const metadata = await loadMetadataForKeyframe(filename);
    
    // Open enhanced thumbnail editor with metadata
    openThumbnailEditor(filename, metadata);
}

async function loadMetadataForKeyframe(filename) {
    try {
        // Fetch metadata from your Flask API endpoint
        const res = await fetch(`/api/video-metadata/${filename}`);
        const data = await res.json();
        
        if (data.success) {
            console.log('Loaded metadata:', data.metadata);
            return data.metadata;
        } else {
            console.warn('No metadata found:', data.error);
            return null;
        }
    } catch (e) {
        console.error('Failed to load metadata:', e);
        return null;
    }
}

// ============================================================================
// Gallery rendering with enhance button in cards
// ============================================================================
const gallery = document.getElementById("gallery");

function renderGallery(data) {
    gallery.innerHTML = "";
    
    displayedFrames = data;
    displayedImages = data.map(item => item.filename);

    if (data.length === 0) {
        gallery.innerHTML = `
            <div class="empty-state">
                <p>No frames match your current filters</p>
            </div>
        `;
        return;
    }

    data.forEach((item, index) => {
        const card = document.createElement("div");
        card.className = "keyframe-card";

        card.innerHTML = `
            <div class="image-wrapper" onclick="openModal(${index})">
                <img src="/keyframes/${item.filename}" loading="lazy" alt="Frame ${index + 1}">
            </div>
            <div class="card-info">
                <div class="card-title">
                    Score: ${item.final_score.toFixed(3)}
                </div>
                <div class="card-meta">
                    IQA: ${item.w_iqa.toFixed(3)} |
                    Face: ${item.face_signal.toFixed(3)} |
                    Emotion: ${item.emotion_intensity.toFixed(3)} |
                    Pose: ${item.pose_signal.toFixed(3)}
                </div>
            </div>
        `;
        gallery.appendChild(card);
    });
}

function applySorting() {
    let sorted = [...keyframes];
    const sortBy = document.getElementById("sortBy").value;
    sorted.sort((a, b) => b[sortBy] - a[sortBy]);
    renderGallery(sorted);
}

document.getElementById("sortBy").addEventListener("change", applySorting);
applySorting();