let currentImageIndex = 0;
let displayedFrames = []; // Track currently displayed metadata
let displayedImages = []; // Track currently displayed image filenames

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
    
    // Use the currently displayed images array
    const imageFilename = displayedImages[index];
    const frame = displayedFrames[index];
    
    // Build the URL using the Flask route
    const imageUrl = `/keyframes/${imageFilename}`;
    modalImage.src = imageUrl;
    
    // Display frame info with metadata
    const infoText = `
        Frame ${index + 1} of ${displayedImages.length} - ${imageFilename}
        | Score: ${frame.final_score.toFixed(3)}
        | IQA: ${frame.w_iqa.toFixed(3)}
        | Face: ${frame.face_signal.toFixed(3)}
        | Emotion: ${frame.emotion_intensity.toFixed(3)}
        | Pose: ${frame.pose_signal.toFixed(3)}
    `;
    modalInfo.textContent = infoText;
    modal.style.display = 'block';
    
    // Prevent body scroll
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
    alert('Download all functionality would require server-side zip creation. Implementation depends on your requirements.');
    // TODO: Implement actual download functionality
    // Maybe create a Flask route that zips all images and sends them
}

const gallery = document.getElementById("gallery");

function renderGallery(data) {
    gallery.innerHTML = "";
    
    // Update the displayed frames and images for modal navigation
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
        card.onclick = () => openModal(index);

        card.innerHTML = `
            <div class="image-wrapper">
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
    // Create a copy of the original keyframes
    let sorted = [...keyframes];
    
    // Get the selected sort option
    const sortBy = document.getElementById("sortBy").value;
    
    // Sort in descending order (highest first)
    sorted.sort((a, b) => b[sortBy] - a[sortBy]);
    
    // Render the sorted gallery
    renderGallery(sorted);
}

// Add event listener for sort dropdown
document.getElementById("sortBy").addEventListener("change", applySorting);

// Initial render with default sorting (final_score)
applySorting();