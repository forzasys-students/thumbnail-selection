let currentImageIndex = 0;

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
    
    // Build the URL using the Flask route
    const imageUrl = `/keyframes/${images[index]}`;
    modalImage.src = imageUrl;
    modalInfo.textContent = `Frame ${index + 1} of ${images.length} - ${images[index]}`;
    modal.style.display = 'block';
    
    // Prevent body scroll
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    document.getElementById('imageModal').style.display = 'none';
    document.body.style.overflow = 'auto';
}

function changeImage(direction) {
    currentImageIndex = (currentImageIndex + direction + images.length) % images.length;
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