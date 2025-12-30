// Show filename when video is selected
document.getElementById('video').addEventListener('change', function(e) {
    const fileName = e.target.files[0]?.name;
    if (fileName) {
        const wrapper = e.target.parentElement;
        wrapper.setAttribute('data-file', fileName);
        
        // Update the before content to show filename
        const style = document.createElement('style');
        style.textContent = `
            .file-input-wrapper[data-file]::before {
                content: '📁 ' attr(data-file) !important;
                color: #667eea;
                font-weight: 600;
            }
        `;
        document.head.appendChild(style);
    }
});