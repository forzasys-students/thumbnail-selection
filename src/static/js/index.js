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
