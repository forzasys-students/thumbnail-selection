const eventSource = new EventSource('/stream');
const progressBar = document.getElementById('progressBar');
const stageTitle = document.getElementById('stageTitle');
const stageMessage = document.getElementById('stageMessage');

const stepMapping = {
    '[STEP 1]': 'step1',
    '[STEP 2]': 'step2',
    '[STEP 3]': 'step3',
    '[STEP 4]': 'step4'
};

eventSource.onmessage = function(event) {
    try {
        const data = JSON.parse(event.data);
        
        // Update progress bar
        progressBar.style.width = data.progress + '%';
        progressBar.textContent = data.progress + '%';
        
        // Update stage information
        stageTitle.textContent = data.stage || 'Processing...';
        stageMessage.textContent = data.message || '';
        
        // Update step indicators
        Object.keys(stepMapping).forEach(key => {
            const stepElement = document.getElementById(stepMapping[key]);
            if (data.stage && data.stage.includes(key)) {
                stepElement.classList.add('active');
            } else if (data.progress > getStepProgress(key)) {
                stepElement.classList.remove('active');
                stepElement.classList.add('completed');
            }
        });
        
        // Handle completion
        if (data.stage === 'Complete' || data.progress === 100) {
            document.querySelector('.spinner').style.display = 'none';
            document.getElementById('completeMessage').style.display = 'block';
            eventSource.close();
            
            // Mark all steps as completed
            Object.values(stepMapping).forEach(id => {
                document.getElementById(id).classList.add('completed');
            });
            
            // Redirect after 2 seconds
            setTimeout(() => {
                window.location.href = '/results';
            }, 2000);
        }
        
        // Handle errors
        if (data.stage === 'Error') {
            document.querySelector('.spinner').style.display = 'none';
            document.getElementById('errorMessage').style.display = 'block';
            document.querySelector('button').style.display = 'block';
            eventSource.close();
        }
        
    } catch (e) {
        console.error('Error parsing SSE data:', e);
    }
};

function getStepProgress(step) {
    const progressMap = {
        '[STEP 1]': 25,
        '[STEP 2]': 50,
        '[STEP 3]': 75,
        '[STEP 4]': 90
    };
    return progressMap[step] || 0;
}

eventSource.onerror = function() {
    console.error('SSE connection error');
};