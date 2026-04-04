document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const predictBtn = document.getElementById('predict-btn');
    const resetBtn = document.getElementById('reset-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsPanel = document.getElementById('results-panel');
    const uploadPanel = document.getElementById('upload-panel');

    let currentFile = null;

    // --- Drag and Drop Logic ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPEG, PNG, etc).');
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            imagePreviewContainer.classList.remove('hidden');
            resultsPanel.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    resetBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        dropZone.classList.remove('hidden');
        imagePreviewContainer.classList.add('hidden');
        resultsPanel.classList.add('hidden');
    });

    // --- Prediction API Logic ---
    predictBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI State
        predictBtn.disabled = true;
        uploadPanel.classList.add('hidden');
        loadingOverlay.classList.remove('hidden');
        resultsPanel.classList.add('hidden');

        const formData = new FormData();
        formData.append('image', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Server error');

            displayResults(data);
        } catch (error) {
            alert('Error making prediction: ' + error.message);
            uploadPanel.classList.remove('hidden');
            predictBtn.disabled = false;
        } finally {
            loadingOverlay.classList.add('hidden');
            predictBtn.disabled = false;
        }
    });

    function displayResults(data) {
        uploadPanel.classList.remove('hidden');
        resultsPanel.classList.remove('hidden');

        // Main Prediction
        document.getElementById('pred-label').textContent = data.prediction;
        const confPercent = (data.confidence * 100).toFixed(1);

        // Reset bars first to allow animation to replay
        document.querySelectorAll('.bar, .fill').forEach(el => el.style.width = '0%');

        // Use timeout to allow CSS transition to animate
        setTimeout(() => {
            document.getElementById('pred-conf-fill').style.width = `${confPercent}%`;
            document.getElementById('pred-conf-text').textContent = `${confPercent}% Confidence`;

            // Routing Level 1
            const pNat = (data.routing.p_natural * 100).toFixed(1);
            const pArt = (data.routing.p_artificial * 100).toFixed(1);
            document.getElementById('route-nat-bar').style.width = `${pNat}%`;
            document.getElementById('route-nat-text').textContent = `${pNat}%`;
            document.getElementById('route-art-bar').style.width = `${pArt}%`;
            document.getElementById('route-art-text').textContent = `${pArt}%`;

            // Routing Level 2 based on winner
            const isNat = data.routing.p_natural > data.routing.p_artificial;
            const weights = isNat ? data.routing.nat_weights : data.routing.art_weights;

            const wBase = (weights.base * 100).toFixed(1);
            const wSpec = (weights.specialized * 100).toFixed(1);

            document.getElementById('weight-base-bar').style.width = `${wBase}%`;
            document.getElementById('weight-base-text').textContent = `${wBase}%`;
            document.getElementById('weight-spec-bar').style.width = `${wSpec}%`;
            document.getElementById('weight-spec-text').textContent = `${wSpec}%`;

            // Text Summary
            const routeName = isNat ? 'Natural' : 'Artificial';
            const specName = isNat ? 'Natural Expert' : 'Artificial Expert';
            document.getElementById('routing-summary').innerHTML =
                `Level 1 routed this image to the <strong>${routeName} Domain</strong>. <br>
                 Level 2 decided to assign <strong>${wBase}%</strong> weight to the Base Expert and <strong>${wSpec}%</strong> weight to the specialized ${specName}.`;
        }, 50);
    }
});
