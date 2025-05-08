/**
 * AutoVerse - Main Application JavaScript
 * Handles UI interactions, AJAX calls, and application state
 * 
 * Note on WaveSurfer.js:
 * This application uses WaveSurfer.js for audio visualization.
 * We use WaveSurfer's plugin system with the registerPlugin method to add timeline features.
 * The timeline plugin is added after each WaveSurfer instance is created.
 */

// Global variables for application state
let wavesurferInstrumental = null;
let wavesurferVocals = null;
let wavesurferMixed = null;
let currentlyPlaying = null;
let appState = {
    textUploaded: false,
    instrumentalUploaded: false,
    vocalsGenerated: false,
    vocalsSynchronized: false,
    mixCreated: false
};

// Initialize the application when the DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    initializeWaveSurfers();
    setupEventListeners();
    updateButtonStates();
});

// Initialize WaveSurfer instances
function initializeWaveSurfers() {
    // Create WaveSurfer instances with appropriate configuration
    wavesurferInstrumental = WaveSurfer.create({
        container: '#waveformInstrumental',
        waveColor: '#6366F1',
        progressColor: '#EC4899',
        cursorColor: '#10B981',
        cursorWidth: 2,
        height: 100,
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        responsive: true,
        normalize: true,
        backgroundColor: 'transparent',

    });
    
    // Add timeline to instrumental waveform
    wavesurferInstrumental.registerPlugin(
        WaveSurfer.timeline.create({
            container: '#timelineInstrumental',
            primaryColor: '#6B7280',
            secondaryColor: '#4B5563',
            primaryFontColor: '#F9FAFB',
            secondaryFontColor: '#E5E7EB',
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: 11
        })
    );

    // Create vocals waveform
    wavesurferVocals = WaveSurfer.create({
        container: '#waveformVocals',
        waveColor: '#EC4899',
        progressColor: '#10B981',
        cursorColor: '#10B981',
        cursorWidth: 2,
        height: 100,
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        responsive: true,
        normalize: true,
        backgroundColor: 'transparent'
    });
    
    // Add timeline to vocals waveform
    wavesurferVocals.registerPlugin(
        WaveSurfer.timeline.create({
            container: '#timelineVocals',
            primaryColor: '#6B7280',
            secondaryColor: '#4B5563',
            primaryFontColor: '#F9FAFB',
            secondaryFontColor: '#E5E7EB',
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: 11
        })
    );

    // Create mixed waveform
    wavesurferMixed = WaveSurfer.create({
        container: '#waveformMixed',
        waveColor: '#10B981',
        progressColor: '#6366F1',
        cursorColor: '#10B981',
        cursorWidth: 2,
        height: 100,
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        responsive: true,
        normalize: true,
        backgroundColor: 'transparent'
    });
    
    // Add timeline to mixed waveform
    wavesurferMixed.registerPlugin(
        WaveSurfer.timeline.create({
            container: '#timelineMixed',
            primaryColor: '#6B7280',
            secondaryColor: '#4B5563',
            primaryFontColor: '#F9FAFB',
            secondaryFontColor: '#E5E7EB',
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: 11
        })
    );

    // Set up waveform event listeners
    wavesurferInstrumental.on('ready', updatePlaybackTime);
    wavesurferVocals.on('ready', updatePlaybackTime);
    wavesurferMixed.on('ready', updatePlaybackTime);

    wavesurferInstrumental.on('audioprocess', updatePlaybackTime);
    wavesurferVocals.on('audioprocess', updatePlaybackTime);
    wavesurferMixed.on('audioprocess', updatePlaybackTime);

    wavesurferInstrumental.on('finish', () => {
        stopAllPlayback();
    });
    wavesurferVocals.on('finish', () => {
        stopAllPlayback();
    });
    wavesurferMixed.on('finish', () => {
        stopAllPlayback();
    });
}

// Setup all event listeners
function setupEventListeners() {
    // Text upload handling
    setupTextUpload();
    
    // Instrumental upload handling
    setupInstrumentalUpload();
    
    // Voice generation handling
    setupVoiceGeneration();
    
    // Synchronization handling
    setupSynchronization();
    
    // Mixing handling
    setupMixing();
    
    // Playback controls
    setupPlaybackControls();
    
    // Download buttons
    setupDownloadButtons();
}

// Set up text upload event listeners
function setupTextUpload() {
    const textDropArea = document.getElementById('textDropArea');
    const textFileInput = document.getElementById('textFileInput');
    const pasteTextBtn = document.getElementById('pasteTextBtn');
    const textInputModal = new bootstrap.Modal(document.getElementById('textInputModal'));
    const confirmTextBtn = document.getElementById('confirmTextBtn');
    const saveTextBtn = document.getElementById('saveTextBtn');
    const editTextBtn = document.getElementById('editTextBtn');
    const textEditor = document.getElementById('textEditor');
    const textInputArea = document.getElementById('textInputArea');
    
    // Drag and drop for text files
    textDropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        textDropArea.classList.add('dragover');
    });
    
    textDropArea.addEventListener('dragleave', () => {
        textDropArea.classList.remove('dragover');
    });
    
    textDropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        textDropArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            handleTextFileUpload(e.dataTransfer.files[0]);
        }
    });
    
    // Click to select file
    textDropArea.addEventListener('click', () => {
        textFileInput.click();
    });
    
    textFileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleTextFileUpload(e.target.files[0]);
        }
    });
    
    // Paste text button
    pasteTextBtn.addEventListener('click', () => {
        textInputArea.value = ''; // Clear previous text
        textInputModal.show();
    });
    
    // Confirm button in paste text modal
    confirmTextBtn.addEventListener('click', () => {
        const text = textInputArea.value.trim();
        if (text) {
            handlePastedText(text);
        }
        textInputModal.hide();
    });
    
    // Save button in text editor
    saveTextBtn.addEventListener('click', () => {
        const text = textEditor.value.trim();
        if (text) {
            handlePastedText(text);
            toggleTextEditor(false);
        }
    });
    
    // Edit text button
    editTextBtn.addEventListener('click', () => {
        toggleTextEditor(true);
    });
}

// Handle text file upload
function handleTextFileUpload(file) {
    if (file.type !== 'text/plain') {
        showError('Please upload a .txt file');
        return;
    }
    
    showLoading('Uploading text file...');
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/upload/text', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.error) {
            showError(data.error);
            return;
        }
        
        // Update the UI to show the file info
        document.getElementById('textUploadContainer').classList.add('d-none');
        document.getElementById('textFileInfo').classList.remove('d-none');
        document.getElementById('textFileName').textContent = data.filename;
        document.getElementById('textWordCount').textContent = `${data.wordCount} words`;
        document.getElementById('textCharCount').textContent = `${data.charCount} characters`;
        
        // Update app state
        appState.textUploaded = true;
        updateButtonStates();
        
        // Store text in editor for potential editing
        fetch('/upload/text', {
            method: 'GET'
        })
        .then(response => response.text())
        .then(text => {
            document.getElementById('textEditor').value = text;
        })
        .catch(error => {
            console.error('Error fetching text content:', error);
        });
    })
    .catch(error => {
        hideLoading();
        showError('Error uploading text file: ' + error.message);
    });
}

// Handle pasted text
function handlePastedText(text) {
    showLoading('Processing text...');
    
    const formData = new FormData();
    formData.append('text', text);
    
    fetch('/upload/text', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.error) {
            showError(data.error);
            return;
        }
        
        // Update the UI to show the text info
        document.getElementById('textUploadContainer').classList.add('d-none');
        document.getElementById('textFileInfo').classList.remove('d-none');
        document.getElementById('textFileName').textContent = 'Text Input';
        document.getElementById('textWordCount').textContent = `${data.wordCount} words`;
        document.getElementById('textCharCount').textContent = `${data.charCount} characters`;
        
        // Store text in editor for potential editing
        document.getElementById('textEditor').value = text;
        
        // Update app state
        appState.textUploaded = true;
        updateButtonStates();
    })
    .catch(error => {
        hideLoading();
        showError('Error processing text: ' + error.message);
    });
}

// Toggle text editor visibility
function toggleTextEditor(show) {
    const editorContainer = document.getElementById('textEditorContainer');
    const fileInfo = document.getElementById('textFileInfo');
    
    if (show) {
        editorContainer.classList.remove('d-none');
        fileInfo.classList.add('d-none');
    } else {
        editorContainer.classList.add('d-none');
        fileInfo.classList.remove('d-none');
    }
}

// Set up instrumental upload event listeners
function setupInstrumentalUpload() {
    const instrumentalDropArea = document.getElementById('instrumentalDropArea');
    const instrumentalFileInput = document.getElementById('instrumentalFileInput');
    const removeInstrumentalBtn = document.getElementById('removeInstrumentalBtn');
    
    // Drag and drop for instrumental files
    instrumentalDropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        instrumentalDropArea.classList.add('dragover');
    });
    
    instrumentalDropArea.addEventListener('dragleave', () => {
        instrumentalDropArea.classList.remove('dragover');
    });
    
    instrumentalDropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        instrumentalDropArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            handleInstrumentalUpload(e.dataTransfer.files[0]);
        }
    });
    
    // Click to select file
    instrumentalDropArea.addEventListener('click', () => {
        instrumentalFileInput.click();
    });
    
    instrumentalFileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleInstrumentalUpload(e.target.files[0]);
        }
    });
    
    // Remove instrumental button
    removeInstrumentalBtn.addEventListener('click', () => {
        document.getElementById('instrumentalUploadContainer').classList.remove('d-none');
        document.getElementById('instrumentalFileInfo').classList.add('d-none');
        
        // Clear waveform
        wavesurferInstrumental.empty();
        
        // Update app state
        appState.instrumentalUploaded = false;
        updateButtonStates();
    });
}

// Handle instrumental file upload
function handleInstrumentalUpload(file) {
    if (!file.type.includes('audio')) {
        showError('Please upload an MP3 or WAV file');
        return;
    }
    
    showLoading('Uploading and analyzing instrumental...');
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/upload/instrumental', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            hideLoading();
            showError(data.error);
            return;
        }
        
        // Update the UI to show the file info
        document.getElementById('instrumentalUploadContainer').classList.add('d-none');
        document.getElementById('instrumentalFileInfo').classList.remove('d-none');
        document.getElementById('instrumentalFileName').textContent = data.filename;
        document.getElementById('instrumentalDuration').textContent = formatTime(data.duration);
        document.getElementById('instrumentalBPM').textContent = `${Math.round(data.bpm)} BPM`;
        
        // Load waveform
        wavesurferInstrumental.load(`/get-file/instrumental`);
        wavesurferInstrumental.on('ready', () => {
            hideLoading();
            
            // Update app state
            appState.instrumentalUploaded = true;
            updateButtonStates();
        });
    })
    .catch(error => {
        hideLoading();
        showError('Error uploading instrumental: ' + error.message);
    });
}

// Set up voice generation event listeners
function setupVoiceGeneration() {
    const generateVocalsBtn = document.getElementById('generateVocalsBtn');
    
    generateVocalsBtn.addEventListener('click', () => {
        const voiceId = document.getElementById('voiceSelect').value;
        const style = document.getElementById('styleSelect').value;
        
        showLoading(`Generating ${style} vocals...`);
        
        const formData = new FormData();
        formData.append('voice', voiceId);
        formData.append('style', style);
        
        fetch('/generate-vocals', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                hideLoading();
                showError(data.error);
                return;
            }
            
            // Load vocals waveform
            wavesurferVocals.load(`/get-file/vocals`);
            wavesurferVocals.on('ready', () => {
                hideLoading();
                
                // Update app state
                appState.vocalsGenerated = true;
                updateButtonStates();
                
                // Scroll to waveform
                document.getElementById('waveformVocals').scrollIntoView({ behavior: 'smooth' });
                
                // Update status
                setStatus('Vocals generated successfully', 'success');
            });
        })
        .catch(error => {
            hideLoading();
            showError('Error generating vocals: ' + error.message);
        });
    });
}

// Set up synchronization event listeners
function setupSynchronization() {
    const synchronizeBtn = document.getElementById('synchronizeBtn');
    
    synchronizeBtn.addEventListener('click', () => {
        const alignmentMode = document.querySelector('input[name="alignmentMode"]:checked').value;
        const timeStretch = document.getElementById('timeStretchToggle').checked;
        
        showLoading('Synchronizing audio...');
        
        const formData = new FormData();
        formData.append('alignmentMode', alignmentMode);
        formData.append('timeStretch', timeStretch);
        
        fetch('/synchronize', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                hideLoading();
                showError(data.error);
                return;
            }
            
            // Load synced vocals into the vocals waveform
            wavesurferVocals.load(`/get-file/synced`);
            wavesurferVocals.on('ready', () => {
                hideLoading();
                
                // Update app state
                appState.vocalsSynchronized = true;
                updateButtonStates();
                
                // Update status
                setStatus('Vocals synchronized successfully', 'success');
            });
        })
        .catch(error => {
            hideLoading();
            showError('Error synchronizing audio: ' + error.message);
        });
    });
}

// Set up mixing event listeners
function setupMixing() {
    const mixAudioBtn = document.getElementById('mixAudioBtn');
    
    mixAudioBtn.addEventListener('click', () => {
        // Get mixing parameters
        const vocalLevel = document.getElementById('vocalLevelSlider').value / 100;
        const eqPreset = document.getElementById('eqPresetSelect').value;
        const compression = document.getElementById('compressionSelect').value;
        const stereoWidth = document.getElementById('stereoWidthSlider').value / 100;
        const autoDucking = document.getElementById('autoDuckingToggle').checked;
        
        showLoading('Mixing audio...');
        
        const formData = new FormData();
        formData.append('vocalLevel', vocalLevel);
        formData.append('eqPreset', eqPreset);
        formData.append('compression', compression);
        formData.append('stereoWidth', stereoWidth);
        formData.append('autoDucking', autoDucking);
        
        fetch('/mix', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                hideLoading();
                showError(data.error);
                return;
            }
            
            // Load mixed audio waveform
            wavesurferMixed.load(`/get-file/final`);
            wavesurferMixed.on('ready', () => {
                hideLoading();
                
                // Update app state
                appState.mixCreated = true;
                updateButtonStates();
                
                // Scroll to mixed waveform
                document.getElementById('waveformMixed').scrollIntoView({ behavior: 'smooth' });
                
                // Update status
                setStatus('Mix created successfully', 'success');
            });
        })
        .catch(error => {
            hideLoading();
            showError('Error mixing audio: ' + error.message);
        });
    });
    
    // Update slider value displays
    const vocalLevelSlider = document.getElementById('vocalLevelSlider');
    const vocalLevelValue = document.getElementById('vocalLevelValue');
    vocalLevelSlider.addEventListener('input', () => {
        vocalLevelValue.textContent = `${vocalLevelSlider.value}%`;
    });
    
    const stereoWidthSlider = document.getElementById('stereoWidthSlider');
    const stereoWidthValue = document.getElementById('stereoWidthValue');
    stereoWidthSlider.addEventListener('input', () => {
        stereoWidthValue.textContent = `${stereoWidthSlider.value}%`;
    });
}

// Set up playback control event listeners
function setupPlaybackControls() {
    const playInstrumentalBtn = document.getElementById('playInstrumentalBtn');
    const playVocalsBtn = document.getElementById('playVocalsBtn');
    const playMixedBtn = document.getElementById('playMixedBtn');
    const stopBtn = document.getElementById('stopBtn');
    
    playInstrumentalBtn.addEventListener('click', () => {
        stopAllPlayback();
        wavesurferInstrumental.play();
        currentlyPlaying = 'instrumental';
        updatePlayButtonStates();
    });
    
    playVocalsBtn.addEventListener('click', () => {
        stopAllPlayback();
        wavesurferVocals.play();
        currentlyPlaying = 'vocals';
        updatePlayButtonStates();
    });
    
    playMixedBtn.addEventListener('click', () => {
        stopAllPlayback();
        wavesurferMixed.play();
        currentlyPlaying = 'mixed';
        updatePlayButtonStates();
    });
    
    stopBtn.addEventListener('click', () => {
        stopAllPlayback();
    });
}

// Set up download buttons
function setupDownloadButtons() {
    const downloadVocalsBtn = document.getElementById('downloadVocalsBtn');
    const downloadSyncedVocalsBtn = document.getElementById('downloadSyncedVocalsBtn');
    const downloadMixBtn = document.getElementById('downloadMixBtn');
    
    downloadVocalsBtn.addEventListener('click', () => {
        window.location.href = '/download/vocals';
    });
    
    downloadSyncedVocalsBtn.addEventListener('click', () => {
        window.location.href = '/download/synced';
    });
    
    downloadMixBtn.addEventListener('click', () => {
        window.location.href = '/download/final';
    });
}

// Update UI button states based on application state
function updateButtonStates() {
    // Generate vocals button
    const generateVocalsBtn = document.getElementById('generateVocalsBtn');
    generateVocalsBtn.disabled = !(appState.textUploaded && appState.instrumentalUploaded);
    
    // Synchronize button
    const synchronizeBtn = document.getElementById('synchronizeBtn');
    synchronizeBtn.disabled = !appState.vocalsGenerated;
    
    // Mix audio button
    const mixAudioBtn = document.getElementById('mixAudioBtn');
    mixAudioBtn.disabled = !appState.vocalsSynchronized;
    
    // Playback buttons
    const playInstrumentalBtn = document.getElementById('playInstrumentalBtn');
    const playVocalsBtn = document.getElementById('playVocalsBtn');
    const playMixedBtn = document.getElementById('playMixedBtn');
    
    playInstrumentalBtn.disabled = !appState.instrumentalUploaded;
    playVocalsBtn.disabled = !appState.vocalsGenerated;
    playMixedBtn.disabled = !appState.mixCreated;
    
    // Download buttons
    const downloadVocalsBtn = document.getElementById('downloadVocalsBtn');
    const downloadSyncedVocalsBtn = document.getElementById('downloadSyncedVocalsBtn');
    const downloadMixBtn = document.getElementById('downloadMixBtn');
    
    downloadVocalsBtn.disabled = !appState.vocalsGenerated;
    downloadSyncedVocalsBtn.disabled = !appState.vocalsSynchronized;
    downloadMixBtn.disabled = !appState.mixCreated;
}

// Update play button states
function updatePlayButtonStates() {
    const playInstrumentalBtn = document.getElementById('playInstrumentalBtn');
    const playVocalsBtn = document.getElementById('playVocalsBtn');
    const playMixedBtn = document.getElementById('playMixedBtn');
    
    playInstrumentalBtn.innerHTML = currentlyPlaying === 'instrumental' 
        ? '<i class="fas fa-pause"></i>' 
        : '<i class="fas fa-music"></i>';
    
    playVocalsBtn.innerHTML = currentlyPlaying === 'vocals' 
        ? '<i class="fas fa-pause"></i>' 
        : '<i class="fas fa-microphone"></i>';
    
    playMixedBtn.innerHTML = currentlyPlaying === 'mixed' 
        ? '<i class="fas fa-pause"></i>' 
        : '<i class="fas fa-headphones"></i>';
}

// Stop all playback
function stopAllPlayback() {
    wavesurferInstrumental.pause();
    wavesurferVocals.pause();
    wavesurferMixed.pause();
    currentlyPlaying = null;
    updatePlayButtonStates();
}

// Update playback time display
function updatePlaybackTime() {
    let currentTime = 0;
    let totalTime = 0;
    
    if (currentlyPlaying === 'instrumental') {
        currentTime = wavesurferInstrumental.getCurrentTime();
        totalTime = wavesurferInstrumental.getDuration();
    } else if (currentlyPlaying === 'vocals') {
        currentTime = wavesurferVocals.getCurrentTime();
        totalTime = wavesurferVocals.getDuration();
    } else if (currentlyPlaying === 'mixed') {
        currentTime = wavesurferMixed.getCurrentTime();
        totalTime = wavesurferMixed.getDuration();
    }
    
    document.getElementById('currentTime').textContent = formatTime(currentTime);
    document.getElementById('totalTime').textContent = formatTime(totalTime);
}

// Format time in seconds to MM:SS format
function formatTime(seconds) {
    if (isNaN(seconds) || seconds === Infinity) return '0:00';
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Show loading modal
function showLoading(message) {
    document.getElementById('loadingMessage').textContent = message || 'Processing...';
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    loadingModal.show();
    setStatus('Processing...', 'processing');
}

// Hide loading modal
function hideLoading() {
    const loadingModalEl = document.getElementById('loadingModal');
    const loadingModal = bootstrap.Modal.getInstance(loadingModalEl);
    if (loadingModal) {
        loadingModal.hide();
    }
    setStatus('Ready');
}

// Show error modal
function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    const errorModal = new bootstrap.Modal(document.getElementById('errorModal'));
    errorModal.show();
    setStatus('Error', 'error');
}

// Set status indicator
function setStatus(message, type) {
    const statusIndicator = document.getElementById('statusIndicator');
    statusIndicator.textContent = message;
    
    // Remove all status classes
    statusIndicator.classList.remove('processing', 'success', 'error');
    
    // Add appropriate class
    if (type) {
        statusIndicator.classList.add(type);
    }
}
