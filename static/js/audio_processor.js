/**
 * AudioVerse - Audio Processing JavaScript
 * Handles client-side audio visualization and processing features
 */

// Global variables for audio visualization
let currentAudioContext = null;
let currentAnalyser = null;
let visualizationFrameId = null;

// Initialize audio processor when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // The visualization features will be initialized when needed
});

/**
 * Create audio context for visualization and analysis
 */
function createAudioContext() {
    // Check if we already have an audio context
    if (currentAudioContext) {
        return currentAudioContext;
    }
    
    // Create a new audio context
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    currentAudioContext = new AudioContext();
    return currentAudioContext;
}

/**
 * Connect audio element to analyzer for visualization
 * @param {HTMLAudioElement} audioElement - The audio element to analyze
 */
function connectAnalyser(audioElement) {
    const audioContext = createAudioContext();
    
    // Disconnect previous analyzer if exists
    if (currentAnalyser) {
        currentAnalyser.disconnect();
    }
    
    // Create a new analyzer
    currentAnalyser = audioContext.createAnalyser();
    currentAnalyser.fftSize = 2048;
    
    // Connect audio element to analyzer
    const source = audioContext.createMediaElementSource(audioElement);
    source.connect(currentAnalyser);
    currentAnalyser.connect(audioContext.destination);
    
    return currentAnalyser;
}

/**
 * Visualize audio waveform on canvas
 * @param {HTMLCanvasElement} canvas - Canvas element for visualization
 * @param {AnalyserNode} analyser - Audio analyzer node
 * @param {string} color - Waveform color
 */
function visualizeWaveform(canvas, analyser, color = '#6366F1') {
    // Cancel any ongoing animation
    if (visualizationFrameId) {
        cancelAnimationFrame(visualizationFrameId);
    }
    
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    function draw() {
        visualizationFrameId = requestAnimationFrame(draw);
        
        // Get waveform data
        analyser.getByteTimeDomainData(dataArray);
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw waveform
        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.beginPath();
        
        const sliceWidth = canvas.width / bufferLength;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            
            x += sliceWidth;
        }
        
        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    }
    
    draw();
}

/**
 * Visualize audio frequency spectrum on canvas
 * @param {HTMLCanvasElement} canvas - Canvas element for visualization
 * @param {AnalyserNode} analyser - Audio analyzer node
 * @param {string} color - Spectrum color
 */
function visualizeSpectrum(canvas, analyser, color = '#EC4899') {
    // Cancel any ongoing animation
    if (visualizationFrameId) {
        cancelAnimationFrame(visualizationFrameId);
    }
    
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    function draw() {
        visualizationFrameId = requestAnimationFrame(draw);
        
        // Get frequency data
        analyser.getByteFrequencyData(dataArray);
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw spectrum
        const barWidth = (canvas.width / bufferLength) * 2.5;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * canvas.height;
            
            // Create gradient
            const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
            gradient.addColorStop(0, '#10B981');
            gradient.addColorStop(0.5, color);
            gradient.addColorStop(1, '#6366F1');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
            
            x += barWidth + 1;
        }
    }
    
    draw();
}

/**
 * Detect beats in audio using threshold algorithm
 * @param {AnalyserNode} analyser - Audio analyzer node
 * @param {number} threshold - Energy threshold for beat detection
 * @param {Function} onBeat - Callback function when beat is detected
 */
function detectBeats(analyser, threshold = 0.15, onBeat) {
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    // Variables for beat detection
    let lastEnergy = 0;
    let lastBeatTime = 0;
    
    function checkBeat() {
        requestAnimationFrame(checkBeat);
        
        // Get frequency data
        analyser.getByteFrequencyData(dataArray);
        
        // Calculate energy in low-frequency range (bass)
        let energy = 0;
        const lowRange = bufferLength * 0.1; // Use first 10% of frequency range
        
        for (let i = 0; i < lowRange; i++) {
            energy += dataArray[i];
        }
        
        energy = energy / lowRange;
        
        // Normalize between 0 and 1
        energy = energy / 255;
        
        // Detect beat if energy increases significantly
        const now = performance.now();
        const timeSinceLastBeat = now - lastBeatTime;
        
        // Only check for beats after a minimum interval (100ms)
        if (timeSinceLastBeat > 100) {
            if (energy > lastEnergy * (1 + threshold) && energy > threshold) {
                // Beat detected
                if (onBeat && typeof onBeat === 'function') {
                    onBeat(now, energy);
                }
                lastBeatTime = now;
            }
        }
        
        lastEnergy = energy;
    }
    
    checkBeat();
}

/**
 * Create a beat grid visualization based on BPM
 * @param {HTMLCanvasElement} canvas - Canvas element for visualization
 * @param {number} bpm - Beats per minute
 * @param {number} duration - Duration of audio in seconds
 */
function createBeatGrid(canvas, bpm, duration) {
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate beat interval in seconds
    const beatInterval = 60 / bpm;
    
    // Calculate number of beats in the duration
    const beatCount = Math.floor(duration / beatInterval);
    
    // Calculate pixel per second
    const pixelsPerSecond = canvas.width / duration;
    
    // Draw beat lines
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.5)';
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= beatCount; i++) {
        const beatTime = i * beatInterval;
        const x = beatTime * pixelsPerSecond;
        
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
        
        // Mark measures (assuming 4/4 time signature)
        if (i % 4 === 0) {
            ctx.lineWidth = 2;
            ctx.strokeStyle = 'rgba(99, 102, 241, 0.9)';
        } else {
            ctx.lineWidth = 1;
            ctx.strokeStyle = 'rgba(99, 102, 241, 0.5)';
        }
    }
}

/**
 * Extract speech segments from audio
 * This is a simplified client-side version of the server-side algorithm
 * @param {Float32Array} audioData - Audio data array
 * @param {number} sampleRate - Audio sample rate
 * @param {number} threshold - Energy threshold for speech detection
 * @returns {Array} Array of speech segment time ranges [start, end]
 */
function extractSpeechSegments(audioData, sampleRate, threshold = 0.05) {
    // Calculate RMS energy in chunks
    const chunkSize = Math.floor(0.025 * sampleRate); // 25ms chunks
    const chunkCount = Math.floor(audioData.length / chunkSize);
    const energyProfile = [];
    
    for (let i = 0; i < chunkCount; i++) {
        const chunk = audioData.slice(i * chunkSize, (i + 1) * chunkSize);
        let sum = 0;
        
        for (let j = 0; j < chunk.length; j++) {
            sum += chunk[j] * chunk[j];
        }
        
        const rms = Math.sqrt(sum / chunk.length);
        energyProfile.push(rms);
    }
    
    // Find maximum energy for normalization
    const maxEnergy = Math.max(...energyProfile);
    
    // Normalize and apply threshold
    const isSpeech = energyProfile.map(energy => energy > threshold * maxEnergy);
    
    // Find speech segments
    const segments = [];
    let inSpeech = false;
    let segmentStart = 0;
    
    for (let i = 0; i < isSpeech.length; i++) {
        if (isSpeech[i] && !inSpeech) {
            // Start of speech segment
            segmentStart = i * chunkSize / sampleRate;
            inSpeech = true;
        } else if (!isSpeech[i] && inSpeech) {
            // End of speech segment
            const segmentEnd = i * chunkSize / sampleRate;
            segments.push([segmentStart, segmentEnd]);
            inSpeech = false;
        }
    }
    
    // If we're still in a speech segment at the end
    if (inSpeech) {
        segments.push([segmentStart, audioData.length / sampleRate]);
    }
    
    return segments;
}

/**
 * Overlay beat markers on waveform visualization
 * @param {HTMLCanvasElement} canvas - Canvas element for visualization
 * @param {Array} beatPositions - Array of beat positions in seconds
 * @param {number} duration - Duration of audio in seconds
 */
function overlayBeatMarkers(canvas, beatPositions, duration) {
    const ctx = canvas.getContext('2d');
    
    // Calculate pixel per second
    const pixelsPerSecond = canvas.width / duration;
    
    // Draw beat markers
    ctx.fillStyle = 'rgba(16, 185, 129, 0.7)';
    
    beatPositions.forEach(beatTime => {
        const x = beatTime * pixelsPerSecond;
        
        // Draw marker
        ctx.beginPath();
        ctx.arc(x, canvas.height / 2, 4, 0, Math.PI * 2);
        ctx.fill();
    });
}

/**
 * Animate synchronization of vocals with beat
 * @param {HTMLCanvasElement} canvas - Canvas element for visualization
 * @param {Array} vocalSegments - Array of vocal segment time ranges [start, end]
 * @param {Array} beatPositions - Array of beat positions in seconds
 * @param {number} duration - Duration of audio in seconds
 */
function animateSynchronization(canvas, vocalSegments, beatPositions, duration) {
    const ctx = canvas.getContext('2d');
    
    // Cancel any ongoing animation
    if (visualizationFrameId) {
        cancelAnimationFrame(visualizationFrameId);
    }
    
    // Calculate pixel per second
    const pixelsPerSecond = canvas.width / duration;
    
    // Animation state
    let currentTime = 0;
    const animationDuration = duration * 1000; // Convert to ms
    const startTime = performance.now();
    
    function animate() {
        visualizationFrameId = requestAnimationFrame(animate);
        
        // Calculate current playback time
        const elapsed = performance.now() - startTime;
        const progress = Math.min(elapsed / animationDuration, 1);
        currentTime = progress * duration;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw beat grid
        beatPositions.forEach(beatTime => {
            const x = beatTime * pixelsPerSecond;
            
            ctx.strokeStyle = beatTime <= currentTime ? 'rgba(16, 185, 129, 0.7)' : 'rgba(99, 102, 241, 0.5)';
            ctx.lineWidth = 1;
            
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        });
        
        // Draw vocal segments
        vocalSegments.forEach(segment => {
            const startX = segment[0] * pixelsPerSecond;
            const endX = segment[1] * pixelsPerSecond;
            const width = endX - startX;
            
            // Different colors based on whether segment has been played
            if (segment[0] <= currentTime) {
                ctx.fillStyle = 'rgba(236, 72, 153, 0.5)';
            } else {
                ctx.fillStyle = 'rgba(236, 72, 153, 0.2)';
            }
            
            ctx.fillRect(startX, canvas.height / 4, width, canvas.height / 2);
        });
        
        // Draw playhead
        const playheadX = currentTime * pixelsPerSecond;
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(playheadX, 0);
        ctx.lineTo(playheadX, canvas.height);
        ctx.stroke();
        
        // Stop animation when complete
        if (progress >= 1) {
            cancelAnimationFrame(visualizationFrameId);
        }
    }
    
    animate();
}

// Export functions for use in main.js
window.audioProcessor = {
    createAudioContext,
    connectAnalyser,
    visualizeWaveform,
    visualizeSpectrum,
    detectBeats,
    createBeatGrid,
    extractSpeechSegments,
    overlayBeatMarkers,
    animateSynchronization
};
