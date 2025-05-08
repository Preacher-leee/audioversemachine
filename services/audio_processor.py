import os
import numpy as np
import librosa
import logging
import soundfile as sf
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

def analyze_audio(file_path):
    """
    Analyze audio file to extract BPM, beat positions, and other features.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        dict: Analysis results containing BPM, duration, beat positions, etc.
    """
    try:
        logger.info(f"Starting analysis of audio file: {file_path}")
        
        # Try loading with soundfile first (faster and more reliable)
        try:
            audio_data, sr = sf.read(file_path)
            # Convert to mono if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                y = np.mean(audio_data, axis=1)
            else:
                y = audio_data
        except Exception as sf_error:
            logger.warning(f"Failed to load with soundfile: {str(sf_error)}, trying librosa")
            # Fallback to librosa
            y, sr = librosa.load(file_path, sr=None)
        
        # Get file duration
        duration = len(y) / sr
        
        # Simplify beat detection to avoid Numba issues
        # Estimate BPM using autocorrelation - a simpler method
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
            beat_positions = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
        except Exception as beat_error:
            logger.warning(f"Beat detection failed: {str(beat_error)}, using fallback method")
            # Fallback to simple BPM calculation
            tempo = estimate_bpm_simple(y, sr)
            # Generate evenly spaced beats
            beat_count = int(duration * tempo / 60)
            beat_interval = 60 / tempo
            beat_positions = np.array([i * beat_interval for i in range(beat_count)])
        
        # Extract simplified features
        # Get amplitude envelope (use less computationally intensive method)
        hop_length = 512
        frame_length = 2048
        
        # Calculate amplitude envelope using simple max in frames
        amplitude_envelope = []
        for i in range(0, len(y), hop_length):
            if i + frame_length < len(y):
                amplitude_envelope.append(np.max(np.abs(y[i:i+frame_length])))
            else:
                amplitude_envelope.append(np.max(np.abs(y[i:])))
        
        amplitude_envelope = np.array(amplitude_envelope)
        amplitude_times = np.arange(len(amplitude_envelope)) * hop_length / sr
        
        # Compute onsets in a simpler way
        try:
            # Detect onsets using spectral flux
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        except Exception as onset_error:
            logger.warning(f"Onset detection failed: {str(onset_error)}, using amplitude peaks")
            # Fallback to peak picking on amplitude envelope
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(amplitude_envelope, height=0.5*np.max(amplitude_envelope), distance=sr//256)
            onset_times = amplitude_times[peaks]
        
        analysis_result = {
            'bpm': float(tempo),
            'duration': float(duration),
            'sample_rate': sr,
            'beat_positions': beat_positions.tolist(),
            'onset_times': onset_times.tolist(),
            'energy_profile': {
                'times': amplitude_times.tolist(),
                'values': amplitude_envelope.tolist()
            },
            'amplitude_envelope': amplitude_envelope.tolist()
        }
        
        logger.info(f"Audio analysis completed. BPM: {tempo}, Duration: {duration}s")
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error analyzing audio file: {str(e)}")
        raise Exception(f"Failed to analyze audio: {str(e)}")

def estimate_bpm_simple(y, sr):
    """
    Simple BPM estimation using autocorrelation.
    This is a fallback method that doesn't rely on librosa's beat tracker.
    """
    # Compute onset strength
    n_fft = 2048
    hop_length = 512
    
    # Compute spectral flux (difference between successive spectra)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    
    # Take the difference
    flux = np.zeros_like(S[:, :-1])
    flux = np.sum(np.maximum(0, S[:, 1:] - S[:, :-1]), axis=0)
    
    # Find the peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(flux, distance=sr/hop_length/2)  # At least 0.5 seconds apart
    
    if len(peaks) < 2:
        # Not enough peaks, return a default value
        return 120.0
    
    # Calculate time differences between peaks
    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    time_diffs = np.diff(peak_times)
    
    # Convert time differences to BPM
    bpms = 60 / time_diffs
    
    # Filter out extreme values
    reasonable_bpms = bpms[(bpms >= 60) & (bpms <= 200)]
    
    if len(reasonable_bpms) == 0:
        # No reasonable BPM found, return a default value
        return 120.0
    
    # Return the median BPM
    return np.median(reasonable_bpms)

def get_audio_waveform(file_path, samples=1000):
    """
    Extract waveform data for visualization.
    
    Args:
        file_path (str): Path to the audio file
        samples (int): Number of samples to return for visualization
        
    Returns:
        list: Waveform amplitude data
    """
    try:
        # Try loading with soundfile first
        try:
            audio_data, sr = sf.read(file_path)
            # Convert to mono if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                y = np.mean(audio_data, axis=1)
            else:
                y = audio_data
        except Exception:
            # Fallback to librosa
            y, sr = librosa.load(file_path, sr=None)
        
        # Simple downsampling approach (more efficient)
        if len(y) > samples:
            indices = np.linspace(0, len(y) - 1, samples, dtype=int)
            waveform = np.abs(y[indices])
        else:
            # If we have fewer points than requested, just use absolute values
            waveform = np.abs(y)
        
        # Normalize to 0-1 range
        if np.max(waveform) > 0:
            waveform = waveform / np.max(waveform)
        
        return waveform.tolist()
        
    except Exception as e:
        logger.error(f"Error extracting waveform: {str(e)}")
        raise Exception(f"Failed to extract waveform: {str(e)}")
