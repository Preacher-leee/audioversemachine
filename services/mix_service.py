import os
import numpy as np
import librosa
import soundfile as sf
import logging
from scipy import signal
import tempfile
from pydub import AudioSegment

logger = logging.getLogger(__name__)

def mix_audio_tracks(vocal_path, instrumental_path, output_path, params=None):
    """
    Mix vocal and instrumental tracks with optional audio effects.
    
    Args:
        vocal_path (str): Path to the vocal audio file
        instrumental_path (str): Path to the instrumental audio file
        output_path (str): Path to save the mixed audio
        params (dict): Mixing parameters
    
    Returns:
        dict: Mixing result with metadata
    """
    try:
        logger.info(f"Starting audio mixing")
        
        # Default parameters
        if params is None:
            params = {
                'vocal_level': 0.8,
                'eq_preset': 'balanced',
                'compression': 'medium',
                'stereo_width': 1.0,
                'auto_ducking': True
            }
        
        # Load audio files using pydub for more precise manipulation
        vocals = AudioSegment.from_file(vocal_path)
        instrumental = AudioSegment.from_file(instrumental_path)
        
        # Ensure both tracks have the same number of channels
        if vocals.channels != instrumental.channels:
            if vocals.channels == 1 and instrumental.channels == 2:
                logger.info("Converting vocals to stereo")
                vocals = vocals.set_channels(2)
            elif vocals.channels == 2 and instrumental.channels == 1:
                logger.info("Converting instrumental to stereo")
                instrumental = instrumental.set_channels(2)
        
        # Match sample rates
        if vocals.frame_rate != instrumental.frame_rate:
            logger.info(f"Converting vocals sample rate from {vocals.frame_rate}Hz to {instrumental.frame_rate}Hz")
            vocals = vocals.set_frame_rate(instrumental.frame_rate)
        
        # Process vocals with EQ and compression
        vocals = apply_vocal_processing(vocals, params['eq_preset'], params['compression'])
        
        # Apply stereo widening if needed
        if params['stereo_width'] != 1.0 and vocals.channels == 2:
            vocals = apply_stereo_widening(vocals, params['stereo_width'])
        
        # Set vocal level
        vocals = vocals - (6 - 6 * params['vocal_level'])  # Adjust level based on parameter
        
        # Apply auto-ducking if enabled
        if params['auto_ducking']:
            instrumental = apply_auto_ducking(instrumental, vocals)
        
        # Mix vocals and instrumental
        mixed = mix_audio_segments(vocals, instrumental)
        
        # Apply final limiter to prevent clipping
        mixed = apply_limiter(mixed)
        
        # Save the mixed audio
        mixed.export(output_path, format='wav')
        
        # Get file metadata
        file_size = os.path.getsize(output_path)
        
        # Load the file with librosa to get peak level
        y, sr = librosa.load(output_path, sr=None)
        peak_level = float(np.max(np.abs(y)))
        duration = float(len(y) / sr)
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'file_size': file_size,
            'peak_level': peak_level
        }
        
    except Exception as e:
        logger.error(f"Error mixing audio: {str(e)}")
        raise Exception(f"Failed to mix audio: {str(e)}")

def apply_vocal_processing(vocals, eq_preset, compression_level):
    """
    Apply EQ and compression to vocals.
    
    Args:
        vocals (AudioSegment): Vocal audio
        eq_preset (str): EQ preset name
        compression_level (str): Compression intensity
        
    Returns:
        AudioSegment: Processed vocals
    """
    logger.info(f"Applying vocal processing with EQ preset: {eq_preset}, compression: {compression_level}")
    
    # Convert to numpy array for processing
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
        vocals.export(temp_file.name, format='wav')
        y, sr = librosa.load(temp_file.name, sr=None, mono=False)
    
    # Apply EQ based on preset
    if eq_preset == 'radio':
        # Radio preset: Cut lows, boost mids, limit highs
        y = apply_eq_bands(y, sr, [
            {'freq': 80, 'gain': -6, 'q': 0.7},    # Cut below 80Hz
            {'freq': 300, 'gain': 2, 'q': 1.0},    # Light boost at 300Hz
            {'freq': 2500, 'gain': 4, 'q': 0.8},   # Presence boost at 2.5kHz
            {'freq': 5000, 'gain': 3, 'q': 0.7},   # Light high boost
            {'freq': 10000, 'gain': -2, 'q': 0.7}  # Roll off extreme highs
        ])
    elif eq_preset == 'warm':
        # Warm preset: Keep lows, boost low-mids, gentle highs
        y = apply_eq_bands(y, sr, [
            {'freq': 100, 'gain': 2, 'q': 0.8},    # Light boost in low end
            {'freq': 300, 'gain': 3, 'q': 0.7},    # Boost at 300Hz for warmth
            {'freq': 1000, 'gain': -1, 'q': 1.2},  # Slight dip in mids
            {'freq': 3000, 'gain': 2, 'q': 0.8},   # Light presence
            {'freq': 8000, 'gain': 1, 'q': 0.7}    # Gentle high end
        ])
    elif eq_preset == 'bright':
        # Bright preset: Cut lows, restrained mids, boost highs
        y = apply_eq_bands(y, sr, [
            {'freq': 80, 'gain': -4, 'q': 0.7},    # Cut below 80Hz
            {'freq': 500, 'gain': -2, 'q': 1.0},   # Reduce low-mids
            {'freq': 2000, 'gain': 2, 'q': 0.8},   # Light boost at 2kHz
            {'freq': 5000, 'gain': 4, 'q': 0.7},   # Boost at 5kHz for clarity
            {'freq': 10000, 'gain': 3, 'q': 0.7}   # Extended highs
        ])
    else:  # balanced
        # Balanced preset: Moderate adjustments
        y = apply_eq_bands(y, sr, [
            {'freq': 100, 'gain': -2, 'q': 0.7},   # Reduce rumble
            {'freq': 300, 'gain': 1, 'q': 1.0},    # Light warmth
            {'freq': 1000, 'gain': 0, 'q': 1.2},   # Neutral mids
            {'freq': 3000, 'gain': 2, 'q': 0.8},   # Light presence
            {'freq': 8000, 'gain': 1, 'q': 0.7}    # Light air
        ])
    
    # Apply compression based on level
    if compression_level == 'light':
        y = apply_compression(y, threshold=-20, ratio=2, attack=20, release=100)
    elif compression_level == 'medium':
        y = apply_compression(y, threshold=-18, ratio=3, attack=10, release=80)
    elif compression_level == 'heavy':
        y = apply_compression(y, threshold=-16, ratio=4, attack=5, release=50)
    
    # Convert back to AudioSegment
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
        sf.write(temp_file.name, y.T if y.ndim > 1 else y, sr)
        processed_vocals = AudioSegment.from_wav(temp_file.name)
    
    return processed_vocals

def apply_eq_bands(y, sr, bands):
    """
    Apply parametric EQ with multiple bands.
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sample rate
        bands (list): List of EQ band parameters (freq, gain, q)
        
    Returns:
        np.ndarray: EQ'd audio
    """
    # Process each channel if stereo
    if y.ndim > 1:
        return np.array([apply_eq_bands(channel, sr, bands) for channel in y])
    
    # Apply each EQ band
    y_eq = y.copy()
    for band in bands:
        freq = band['freq']
        gain = band['gain']
        q = band['q']
        
        # Skip if no gain
        if abs(gain) < 0.1:
            continue
        
        # Convert gain from dB to linear
        gain_linear = 10 ** (gain / 20.0)
        
        # Design a peaking EQ filter
        b, a = signal.iirpeak(freq, q, sr, gain_linear)
        
        # Apply the filter
        y_eq = signal.lfilter(b, a, y_eq)
    
    return y_eq

def apply_compression(y, threshold=-20, ratio=3, attack=10, release=100):
    """
    Apply dynamic range compression.
    
    Args:
        y (np.ndarray): Audio signal
        threshold (float): Threshold in dB
        ratio (float): Compression ratio
        attack (float): Attack time in ms
        release (float): Release time in ms
        
    Returns:
        np.ndarray: Compressed audio
    """
    # Process each channel if stereo
    if y.ndim > 1:
        return np.array([apply_compression(channel, threshold, ratio, attack, release) for channel in y])
    
    # Convert threshold from dB to linear
    threshold_linear = 10 ** (threshold / 20.0)
    
    # Calculate envelope (simple approach)
    # In a real implementation, you would use a proper envelope follower
    abs_y = np.abs(y)
    env = np.zeros_like(y)
    
    # Convert attack/release from ms to samples
    a = np.exp(-1/(attack * 0.001))
    r = np.exp(-1/(release * 0.001))
    
    # Simple envelope follower
    for i in range(1, len(y)):
        if abs_y[i] > env[i-1]:
            # Attack phase
            env[i] = a * env[i-1] + (1-a) * abs_y[i]
        else:
            # Release phase
            env[i] = r * env[i-1] + (1-r) * abs_y[i]
    
    # Apply compression
    gain = np.ones_like(y)
    mask = env > threshold_linear
    gain[mask] = (env[mask] / threshold_linear) ** (1/ratio - 1)
    
    # Apply gain
    y_compressed = y * gain
    
    # Apply makeup gain
    max_gain_reduction = np.min(gain[mask]) if np.any(mask) else 1.0
    makeup_gain = 1.0 / max_gain_reduction if max_gain_reduction > 0 else 1.0
    
    # Ensure makeup gain is reasonable
    makeup_gain = min(2.0, makeup_gain)
    
    y_compressed = y_compressed * makeup_gain
    
    return y_compressed

def apply_stereo_widening(vocals, width_factor):
    """
    Apply stereo widening effect.
    
    Args:
        vocals (AudioSegment): Vocal audio
        width_factor (float): Stereo width factor (1.0 = normal, >1.0 = wider)
        
    Returns:
        AudioSegment: Processed vocals with enhanced stereo width
    """
    logger.info(f"Applying stereo widening with factor: {width_factor}")
    
    if vocals.channels != 2:
        logger.warning("Can't apply stereo widening to mono audio")
        return vocals
    
    # Convert to numpy array for processing
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
        vocals.export(temp_file.name, format='wav')
        y, sr = librosa.load(temp_file.name, sr=None, mono=False)
    
    # Ensure we have stereo
    if y.ndim < 2 or y.shape[0] < 2:
        logger.warning("Audio doesn't appear to be stereo")
        return vocals
    
    # Create mid and side channels
    mid = (y[0] + y[1]) / 2
    side = (y[0] - y[1]) / 2
    
    # Apply width factor to side channel
    side = side * width_factor
    
    # Recombine to stereo
    y_wide = np.array([mid + side, mid - side])
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(y_wide))
    if max_val > 1.0:
        y_wide = y_wide / max_val
    
    # Convert back to AudioSegment
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
        sf.write(temp_file.name, y_wide.T, sr)
        widened_vocals = AudioSegment.from_wav(temp_file.name)
    
    return widened_vocals

def apply_auto_ducking(instrumental, vocals, threshold=-24, reduction=0.7, attack=10, release=100):
    """
    Apply auto-ducking to reduce instrumental volume when vocals are present.
    
    Args:
        instrumental (AudioSegment): Instrumental audio
        vocals (AudioSegment): Vocal audio
        threshold (float): Threshold in dB
        reduction (float): Volume reduction amount (0-1)
        attack (float): Attack time in ms
        release (float): Release time in ms
        
    Returns:
        AudioSegment: Ducked instrumental
    """
    logger.info(f"Applying auto-ducking with threshold: {threshold}dB, reduction: {reduction}")
    
    # Convert to numpy arrays
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
        vocals.export(temp_file.name, format='wav')
        y_vocals, sr_vocals = librosa.load(temp_file.name, sr=None, mono=True)
    
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
        instrumental.export(temp_file.name, format='wav')
        y_inst, sr_inst = librosa.load(temp_file.name, sr=None, mono=False)
        inst_channels = 1 if y_inst.ndim == 1 else y_inst.shape[0]
    
    # Convert threshold from dB to linear
    threshold_linear = 10 ** (threshold / 20.0)
    
    # Calculate envelope of vocals
    abs_vocals = np.abs(y_vocals)
    vocal_env = np.zeros_like(y_vocals)
    
    # Convert attack/release from ms to samples
    a = np.exp(-1/(attack * 0.001))
    r = np.exp(-1/(release * 0.001))
    
    # Simple envelope follower
    for i in range(1, len(y_vocals)):
        if abs_vocals[i] > vocal_env[i-1]:
            # Attack phase
            vocal_env[i] = a * vocal_env[i-1] + (1-a) * abs_vocals[i]
        else:
            # Release phase
            vocal_env[i] = r * vocal_env[i-1] + (1-r) * abs_vocals[i]
    
    # Calculate ducking gain
    ducking_gain = np.ones_like(vocal_env)
    mask = vocal_env > threshold_linear
    
    # Apply reduction
    ducking_gain[mask] = 1.0 - (reduction * (vocal_env[mask] - threshold_linear) / (1.0 - threshold_linear))
    
    # Make sure gain doesn't go negative
    ducking_gain = np.clip(ducking_gain, 1.0 - reduction, 1.0)
    
    # Apply smoothing to gain changes
    window_size = int(0.05 * sr_vocals)  # 50ms smoothing window
    if window_size > 1:
        ducking_gain = np.convolve(ducking_gain, np.ones(window_size)/window_size, mode='same')
    
    # Resample gain curve if sample rates differ
    if sr_vocals != sr_inst:
        # Create time points for original gain curve
        t_original = np.linspace(0, len(ducking_gain) / sr_vocals, len(ducking_gain), endpoint=False)
        
        # Create time points for resampled gain curve
        t_resampled = np.linspace(0, len(ducking_gain) / sr_vocals, int(len(ducking_gain) * sr_inst / sr_vocals), endpoint=False)
        
        # Resample the gain curve
        ducking_gain = np.interp(t_resampled, t_original, ducking_gain)
    
    # Ensure gain curve matches instrumental length
    if len(ducking_gain) > len(y_inst[0]) if inst_channels > 1 else len(y_inst):
        ducking_gain = ducking_gain[:len(y_inst[0]) if inst_channels > 1 else len(y_inst)]
    elif len(ducking_gain) < len(y_inst[0]) if inst_channels > 1 else len(y_inst):
        padding = np.ones(len(y_inst[0]) if inst_channels > 1 else len(y_inst) - len(ducking_gain))
        ducking_gain = np.concatenate([ducking_gain, padding])
    
    # Apply gain to instrumental
    if inst_channels > 1:
        y_ducked = np.array([channel * ducking_gain for channel in y_inst])
    else:
        y_ducked = y_inst * ducking_gain
    
    # Convert back to AudioSegment
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
        sf.write(temp_file.name, y_ducked.T if inst_channels > 1 else y_ducked, sr_inst)
        ducked_instrumental = AudioSegment.from_wav(temp_file.name)
    
    return ducked_instrumental

def mix_audio_segments(a, b):
    """
    Mix two AudioSegments with proper gain compensation.
    
    Args:
        a (AudioSegment): First audio segment
        b (AudioSegment): Second audio segment
        
    Returns:
        AudioSegment: Mixed audio
    """
    # Ensure same length
    max_length = max(len(a), len(b))
    if len(a) < max_length:
        a = a + AudioSegment.silent(duration=max_length - len(a))
    if len(b) < max_length:
        b = b + AudioSegment.silent(duration=max_length - len(b))
    
    # Mix with 3dB headroom
    return (a.overlay(b))

def apply_limiter(audio, threshold=-0.5, release=50):
    """
    Apply a limiter to prevent clipping.
    
    Args:
        audio (AudioSegment): Audio to process
        threshold (float): Threshold in dB (negative)
        release (float): Release time in ms
        
    Returns:
        AudioSegment: Limited audio
    """
    # Convert to numpy array
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
        audio.export(temp_file.name, format='wav')
        y, sr = librosa.load(temp_file.name, sr=None, mono=False)
    
    # Process channels
    channels = 1 if y.ndim == 1 else y.shape[0]
    
    # Convert threshold from dB to linear
    threshold_linear = 10 ** (threshold / 20.0)
    
    # Apply limiting to each channel
    if channels > 1:
        y_limited = np.array([apply_limiter_to_channel(channel, threshold_linear, release, sr) for channel in y])
    else:
        y_limited = apply_limiter_to_channel(y, threshold_linear, release, sr)
    
    # Convert back to AudioSegment
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
        sf.write(temp_file.name, y_limited.T if channels > 1 else y_limited, sr)
        limited_audio = AudioSegment.from_wav(temp_file.name)
    
    return limited_audio

def apply_limiter_to_channel(y, threshold, release, sr):
    """
    Apply limiting to a single channel.
    
    Args:
        y (np.ndarray): Audio channel
        threshold (float): Threshold (linear)
        release (float): Release time in ms
        sr (int): Sample rate
        
    Returns:
        np.ndarray: Limited audio channel
    """
    # Find peaks above threshold
    peaks = np.where(np.abs(y) > threshold)[0]
    
    if len(peaks) == 0:
        return y  # No limiting needed
    
    # Create gain reduction array
    gain_reduction = np.ones_like(y)
    
    # Release coefficient
    r = np.exp(-1/(release * 0.001))
    
    # Calculate gain reduction for each peak
    for peak_idx in peaks:
        gain_reduction[peak_idx] = threshold / np.abs(y[peak_idx])
    
    # Apply release envelope
    for i in range(1, len(y)):
        if gain_reduction[i] < gain_reduction[i-1]:
            # New reduction point
            pass
        else:
            # Release phase
            gain_reduction[i] = gain_reduction[i-1] + (1.0 - gain_reduction[i-1]) * (1.0 - r)
            if gain_reduction[i] > 0.999:
                gain_reduction[i] = 1.0
    
    # Apply gain reduction
    return y * gain_reduction
