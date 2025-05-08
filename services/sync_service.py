import os
import numpy as np
import librosa
import soundfile as sf
import logging
import tempfile
from pydub import AudioSegment
from scipy import signal

logger = logging.getLogger(__name__)

def synchronize_audio(vocal_path, instrumental_path, instrumental_analysis, output_path, params=None):
    """
    Synchronize vocal audio with instrumental based on beat detection.
    
    Args:
        vocal_path (str): Path to the vocal audio file
        instrumental_path (str): Path to the instrumental audio file
        instrumental_analysis (dict): Pre-computed analysis of the instrumental
        output_path (str): Path to save the synchronized audio
        params (dict): Synchronization parameters
    
    Returns:
        dict: Synchronization result with metadata
    """
    try:
        logger.info(f"Starting audio synchronization")
        
        # Default parameters
        if params is None:
            params = {
                'alignment_mode': 'beat',
                'time_stretch': True,
                'timing_adjustments': []
            }
        
        # Load vocal audio
        y_vocal, sr_vocal = librosa.load(vocal_path, sr=None)
        
        # Load instrumental audio (for reference)
        y_inst, sr_inst = librosa.load(instrumental_path, sr=None)
        
        # Resample vocals to match instrumental sample rate if needed
        if sr_vocal != sr_inst:
            logger.info(f"Resampling vocals from {sr_vocal}Hz to {sr_inst}Hz")
            y_vocal = librosa.resample(y_vocal, orig_sr=sr_vocal, target_sr=sr_inst)
            sr_vocal = sr_inst
        
        # Get beat positions from instrumental analysis
        beat_positions = np.array(instrumental_analysis['beat_positions'])
        bpm = instrumental_analysis['bpm']
        
        # Analyze the vocals to get speech segments/onsets
        logger.info("Analyzing vocals for speech segments")
        vocal_segments = detect_speech_segments(y_vocal, sr_vocal)
        
        # Determine segment durations
        segment_durations = np.diff(np.append(vocal_segments, len(y_vocal)/sr_vocal))
        
        # Alignment strategy based on mode
        if params['alignment_mode'] == 'beat':
            # Align vocal segments to beats
            synced_vocals = align_to_beats(
                y_vocal, sr_vocal, vocal_segments, segment_durations, 
                beat_positions, params['time_stretch']
            )
        elif params['alignment_mode'] == 'phrase':
            # Align entire phrases to broader sections
            synced_vocals = align_to_phrases(
                y_vocal, sr_vocal, vocal_segments, segment_durations, 
                beat_positions, params['time_stretch']
            )
        else:
            # Default: Simple time-based alignment
            synced_vocals = y_vocal
        
        # Apply any manual timing adjustments
        if params['timing_adjustments']:
            synced_vocals = apply_timing_adjustments(
                synced_vocals, sr_vocal, params['timing_adjustments']
            )
        
        # Ensure the synchronized vocals don't exceed the instrumental length
        max_length = len(y_inst)
        if len(synced_vocals) > max_length:
            logger.info(f"Trimming synced vocals to match instrumental length")
            synced_vocals = synced_vocals[:max_length]
        
        # If the synchronized vocals are shorter, pad with silence
        elif len(synced_vocals) < max_length:
            logger.info(f"Padding synced vocals to match instrumental length")
            padding = np.zeros(max_length - len(synced_vocals))
            synced_vocals = np.concatenate([synced_vocals, padding])
        
        # Save the synchronized vocals
        sf.write(output_path, synced_vocals, sr_vocal)
        
        # Prepare result data
        duration = len(synced_vocals) / sr_vocal
        
        # Calculate alignment points for visualization
        # These are time points where vocal segments align with beats
        alignment_points = []
        for i, seg_start in enumerate(vocal_segments):
            if i < len(segment_durations):
                # Find the closest beat
                closest_beat_idx = np.argmin(np.abs(beat_positions - seg_start))
                alignment_points.append({
                    'vocalTime': float(seg_start),
                    'beatTime': float(beat_positions[closest_beat_idx]),
                    'duration': float(segment_durations[i])
                })
        
        return {
            'duration': duration,
            'sample_rate': sr_vocal,
            'file_size': os.path.getsize(output_path),
            'alignment_points': alignment_points
        }
        
    except Exception as e:
        logger.error(f"Error synchronizing audio: {str(e)}")
        raise Exception(f"Failed to synchronize audio: {str(e)}")

def detect_speech_segments(y, sr, threshold=0.05, min_silence_duration=0.3):
    """
    Detect speech segments in the vocal audio.
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sample rate
        threshold (float): Energy threshold for speech detection
        min_silence_duration (float): Minimum silence duration in seconds
        
    Returns:
        np.ndarray: Array of segment start times in seconds
    """
    # Compute RMS energy
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    
    # Find segments above threshold
    is_speech = rms > threshold * np.max(rms)
    
    # Convert to sample indices
    speech_samples = librosa.frames_to_samples(np.where(is_speech)[0], hop_length=hop_length)
    
    # Find segment boundaries
    boundaries = np.concatenate([[0], np.where(np.diff(speech_samples) > min_silence_duration * sr)[0] + 1, [len(speech_samples)]])
    
    # Extract segment start times
    segment_starts = []
    for i in range(len(boundaries) - 1):
        if boundaries[i+1] - boundaries[i] > 0:  # Skip empty segments
            start_idx = speech_samples[boundaries[i]]
            segment_starts.append(start_idx / sr)
    
    return np.array(segment_starts)

def align_to_beats(y_vocal, sr, vocal_segments, segment_durations, beat_positions, allow_time_stretch=True):
    """
    Align vocal segments to instrumental beats.
    
    Args:
        y_vocal (np.ndarray): Vocal audio signal
        sr (int): Sample rate
        vocal_segments (np.ndarray): Start times of vocal segments
        segment_durations (np.ndarray): Durations of vocal segments
        beat_positions (np.ndarray): Beat positions in seconds
        allow_time_stretch (bool): Whether to allow time-stretching
        
    Returns:
        np.ndarray: Aligned vocal audio
    """
    logger.info(f"Aligning {len(vocal_segments)} vocal segments to beats")
    
    # Initialize output array
    output_duration = beat_positions[-1] + 2.0  # Add some buffer at the end
    output = np.zeros(int(output_duration * sr))
    
    # Process each vocal segment
    for i, seg_start in enumerate(vocal_segments):
        if i >= len(segment_durations):
            break
            
        # Get segment duration
        seg_duration = segment_durations[i]
        
        # Extract the segment
        start_sample = int(seg_start * sr)
        end_sample = int((seg_start + seg_duration) * sr)
        if end_sample > len(y_vocal):
            end_sample = len(y_vocal)
        
        segment = y_vocal[start_sample:end_sample]
        
        # Find closest beat
        closest_beat_idx = np.argmin(np.abs(beat_positions - seg_start))
        target_start = beat_positions[closest_beat_idx]
        
        # Find next beat for segment end
        beat_spacing = np.median(np.diff(beat_positions))
        beats_needed = max(1, int(np.ceil(seg_duration / beat_spacing)))
        
        if closest_beat_idx + beats_needed < len(beat_positions):
            target_end = beat_positions[closest_beat_idx + beats_needed]
        else:
            # If we run out of beats, estimate based on BPM
            target_end = target_start + beats_needed * beat_spacing
        
        target_duration = target_end - target_start
        
        # Apply time stretching if needed and allowed
        if allow_time_stretch and abs(target_duration - seg_duration) > 0.05:
            # Time stretch factor (ratio of target to original duration)
            stretch_factor = target_duration / seg_duration
            
            # Avoid extreme stretching
            if 0.5 <= stretch_factor <= 2.0:
                logger.info(f"Time-stretching segment {i} by factor {stretch_factor:.2f}")
                segment = librosa.effects.time_stretch(segment, rate=stretch_factor)
        
        # Place in output
        target_start_sample = int(target_start * sr)
        target_end_sample = target_start_sample + len(segment)
        
        # Ensure we don't exceed the output array
        if target_end_sample > len(output):
            segment = segment[:len(output) - target_start_sample]
            target_end_sample = len(output)
        
        # Apply crossfade if overlapping with existing content
        if np.any(output[target_start_sample:target_end_sample] != 0):
            # Apply simple linear crossfade
            fade_length = min(1000, len(segment) // 4)  # 1000 samples or 1/4 of segment
            fade_in = np.linspace(0, 1, fade_length)
            fade_out = np.linspace(1, 0, fade_length)
            
            # Apply fade-in to new segment
            segment[:fade_length] *= fade_in
            
            # Apply fade-out to existing content
            output[target_start_sample:target_start_sample+fade_length] *= fade_out
        
        # Add segment to output
        output[target_start_sample:target_end_sample] += segment
    
    # Normalize output
    if np.max(np.abs(output)) > 0:
        output = output / np.max(np.abs(output))
    
    return output

def align_to_phrases(y_vocal, sr, vocal_segments, segment_durations, beat_positions, allow_time_stretch=True):
    """
    Align vocal phrases to broader sections of the instrumental.
    
    Args:
        y_vocal (np.ndarray): Vocal audio signal
        sr (int): Sample rate
        vocal_segments (np.ndarray): Start times of vocal segments
        segment_durations (np.ndarray): Durations of vocal segments
        beat_positions (np.ndarray): Beat positions in seconds
        allow_time_stretch (bool): Whether to allow time-stretching
        
    Returns:
        np.ndarray: Aligned vocal audio
    """
    logger.info(f"Aligning vocal phrases to instrumental sections")
    
    # Group vocal segments into phrases (segments separated by short silences)
    phrase_boundaries = [0]
    for i in range(1, len(vocal_segments)):
        gap = vocal_segments[i] - (vocal_segments[i-1] + segment_durations[i-1])
        if gap > 0.5:  # New phrase if gap > 0.5 seconds
            phrase_boundaries.append(i)
    phrase_boundaries.append(len(vocal_segments))
    
    # Group beats into phrases (e.g., 4 or 8 beats per phrase)
    beats_per_phrase = 8  # Typical for music
    beat_phrases = [beat_positions[i:i+beats_per_phrase] 
                   for i in range(0, len(beat_positions), beats_per_phrase)]
    
    # Initialize output array
    output_duration = beat_positions[-1] + 2.0
    output = np.zeros(int(output_duration * sr))
    
    # Process each vocal phrase
    current_beat_phrase = 0
    for i in range(len(phrase_boundaries) - 1):
        start_idx = phrase_boundaries[i]
        end_idx = phrase_boundaries[i+1]
        
        # Calculate phrase duration
        phrase_start = vocal_segments[start_idx]
        phrase_end = vocal_segments[end_idx-1] + segment_durations[end_idx-1] if end_idx > 0 else phrase_start
        phrase_duration = phrase_end - phrase_start
        
        # Extract the phrase
        start_sample = int(phrase_start * sr)
        end_sample = int(phrase_end * sr)
        if end_sample > len(y_vocal):
            end_sample = len(y_vocal)
        
        phrase = y_vocal[start_sample:end_sample]
        
        # Select target beat phrase
        if current_beat_phrase < len(beat_phrases):
            target_beats = beat_phrases[current_beat_phrase]
            target_start = target_beats[0]
            target_end = target_beats[-1] if len(target_beats) > 1 else target_start + phrase_duration
            target_duration = target_end - target_start
            
            # Apply time stretching if needed
            if allow_time_stretch and abs(target_duration - phrase_duration) > 0.1:
                stretch_factor = target_duration / phrase_duration
                
                # Avoid extreme stretching
                if 0.5 <= stretch_factor <= 2.0:
                    logger.info(f"Time-stretching phrase {i} by factor {stretch_factor:.2f}")
                    phrase = librosa.effects.time_stretch(phrase, rate=stretch_factor)
            
            # Place in output
            target_start_sample = int(target_start * sr)
            target_end_sample = target_start_sample + len(phrase)
            
            # Ensure we don't exceed the output array
            if target_end_sample > len(output):
                phrase = phrase[:len(output) - target_start_sample]
                target_end_sample = len(output)
            
            # Apply fade if needed
            if np.any(output[target_start_sample:target_end_sample] != 0):
                fade_length = min(1000, len(phrase) // 4)
                fade_in = np.linspace(0, 1, fade_length)
                fade_out = np.linspace(1, 0, fade_length)
                
                phrase[:fade_length] *= fade_in
                output[target_start_sample:target_start_sample+fade_length] *= fade_out
            
            # Add phrase to output
            output[target_start_sample:target_end_sample] += phrase
            
            # Move to next beat phrase
            current_beat_phrase += 1
        else:
            # If we run out of beat phrases, just append at the end
            logger.warning(f"Ran out of beat phrases, appending phrase {i} at the end")
            target_start = output_duration - phrase_duration
            target_start_sample = int(target_start * sr)
            target_end_sample = target_start_sample + len(phrase)
            
            if target_end_sample > len(output):
                phrase = phrase[:len(output) - target_start_sample]
                target_end_sample = len(output)
            
            output[target_start_sample:target_end_sample] += phrase
    
    # Normalize output
    if np.max(np.abs(output)) > 0:
        output = output / np.max(np.abs(output))
    
    return output

def apply_timing_adjustments(y, sr, adjustments):
    """
    Apply manual timing adjustments to the audio.
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sample rate
        adjustments (list): List of timing adjustments
    
    Returns:
        np.ndarray: Adjusted audio
    """
    logger.info(f"Applying {len(adjustments)} manual timing adjustments")
    
    # Convert to AudioSegment for easier manipulation
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
        sf.write(temp_file.name, y, sr)
        audio = AudioSegment.from_wav(temp_file.name)
    
    # Sort adjustments by start time
    adjustments.sort(key=lambda x: x.get('startTime', 0))
    
    # Initialize output
    output = AudioSegment.empty()
    last_end = 0
    
    for adj in adjustments:
        start_time = adj.get('startTime', 0) * 1000  # Convert to ms
        end_time = adj.get('endTime', 0) * 1000
        new_start = adj.get('newStartTime', 0) * 1000
        new_end = adj.get('newEndTime', 0) * 1000
        
        # Add audio before this segment
        if start_time > last_end:
            output += audio[last_end:start_time]
        
        # Extract the segment to adjust
        segment = audio[start_time:end_time]
        
        # Apply time stretching if needed
        orig_duration = end_time - start_time
        new_duration = new_end - new_start
        
        if abs(new_duration - orig_duration) > 10:  # More than 10ms difference
            # Calculate stretch factor
            stretch_factor = new_duration / orig_duration if orig_duration > 0 else 1.0
            
            # Limit extreme stretching
            stretch_factor = max(0.5, min(2.0, stretch_factor))
            
            # Time stretch using librosa and convert back to AudioSegment
            with tempfile.NamedTemporaryFile(suffix='.wav') as seg_file:
                segment.export(seg_file.name, format='wav')
                y_seg, sr_seg = librosa.load(seg_file.name, sr=None)
                y_stretched = librosa.effects.time_stretch(y_seg, rate=stretch_factor)
                
                with tempfile.NamedTemporaryFile(suffix='.wav') as out_file:
                    sf.write(out_file.name, y_stretched, sr_seg)
                    segment = AudioSegment.from_wav(out_file.name)
        
        # Add silence at the beginning if needed
        if new_start > output.duration_seconds * 1000:
            silence_duration = new_start - output.duration_seconds * 1000
            output += AudioSegment.silent(duration=silence_duration)
        
        # Add the adjusted segment
        output += segment
        
        # Update last_end
        last_end = end_time
    
    # Add any remaining audio
    if last_end < len(audio):
        output += audio[last_end:]
    
    # Convert back to numpy array
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
        output.export(temp_file.name, format='wav')
        y_out, sr_out = librosa.load(temp_file.name, sr=sr)
    
    return y_out
