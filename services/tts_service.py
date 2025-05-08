import os
import logging
import requests
import time
import numpy as np
import soundfile as sf
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

# Mock API call for development if API key is not available
# In real implementation, this would connect to actual TTS API like ElevenLabs
def generate_vocals(text, voice_id="default", style="Spoken Word", output_path=None):
    """
    Generate vocals from text using TTS API.
    
    Args:
        text (str): The text to convert to speech
        voice_id (str): ID or name of the voice to use
        style (str): Style of delivery (Spoken Word, Rap, ASMR, Movie Trailer, Motivational)
        output_path (str): Path to save the generated audio
        
    Returns:
        dict: Generation result with metadata
    """
    try:
        logger.info(f"Generating vocals with voice: {voice_id}, style: {style}")
        
        # Get API key from environment
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        
        if api_key:
            # This would be a real API call in production
            logger.info("Using ElevenLabs API for TTS generation")
            result = call_elevenlabs_api(text, voice_id, style, api_key, output_path)
        else:
            # For development without API key, create a synthetic response
            logger.warning("No API key found, using development mode with synthetic TTS")
            result = generate_synthetic_tts(text, style, output_path)
            
        return result
    
    except Exception as e:
        logger.error(f"Error generating vocals: {str(e)}")
        raise Exception(f"Failed to generate vocals: {str(e)}")

def call_elevenlabs_api(text, voice_id, style, api_key, output_path):
    """
    Call the ElevenLabs API to generate TTS audio.
    
    Args:
        text (str): The text to convert to speech
        voice_id (str): ID of the voice to use
        style (str): Style of delivery
        api_key (str): ElevenLabs API key
        output_path (str): Path to save the audio
        
    Returns:
        dict: Generation result with metadata
    """
    BASE_URL = "https://api.elevenlabs.io/v1/"
    
    # Voice mapping based on style
    voice_mapping = {
        "default": "21m00Tcm4TlvDq8ikWAM",  # Default voice (Rachel)
        "Spoken Word": "21m00Tcm4TlvDq8ikWAM",  # Rachel
        "Rap": "AZnzlk1XvdvUeBnXmlld",  # Domi
        "ASMR": "EXAVITQu4vr4xnSDxMaL",  # Bella
        "Movie Trailer": "VR6AewLTigWG4xSOukaG",  # Adam
        "Motivational": "pNInz6obpgDQGcFmaJgB"  # Sam
    }
    
    # Get the voice ID based on style if a specific ID isn't provided
    if voice_id == "default":
        voice_id = voice_mapping.get(style, voice_mapping["default"])
    
    # Prepare SSML with style-specific prosody
    ssml = prepare_ssml_for_style(text, style)
    
    # Configure voice settings based on style
    voice_settings = get_voice_settings_for_style(style)
    
    # Prepare API endpoint
    endpoint = urljoin(BASE_URL, f"text-to-speech/{voice_id}")
    
    # Prepare headers
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Prepare data
    data = {
        "text": ssml,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": voice_settings
    }
    
    # Make the API call
    response = requests.post(endpoint, headers=headers, json=data)
    
    if response.status_code != 200:
        raise Exception(f"API call failed with status {response.status_code}: {response.text}")
    
    # Save the audio to the specified path
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    # Get audio metadata (would normally parse from API response)
    # Here we'll read the file to get some basic info
    y, sr = sf.read(output_path)
    duration = len(y) / sr
    
    return {
        "duration": duration,
        "sample_rate": sr,
        "file_size": os.path.getsize(output_path),
        "voice_id": voice_id,
        "style": style
    }

def prepare_ssml_for_style(text, style):
    """
    Prepare SSML markup based on delivery style.
    
    Args:
        text (str): Original text
        style (str): Delivery style
        
    Returns:
        str: SSML-formatted text
    """
    # Split text into sentences or phrases
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
    
    ssml_parts = []
    
    if style == "Spoken Word":
        for sentence in sentences:
            ssml_parts.append(f'<s>{sentence}.</s>')
        ssml = f'<speak><prosody rate="medium" pitch="medium">{" ".join(ssml_parts)}</prosody></speak>'
    
    elif style == "Rap":
        for sentence in sentences:
            ssml_parts.append(f'<s><prosody rate="fast" pitch="medium">{sentence}.</prosody></s>')
        ssml = f'<speak>{" ".join(ssml_parts)}</speak>'
    
    elif style == "ASMR":
        for sentence in sentences:
            ssml_parts.append(f'<s><prosody rate="slow" volume="soft" pitch="low">{sentence}.</prosody></s>')
        ssml = f'<speak>{" ".join(ssml_parts)}</speak>'
    
    elif style == "Movie Trailer":
        for sentence in sentences:
            ssml_parts.append(f'<s><prosody rate="medium" volume="loud" pitch="low">{sentence}.</prosody></s>')
        ssml = f'<speak>{" ".join(ssml_parts)}</speak>'
    
    elif style == "Motivational":
        for sentence in sentences:
            ssml_parts.append(f'<s><prosody rate="medium" volume="loud" pitch="high">{sentence}.</prosody></s>')
        ssml = f'<speak>{" ".join(ssml_parts)}</speak>'
    
    else:  # Default
        ssml = f'<speak>{text}</speak>'
    
    return ssml

def get_voice_settings_for_style(style):
    """
    Get voice settings based on delivery style.
    
    Args:
        style (str): Delivery style
        
    Returns:
        dict: Voice settings
    """
    settings = {
        "Spoken Word": {
            "stability": 0.5,
            "similarity_boost": 0.75
        },
        "Rap": {
            "stability": 0.3,
            "similarity_boost": 0.5
        },
        "ASMR": {
            "stability": 0.8,
            "similarity_boost": 0.5
        },
        "Movie Trailer": {
            "stability": 0.4,
            "similarity_boost": 0.75
        },
        "Motivational": {
            "stability": 0.5,
            "similarity_boost": 0.8
        }
    }
    
    return settings.get(style, {"stability": 0.5, "similarity_boost": 0.75})

def generate_synthetic_tts(text, style, output_path):
    """
    Generate synthetic TTS for development/testing purposes.
    
    Args:
        text (str): The text to convert to speech
        style (str): Style of delivery
        output_path (str): Path to save the audio
        
    Returns:
        dict: Generation result with metadata
    """
    # Create a 3-second sine wave as a placeholder for each word
    # In a real implementation, this would be replaced with actual TTS
    word_count = len(text.split())
    duration = max(3, word_count * 0.3)  # At least 3 seconds, or 0.3s per word
    sample_rate = 22050
    
    # Create different waveforms based on style
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    
    if style == "Rap":
        # Faster rhythm for rap
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) * np.sin(2 * np.pi * 4 * t)
    elif style == "ASMR":
        # Softer sound for ASMR
        audio = 0.2 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    elif style == "Movie Trailer":
        # Dramatic sound for movie trailer
        audio = 0.5 * np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 300 * t)
    elif style == "Motivational":
        # Energetic sound for motivational
        audio = 0.5 * np.sin(2 * np.pi * 330 * t) * (0.8 + 0.2 * np.sin(2 * np.pi * 2 * t))
    else:  # Spoken Word or default
        audio = 0.5 * np.sin(2 * np.pi * 220 * t)
    
    # Apply fade in/out
    fade_duration = int(0.1 * sample_rate)
    fade_in = np.linspace(0, 1, fade_duration)
    fade_out = np.linspace(1, 0, fade_duration)
    
    audio[:fade_duration] *= fade_in
    audio[-fade_duration:] *= fade_out
    
    # Save the audio file
    sf.write(output_path, audio, sample_rate)
    
    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "file_size": os.path.getsize(output_path),
        "voice_id": "synthetic",
        "style": style
    }
