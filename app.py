import os
import logging
import tempfile
import uuid
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from services.audio_processor import analyze_audio
from services.tts_service import generate_vocals
from services.sync_service import synchronize_audio
from services.mix_service import mix_audio_tracks

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav'}
ALLOWED_TEXT_EXTENSIONS = {'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_unique_filename(filename):
    """Generate a unique filename while preserving extension."""
    basename, extension = os.path.splitext(filename)
    return f"{basename}_{uuid.uuid4().hex}{extension}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload/instrumental', methods=['POST'])
def upload_instrumental():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        return jsonify({'error': f'File type not allowed. Please upload: {", ".join(ALLOWED_AUDIO_EXTENSIONS)}'}), 400
    
    try:
        # Save the uploaded file with a unique name
        filename = secure_filename(file.filename)
        unique_filename = get_unique_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Analyze the audio file (BPM, beat grid, etc.)
        analysis_result = analyze_audio(file_path)
        
        # Store path and analysis in session
        session['instrumental_path'] = file_path
        session['instrumental_analysis'] = analysis_result
        
        return jsonify({
            'message': 'Instrumental uploaded successfully',
            'filename': filename,
            'bpm': analysis_result['bpm'],
            'duration': analysis_result['duration'],
            'beat_positions': analysis_result['beat_positions'][:10]  # Just sending first 10 beats for UI
        })
    
    except Exception as e:
        logger.error(f"Error processing instrumental: {str(e)}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/upload/text', methods=['POST'])
def upload_text():
    # Handle direct text input
    if 'text' in request.form:
        text_content = request.form['text']
        
        # Store in session
        session['lyrics'] = text_content
        
        return jsonify({
            'message': 'Text content received',
            'wordCount': len(text_content.split()),
            'charCount': len(text_content)
        })
    
    # Handle file upload
    elif 'file' in request.files:
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, ALLOWED_TEXT_EXTENSIONS):
            return jsonify({'error': f'File type not allowed. Please upload: {", ".join(ALLOWED_TEXT_EXTENSIONS)}'}), 400
        
        try:
            # Read text content
            text_content = file.read().decode('utf-8')
            
            # Store in session
            session['lyrics'] = text_content
            
            return jsonify({
                'message': 'Text file processed',
                'filename': file.filename,
                'wordCount': len(text_content.split()),
                'charCount': len(text_content)
            })
        
        except Exception as e:
            logger.error(f"Error processing text file: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    else:
        return jsonify({'error': 'No text content provided'}), 400

@app.route('/generate-vocals', methods=['POST'])
def generate_vocals_endpoint():
    if 'lyrics' not in session:
        return jsonify({'error': 'No lyrics found. Please upload text first.'}), 400
    
    try:
        voice_id = request.form.get('voice', 'default')
        style = request.form.get('style', 'Spoken Word')
        
        # Generate a unique file path for the output
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"vocals_{uuid.uuid4().hex}.wav")
        
        # Call TTS service to generate vocals
        generate_result = generate_vocals(
            text=session['lyrics'],
            voice_id=voice_id,
            style=style,
            output_path=output_path
        )
        
        # Store the output path in session
        session['vocals_path'] = output_path
        
        return jsonify({
            'message': 'Vocals generated successfully',
            'duration': generate_result['duration'],
            'samplingRate': generate_result['sample_rate']
        })
        
    except Exception as e:
        logger.error(f"Error generating vocals: {str(e)}")
        return jsonify({'error': f'Error generating vocals: {str(e)}'}), 500

@app.route('/synchronize', methods=['POST'])
def synchronize_endpoint():
    if 'instrumental_path' not in session or 'vocals_path' not in session:
        return jsonify({'error': 'Missing audio files. Please upload instrumental and generate vocals first.'}), 400
    
    try:
        # Get synchronization parameters
        params = {
            'alignment_mode': request.form.get('alignmentMode', 'beat'),
            'time_stretch': request.form.get('timeStretch', 'true').lower() == 'true',
            'timing_adjustments': request.json.get('timingAdjustments', []) if request.is_json else []
        }
        
        # Generate a unique file path for synchronized vocals
        synced_output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"synced_vocals_{uuid.uuid4().hex}.wav")
        
        # Call synchronization service
        sync_result = synchronize_audio(
            vocal_path=session['vocals_path'],
            instrumental_path=session['instrumental_path'],
            instrumental_analysis=session['instrumental_analysis'],
            output_path=synced_output_path,
            params=params
        )
        
        # Store the synchronized vocals path in session
        session['synced_vocals_path'] = synced_output_path
        
        return jsonify({
            'message': 'Vocals synchronized successfully',
            'syncedDuration': sync_result['duration'],
            'beatAlignmentPoints': sync_result['alignment_points'][:10]  # Just sending first 10 points
        })
        
    except Exception as e:
        logger.error(f"Error synchronizing audio: {str(e)}")
        return jsonify({'error': f'Error synchronizing audio: {str(e)}'}), 500

@app.route('/mix', methods=['POST'])
def mix_endpoint():
    if 'instrumental_path' not in session or 'synced_vocals_path' not in session:
        return jsonify({'error': 'Missing synchronized audio files. Please complete previous steps first.'}), 400
    
    try:
        # Get mixing parameters
        params = {
            'vocal_level': float(request.form.get('vocalLevel', 0.8)),
            'eq_preset': request.form.get('eqPreset', 'balanced'),
            'compression': request.form.get('compression', 'medium'),
            'stereo_width': float(request.form.get('stereoWidth', 1.0)),
            'auto_ducking': request.form.get('autoDucking', 'true').lower() == 'true'
        }
        
        # Generate a unique file path for the final mix
        final_output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"final_mix_{uuid.uuid4().hex}.wav")
        
        # Call mixing service
        mix_result = mix_audio_tracks(
            vocal_path=session['synced_vocals_path'],
            instrumental_path=session['instrumental_path'],
            output_path=final_output_path,
            params=params
        )
        
        # Store the final mix path in session
        session['final_mix_path'] = final_output_path
        
        return jsonify({
            'message': 'Mix completed successfully',
            'duration': mix_result['duration'],
            'fileSize': mix_result['file_size'],
            'peakLevel': mix_result['peak_level']
        })
        
    except Exception as e:
        logger.error(f"Error mixing audio: {str(e)}")
        return jsonify({'error': f'Error mixing audio: {str(e)}'}), 500

@app.route('/download/<file_type>')
def download_file(file_type):
    file_path = None
    filename = None
    
    if file_type == 'vocals' and 'vocals_path' in session:
        file_path = session['vocals_path']
        filename = 'generated_vocals.wav'
    elif file_type == 'synced' and 'synced_vocals_path' in session:
        file_path = session['synced_vocals_path']
        filename = 'synchronized_vocals.wav'
    elif file_type == 'final' and 'final_mix_path' in session:
        file_path = session['final_mix_path']
        filename = 'final_mix.wav'
    else:
        return jsonify({'error': 'Invalid file type or file not found'}), 404
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        return send_file(file_path, as_attachment=True, download_name=filename)
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500

@app.route('/get-file/<file_type>')
def get_file(file_type):
    """Get file path for playing in browser"""
    file_path = None
    
    if file_type == 'instrumental' and 'instrumental_path' in session:
        file_path = session['instrumental_path']
    elif file_type == 'vocals' and 'vocals_path' in session:
        file_path = session['vocals_path']
    elif file_type == 'synced' and 'synced_vocals_path' in session:
        file_path = session['synced_vocals_path']
    elif file_type == 'final' and 'final_mix_path' in session:
        file_path = session['final_mix_path']
    else:
        return jsonify({'error': 'Invalid file type or file not found'}), 404
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        return send_file(file_path)
    except Exception as e:
        logger.error(f"Error serving file: {str(e)}")
        return jsonify({'error': f'Error serving file: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
