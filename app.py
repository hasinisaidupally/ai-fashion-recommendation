from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
from werkzeug.utils import secure_filename
from recommender import FashionRecommender
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Allowed file extensions (updated to include webp for better compatibility)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('dataset', exist_ok=True)

# Initialize the recommender system
recommender = FashionRecommender()

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and generate recommendations"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Allowed types: png, jpg, jpeg, gif, bmp, webp'}), 400
    
    try:
        # Generate unique filename
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Generate recommendations using the upgraded AI system
        start_time = time.time()
        recommendations = recommender.find_similar_items(filepath, top_k=5)  # Updated to top_k=5 as requested
        processing_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            'success': True,
            'uploaded_image': f'/uploads/{filename}',
            'recommendations': [],
            'processing_time': round(processing_time, 2)
        }
        
        # Process recommendations
        for rec in recommendations:
            # Convert dataset path to web-accessible URL
            rel_path = os.path.relpath(rec['image_path'], 'dataset')
            response_data['recommendations'].append({
                'image_url': f'/dataset/{rel_path}',
                'similarity': round(rec['similarity'] * 100, 2)
            })
        
        # If no recommendations found, create sample images and try again
        if not response_data['recommendations']:
            print("🎭 No recommendations found - creating sample images for demonstration")
            recommender.add_sample_images()
            recommender.load_dataset()
            recommendations = recommender.find_similar_items(filepath, top_k=5)  # Updated to top_k=5
            
            for rec in recommendations:
                rel_path = os.path.relpath(rec['image_path'], 'dataset')
                response_data['recommendations'].append({
                    'image_url': f'/dataset/{rel_path}',
                    'similarity': round(rec['similarity'] * 100, 2)
                })
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/dataset/<path:filename>')
def dataset_file(filename):
    """Serve dataset files"""
    return send_from_directory('dataset', filename)

@app.route('/api/dataset-info')
def dataset_info():
    """Get information about the dataset"""
    try:
        dataset_path = 'dataset'
        if not os.path.exists(dataset_path):
            return jsonify({'count': 0, 'message': 'Dataset directory not found'})
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_count = 0
        
        for ext in image_extensions:
            image_count += len(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
            image_count += len(glob.glob(os.path.join(dataset_path, '**', ext.upper()), recursive=True))
        
        return jsonify({
            'count': image_count,
            'path': dataset_path,
            'message': f'Dataset contains {image_count} images'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting AI Fashion Recommendation System...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("Dataset folder: dataset")

    # Just load dataset normally
    recommender.load_dataset()

    app.run(debug=True)
