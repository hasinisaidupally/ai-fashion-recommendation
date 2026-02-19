# AI Fashion Recommendation System

A complete Flask web application that uses AI to recommend similar fashion items based on uploaded clothing images.

## Features

- **Image Upload**: Drag & drop or click to upload clothing images
- **AI-Powered Recommendations**: Uses MobileNetV2 for feature extraction and cosine similarity for matching
- **Modern UI**: Beautiful, responsive interface with smooth animations
- **Real-time Processing**: Fast similarity matching with processing time display
- **Auto-Sample Generation**: Creates sample fashion items if dataset is empty

## Project Structure

```
ai-fashion-recommendation/
├── app.py                 # Main Flask application
├── recommender.py         # AI recommendation engine
├── requirements.txt       # Python dependencies
├── dataset/              # Fashion item images (auto-populated if empty)
├── uploads/              # User uploaded images
├── templates/
│   └── index.html        # Frontend interface
└── README.md            # This file
```

## Installation

1. **Install Python** (3.8 or higher)

2. **Clone or download** the project files

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open in browser**: Navigate to `http://localhost:5000`

## Usage

1. **Upload an Image**: Click the upload area or drag & drop a clothing image
2. **Find Similar Items**: Click "Find Similar Items" to process the image
3. **View Results**: See recommended similar fashion items with similarity scores

## Supported Image Formats

- PNG
- JPG/JPEG
- GIF
- BMP

Maximum file size: 16MB

## How It Works

1. **Feature Extraction**: Uses MobileNetV2 (pre-trained on ImageNet) to extract visual features from images
2. **Similarity Matching**: Compares features using cosine similarity to find the most similar items
3. **Ranking**: Returns top 6 most similar items with similarity percentages

## Dataset

- The system automatically creates sample fashion items if the `dataset/` folder is empty
- Add your own fashion images to the `dataset/` folder for better recommendations
- Supports nested folder structures within the dataset directory

## Technical Details

- **Backend**: Flask web framework
- **AI Model**: MobileNetV2 for feature extraction
- **Similarity Algorithm**: Cosine similarity
- **Frontend**: HTML5, CSS3, JavaScript (no external frameworks)
- **Image Processing**: OpenCV, PIL, TensorFlow

## API Endpoints

- `GET /` - Main page
- `POST /upload` - Upload image and get recommendations
- `GET /uploads/<filename>` - Serve uploaded files
- `GET /dataset/<path:filename>` - Serve dataset files
- `GET /api/dataset-info` - Get dataset information

## Customization

- **Change similarity threshold**: Modify the `find_similar_items` method in `recommender.py`
- **Adjust number of recommendations**: Change the `top_k` parameter
- **Custom UI**: Modify `templates/index.html`
- **Add more image formats**: Update `ALLOWED_EXTENSIONS` in `app.py`

## Troubleshooting

1. **TensorFlow Installation Issues**: 
   - Try installing CPU-only version: `pip install tensorflow-cpu`
   - Or use conda: `conda install tensorflow`

2. **Memory Issues**:
   - Reduce dataset size
   - Use smaller image dimensions

3. **Port Already in Use**:
   - Change port in `app.py`: `app.run(port=5001)`

## License

This project is for educational purposes. Feel free to modify and distribute.
