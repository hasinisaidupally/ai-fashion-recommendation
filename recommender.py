import os
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import glob
import time

class FashionRecommender:
    """
    AI-Powered Fashion Recommendation System
    
    This class uses MobileNetV2 (pre-trained on ImageNet) to extract deep features
    from clothing images and find similar items using cosine similarity.
    
    Key Features:
    - Uses MobileNetV2 for feature extraction (1280-dimensional embeddings)
    - Cosine similarity for comparing image embeddings
    - Efficient batch processing for dataset loading
    - Model loaded once for optimal performance
    """
    
    def __init__(self, dataset_path='dataset'):
        """
        Initialize the Fashion Recommender system
        
        Args:
            dataset_path (str): Path to the directory containing fashion images
        """
        self.dataset_path = dataset_path
        self.model = None                    # MobileNetV2 model (loaded once)
        self.dataset_features = None         # Feature vectors for all dataset images
        self.dataset_images = None           # Paths to dataset images
        self.feature_dimension = 1280        # MobileNetV2 output dimension
        
        # Initialize the system
        self.load_model()
        self.load_dataset()
    
    def load_model(self):
        """
        Load the pre-trained MobileNetV2 model for feature extraction
        
        Model Configuration:
        - Weights: ImageNet (1.4M images, 1000 classes)
        - Architecture: MobileNetV2 (efficient for mobile/web deployment)
        - Output: Global average pooling (1280-dimensional feature vectors)
        - Input: 224x224 RGB images
        
        The model is loaded only once during initialization to ensure optimal performance.
        """
        try:
            # Load MobileNetV2 without the top classification layer
            # include_top=False removes the final 1000-class softmax layer
            # pooling='avg' applies global average pooling to output a feature vector
            self.model = MobileNetV2(
                weights='imagenet',      # Pre-trained on ImageNet dataset
                include_top=False,       # Remove classification layer
                pooling='avg',           # Global average pooling
                input_shape=(224, 224, 3)  # Standard input size
            )
            
            print("✅ MobileNetV2 model loaded successfully")
            print(f"📊 Feature vector dimension: {self.feature_dimension}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, img_path, target_size=(224, 224)):
        """
        Preprocess image for MobileNetV2 input requirements
        
        Steps:
        1. Load image and convert to RGB format
        2. Resize to target dimensions (224x224)
        3. Convert to numpy array
        4. Add batch dimension
        5. Apply MobileNetV2-specific preprocessing
        
        Args:
            img_path (str): Path to the image file
            target_size (tuple): Target dimensions for resizing
            
        Returns:
            numpy.ndarray: Preprocessed image array ready for model input
                          Returns None if preprocessing fails
        """
        try:
            # Step 1: Load image and ensure RGB format
            img = Image.open(img_path).convert('RGB')
            
            # Step 2: Resize to MobileNetV2 input size
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Step 3: Convert to numpy array (float32)
            img_array = image.img_to_array(img, dtype=np.float32)
            
            # Step 4: Add batch dimension (shape: [1, 224, 224, 3])
            img_array = np.expand_dims(img_array, axis=0)
            
            # Step 5: Apply MobileNetV2 preprocessing
            # This scales pixel values to [-1, 1] range as expected by MobileNetV2
            img_array = preprocess_input(img_array)
            
            return img_array
            
        except Exception as e:
            print(f"❌ Error preprocessing image {img_path}: {e}")
            return None
    
    def extract_features(self, img_path):
        """
        Extract deep features (embeddings) from an image using MobileNetV2
        
        The extracted features represent the visual characteristics of the image
        in a 1280-dimensional vector space. These embeddings capture:
        - Colors and textures
        - Shapes and patterns
        - Object parts and spatial relationships
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: 1280-dimensional feature vector
                          Returns None if extraction fails
        """
        try:
            # Preprocess the image for model input
            img_array = self.preprocess_image(img_path)
            if img_array is None:
                return None
            
            # Extract features using the pre-trained model
            # The model outputs a 1280-dimensional embedding
            features = self.model.predict(img_array, verbose=0)
            
            # Flatten to 1D array for easier comparison
            feature_vector = features.flatten()
            
            # Normalize the feature vector to unit length
            # This improves cosine similarity calculations
            feature_vector = feature_vector / np.linalg.norm(feature_vector)
            
            return feature_vector
            
        except Exception as e:
            print(f"❌ Error extracting features from {img_path}: {e}")
            return None
    
    def load_dataset(self):
        """
        Load and process all images in the dataset for efficient similarity search
        
        This method:
        1. Scans the dataset directory for image files
        2. Extracts features from each image using MobileNetV2
        3. Stores feature vectors in memory for fast similarity computation
        
        The feature extraction is done once during initialization to ensure
        real-time performance during recommendation queries.
        """
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset directory {self.dataset_path} does not exist")
            return
        
        # Find all image files in the dataset (recursive search)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
        image_paths = []
        
        print("🔍 Scanning dataset for images...")
        for ext in image_extensions:
            # Search for both lowercase and uppercase extensions
            image_paths.extend(glob.glob(os.path.join(self.dataset_path, '**', ext), recursive=True))
            image_paths.extend(glob.glob(os.path.join(self.dataset_path, '**', ext.upper()), recursive=True))
        
        if not image_paths:
            print("⚠️  No images found in dataset directory")
            self.dataset_images = []
            self.dataset_features = np.array([])
            return
        
        print(f"📁 Found {len(image_paths)} images in dataset")
        
        self.dataset_images = image_paths
        features_list = []
        
        # Extract features from each image in the dataset
        print("🧠 Extracting features from dataset images...")
        start_time = time.time()
        
        for i, img_path in enumerate(image_paths):
            if i % 10 == 0:  # Progress update every 10 images
                print(f"   Processing {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            features = self.extract_features(img_path)
            if features is not None:
                features_list.append(features)
        
        # Convert list of features to numpy array for efficient computation
        if features_list:
            self.dataset_features = np.array(features_list)
            processing_time = time.time() - start_time
            print(f"✅ Successfully extracted features from {len(features_list)} images")
            print(f"⏱️  Feature extraction completed in {processing_time:.2f} seconds")
            print(f"📊 Feature matrix shape: {self.dataset_features.shape}")
        else:
            self.dataset_features = np.array([])
            print("❌ No features could be extracted from dataset images")
    
    def find_similar_items(self, query_img_path, top_k=5):
        """
        Find the most similar fashion items in the dataset using AI-powered similarity
        
        This method:
        1. Extracts features from the query image
        2. Computes cosine similarity with all dataset images
        3. Returns top-k most similar items with similarity scores
        
        Cosine similarity measures the cosine of the angle between two vectors:
        - Range: [-1, 1] where 1 means identical, 0 means orthogonal, -1 means opposite
        - For normalized vectors: equivalent to dot product
        - Robust to vector magnitude, focuses on direction/similarity
        
        Args:
            query_img_path (str): Path to the uploaded query image
            top_k (int): Number of similar items to return (default: 5)
            
        Returns:
            list: List of dictionaries containing:
                  - 'image_path': Path to similar image
                  - 'similarity': Cosine similarity score (0-1)
                  Empty list if no recommendations found
        """
        # Check if dataset is available for comparison
        if self.dataset_features.size == 0:
            print("❌ No dataset features available for comparison")
            return []
        
        # Extract features from the query image
        print(f"🔍 Analyzing query image: {os.path.basename(query_img_path)}")
        query_features = self.extract_features(query_img_path)
        
        if query_features is None:
            print("❌ Could not extract features from query image")
            return []
        
        # Compute cosine similarity between query and all dataset items
        # Shape: [1, N] where N is number of dataset images
        similarities = cosine_similarity([query_features], self.dataset_features)[0]
        
        # Get indices of top-k most similar items (sorted by similarity)
        # np.argsort returns indices that would sort the array in ascending order
        # [::-1] reverses to get descending order (highest similarity first)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare recommendations with similarity scores
        recommendations = []
        for rank, idx in enumerate(top_indices):
            if idx < len(self.dataset_images):
                similarity_score = float(similarities[idx])
                
                # Only include items with meaningful similarity (threshold: 0.1)
                # This filters out very dissimilar items
                if similarity_score > 0.1:
                    recommendations.append({
                        'image_path': self.dataset_images[idx],
                        'similarity': similarity_score,
                        'rank': rank + 1  # 1-based ranking
                    })
        
        print(f"🎯 Found {len(recommendations)} similar items")
        if recommendations:
            print(f"📈 Top similarity: {recommendations[0]['similarity']:.3f}")
        
        return recommendations
    
    def add_sample_images(self):
        """
        Create sample fashion images for demonstration purposes
        
        This method generates simple geometric shapes representing different clothing items:
        - T-shirts (rectangle with sleeves)
        - Pants (rectangle with separate legs)
        - Dresses (rectangle with triangular bottom)
        
        These samples allow the system to demonstrate functionality even without
        a real fashion dataset. The AI can still find similarities based on:
        - Color differences
        - Shape variations
        - Spatial patterns
        
        Generated images are 300x200 pixels with different colors.
        """
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        
        # Define colors for different fashion items
        colors = [
            (255, 0, 0),    # Red - T-shirt
            (0, 255, 0),    # Green - Pants
            (0, 0, 255),    # Blue - Dress
            (255, 255, 0),  # Yellow - T-shirt
            (255, 0, 255),  # Magenta - Pants
            (0, 255, 255),  # Cyan - Dress
            (128, 128, 128), # Gray - T-shirt
            (255, 165, 0),  # Orange - Pants
        ]
        
        print("🎨 Creating sample fashion images...")
        
        for i, color in enumerate(colors):
            # Create blank canvas (300x200 pixels)
            img = np.zeros((300, 200, 3), dtype=np.uint8)
            
            # Create different clothing shapes based on index
            if i % 3 == 0:  # T-shirt shape
                # Main body
                cv2.rectangle(img, (50, 50), (150, 200), color, -1)
                # Sleeves
                cv2.rectangle(img, (30, 50), (170, 80), color, -1)
                item_type = "tshirt"
                
            elif i % 3 == 1:  # Pants shape
                # Waist to knees
                cv2.rectangle(img, (70, 50), (130, 150), color, -1)
                # Left leg
                cv2.rectangle(img, (70, 150), (90, 250), color, -1)
                # Right leg
                cv2.rectangle(img, (110, 150), (130, 250), color, -1)
                item_type = "pants"
                
            else:  # Dress shape
                # Top part
                cv2.rectangle(img, (60, 50), (140, 100), color, -1)
                # Triangular bottom
                points = np.array([[60, 100], [140, 100], [130, 250], [70, 250]], np.int32)
                cv2.fillPoly(img, [points], color)
                item_type = "dress"
            
            # Save the image with descriptive filename
            filename = f"fashion_{item_type}_{i+1:02d}.jpg"
            filepath = os.path.join(self.dataset_path, filename)
            cv2.imwrite(filepath, img)
        
        print(f"✅ Created {len(colors)} sample fashion images in dataset")
        print("📁 Images saved as: fashion_tshirt_XX.jpg, fashion_pants_XX.jpg, fashion_dress_XX.jpg")
    
    def get_dataset_info(self):
        """
        Get information about the current dataset
        
        Returns:
            dict: Dataset statistics including image count and feature matrix info
        """
        if self.dataset_features is None:
            return {
                'image_count': 0,
                'feature_matrix_shape': None,
                'dataset_path': self.dataset_path,
                'model_loaded': self.model is not None
            }
        
        return {
            'image_count': len(self.dataset_images) if self.dataset_images else 0,
            'feature_matrix_shape': self.dataset_features.shape,
            'dataset_path': self.dataset_path,
            'model_loaded': self.model is not None,
            'feature_dimension': self.feature_dimension
        }
