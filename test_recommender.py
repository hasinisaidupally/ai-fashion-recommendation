import os
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import glob
import time

class FashionRecommender:
    """
    Simplified Fashion Recommendation System for testing
    Uses basic image features instead of deep learning
    """
    
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        self.dataset_features = None
        self.dataset_images = None
        self.load_dataset()
    
    def extract_basic_features(self, img_path):
        """Extract basic color and texture features"""
        try:
            # Load and resize image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((64, 64))  # Smaller size for faster processing
            img_array = np.array(img)
            
            # Extract color histogram features
            hist_r = np.histogram(img_array[:,:,0].flatten(), bins=16, range=(0, 256))[0]
            hist_g = np.histogram(img_array[:,:,1].flatten(), bins=16, range=(0, 256))[0]
            hist_b = np.histogram(img_array[:,:,2].flatten(), bins=16, range=(0, 256))[0]
            
            # Combine features
            features = np.concatenate([hist_r, hist_g, hist_b])
            
            # Normalize
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features from {img_path}: {e}")
            return None
    
    def load_dataset(self):
        """Load and process all images in the dataset"""
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset directory {self.dataset_path} does not exist")
            return
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
        image_paths = []
        
        print("🔍 Scanning dataset for images...")
        for ext in image_extensions:
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
        
        # Extract features from each image
        print("🧠 Extracting features from dataset images...")
        start_time = time.time()
        
        for i, img_path in enumerate(image_paths):
            if i % 10 == 0:
                print(f"   Processing {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            features = self.extract_basic_features(img_path)
            if features is not None:
                features_list.append(features)
        
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
        """Find similar items using basic features"""
        if self.dataset_features.size == 0:
            print("❌ No dataset features available for comparison")
            return []
        
        # Extract features from query image
        print(f"🔍 Analyzing query image: {os.path.basename(query_img_path)}")
        query_features = self.extract_basic_features(query_img_path)
        
        if query_features is None:
            print("❌ Could not extract features from query image")
            return []
        
        # Compute cosine similarity
        similarities = cosine_similarity([query_features], self.dataset_features)[0]
        
        # Get top-k most similar items
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        recommendations = []
        for rank, idx in enumerate(top_indices):
            if idx < len(self.dataset_images):
                similarity_score = float(similarities[idx])
                
                if similarity_score > 0.1:  # Threshold
                    recommendations.append({
                        'image_path': self.dataset_images[idx],
                        'similarity': similarity_score,
                        'rank': rank + 1
                    })
        
        print(f"🎯 Found {len(recommendations)} similar items")
        if recommendations:
            print(f"📈 Top similarity: {recommendations[0]['similarity']:.3f}")
        
        return recommendations
    
    def add_sample_images(self):
        """Create sample fashion images"""
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        
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
            img = np.zeros((300, 200, 3), dtype=np.uint8)
            
            if i % 3 == 0:  # T-shirt
                cv2.rectangle(img, (50, 50), (150, 200), color, -1)
                cv2.rectangle(img, (30, 50), (170, 80), color, -1)
                item_type = "tshirt"
            elif i % 3 == 1:  # Pants
                cv2.rectangle(img, (70, 50), (130, 150), color, -1)
                cv2.rectangle(img, (70, 150), (90, 250), color, -1)
                cv2.rectangle(img, (110, 150), (130, 250), color, -1)
                item_type = "pants"
            else:  # Dress
                cv2.rectangle(img, (60, 50), (140, 100), color, -1)
                points = np.array([[60, 100], [140, 100], [130, 250], [70, 250]], np.int32)
                cv2.fillPoly(img, [points], color)
                item_type = "dress"
            
            filename = f"fashion_{item_type}_{i+1:02d}.jpg"
            filepath = os.path.join(self.dataset_path, filename)
            cv2.imwrite(filepath, img)
        
        print(f"✅ Created {len(colors)} sample fashion images in dataset")
    
    def get_dataset_info(self):
        """Get dataset information"""
        if self.dataset_features is None:
            return {
                'image_count': 0,
                'feature_matrix_shape': None,
                'dataset_path': self.dataset_path,
                'model_loaded': True
            }
        
        return {
            'image_count': len(self.dataset_images) if self.dataset_images else 0,
            'feature_matrix_shape': self.dataset_features.shape,
            'dataset_path': self.dataset_path,
            'model_loaded': True,
            'feature_dimension': 48  # 16 bins * 3 channels
        }
