import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from PIL import Image
from yolo import YOLO

if __name__ == "__main__":
    # Specify the weight file path
    model_path = 'd:\\waveformer-yolov8-pytorch-master - copy\\logs_voc_full\\best_epoch_weights.pth'
    # Specify the classes file path
    classes_path = 'model_data/voc_classes.txt'
    
    # Initialize YOLO model, set confidence to 0.4
    yolo = YOLO(model_path=model_path, classes_path=classes_path, confidence=0.4)
    
    # Specify the image directory
    image_dir = 'd:\\waveformer-yolov8-pytorch-master - copy\\detection Q'
    # Specify the output directory
    output_dir = 'batch_detect_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    # Process all images in this directory
    # No need to slice, use all images directly
    
    print(f"Start processing {len(image_files)} images...")
    
    # Process each image iteratively
    for i, img_name in enumerate(image_files):
        print(f"Processing image {i+1}: {img_name}")
        
        # Build the image path
        img_path = os.path.join(image_dir, img_name)
        
        try:
            # Open the image
            image = Image.open(img_path)
            # Detect targets in the image
            r_image = yolo.detect_image(image)
            # Save the detection result
            output_path = os.path.join(output_dir, img_name)
            r_image.save(output_path)
            print(f"Result saved to: {output_path}")
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
    
    print("Batch detection completed!")
