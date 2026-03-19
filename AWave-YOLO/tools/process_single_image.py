import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from yolo import YOLO

if __name__ == "__main__":
    # Specify weight file path
    model_path = 'd:\\waveformer-yolov8-pytorch-master - copy\\logs_voc_full\\best_epoch_weights.pth'
    # Specify category file path
    classes_path = 'model_data/voc_classes.txt'
    
    # Initialize YOLO model
    yolo = YOLO(model_path=model_path, classes_path=classes_path)
    
    # Specify the image path to be processed
    image_path = 'd:\\waveformer-yolov8-pytorch-master - copy\\VOCdevkit\\VOC2007\\JPEGImages\\IMG_7752.JPG'
    # Specify output path
    output_path = 'd:\\waveformer-yolov8-pytorch-master - copy\\batch_detect_output\\IMG_7752.JPG'
    
    print(f"Processing images: {image_path}")
    
    try:
        # Open picture
        image = Image.open(image_path)
        # Detect pictures
        r_image = yolo.detect_image(image)
        # Save results
        r_image.save(output_path)
        print(f"Save the results to: {output_path}")
        print("Processing completed!")
    except Exception as e:
        print(f"Error while processing image: {e}")
