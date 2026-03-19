import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch
import colorsys
from nets.yolo import YoloBody
from utils.utils import get_classes
from utils.utils_bbox import DecodeBox

# Configuration parameters
classes_path = 'model_data/voc_classes.txt'
model_path = 'logs_voc_full/best_epoch_weights.pth'
input_shape = [640, 640]
phi = 's'
confidence = 0.3  # Lower the confidence threshold to ensure targets can be detected
nms_iou = 0.5
# Get class information
class_names, num_classes = get_classes(classes_path)

# Load model
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YoloBody(input_shape, num_classes, phi, pretrained=False)
    
    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Weights loaded: {model_path}")
    else:
        print(f"Weights file not found: {model_path}")
        # Try loading weights from the original logs directory
        alt_model_path = 'logs/best_epoch_weights.pth'
        if os.path.exists(alt_model_path):
            model.load_state_dict(torch.load(alt_model_path, map_location=device))
            print(f"Alternate weights loaded: {alt_model_path}")
        else:
            raise FileNotFoundError("No weights file found")
    
    model.eval()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model = model.cuda()
    
    return model, device

# Detect image
def detect_image(model, device, image_path, bbox_util):
    # Read image
    image = Image.open(image_path)
    image = image.convert('RGB')
    image_shape = np.array(np.shape(image)[0:2])
    
    # Preprocessing
    from utils.utils import cvtColor, resize_image, preprocess_input
    image = cvtColor(image)
    image_data = resize_image(image, (input_shape[1], input_shape[0]), True)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    
    # Forward pass
    with torch.no_grad():
        images = torch.from_numpy(image_data).to(device)
        outputs = model(images)
        outputs = bbox_util.decode_box(outputs)
        
        # Non-maximum suppression
        results = bbox_util.non_max_suppression(outputs, num_classes, input_shape,
                                            image_shape, True, conf_thres=confidence, nms_thres=nms_iou)
    
    return image, results

# Draw detection results
def draw_detections(image, results, image_name):
    draw = ImageDraw.Draw(image)
    
    # Add image name
    try:
        font = ImageFont.truetype("model_data/simhei.ttf", 200)
    except:
        font = ImageFont.load_default()
    
    # Draw image name
    draw.text((10, 10), image_name, fill=(255, 0, 0), font=font)
    
    # Create a 5x scaled font for recognition result labels
    try:
        label_font = ImageFont.truetype("model_data/simhei.ttf", 1000)
    except:
        label_font = ImageFont.load_default()
    
    # Draw bounding boxes
    if results[0] is not None:
        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]
        
        # Generate colors
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: tuple(map(lambda y: int(y * 255), colorsys.hsv_to_rgb(*x))), hsv_tuples))
        colors = np.array(colors, dtype=np.uint8)
        
        for i, c in list(enumerate(top_label)):
            predicted_class = class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]
            
            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            
            # Draw borders
            draw.rectangle([left, top, right, bottom], outline=tuple(colors[c]), width=30)
            
            # Draw labels - only show class name (uppercase), remove confidence and background color block
            label = f'{predicted_class.upper()}'
            # Calculate text bounding box
            bbox = draw.textbbox((0, 0), label, font=label_font)
            label_width = bbox[2] - bbox[0]
            label_height = bbox[3] - bbox[1]
            # Place in the upper right corner of the image
            label_x = image.size[0] - label_width - 20
            label_y = 100  # Ensure it does not overlap with the image name
            # Only draw text, do not draw background color block
            draw.text([label_x, label_y], label, fill=(255, 255, 255), font=label_font)
    
    del draw
    return image

# Get images 121-150 from the training set
def get_train_images():
    # Read images directly from the JPEGImages directory
    image_dir = 'VOCdevkit/VOC2007/JPEGImages'
    
    # Get all images
    image_list = []
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.JPG')):
            image_path = os.path.join(image_dir, filename)
            image_list.append(image_path)
    
    # Sort by filename to ensure consistent order
    image_list.sort()
    
    # Return 121st to 150th images
    return image_list[120:150]

# Main function
def main():
    import colorsys
    
    # Load model
    model, device = load_model()
    
    # Initialize decoding tool
    bbox_util = DecodeBox(num_classes, (input_shape[0], input_shape[1]))
    
    # Get the first thirty images
    image_paths = get_train_images()
    print(f"Got {len(image_paths)} training images")
    
    # Detect each image
    detected_images = []
    for i, image_path in enumerate(image_paths):
        print(f"Detecting image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        image, results = detect_image(model, device, image_path, bbox_util)
        
        # Draw detection results
        image_name = os.path.basename(image_path)
        detected_image = draw_detections(image, results, image_name)
        detected_images.append(detected_image)
    
    # Create grid layout - 30 images use 5 rows and 6 columns layout
    rows = 5
    cols = 6
    
    # Create large image - use reasonable size to ensure it can be opened
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    
    # Draw each image
    for i, img in enumerate(detected_images):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(img)
        ax.axis('off')
        # Add image index label
        ax.set_title(f'Image {i+1}', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save results - use reasonable dpi to ensure it can be opened, save as new file
    output_path = 'detection_results_grid_121_150.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Detection results saved to: {output_path}")

if __name__ == "__main__":
    main()
