import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.utils import get_classes
from utils.dataloader import YoloDataset, yolo_dataset_collate
from torch.utils.data import DataLoader
from nets.yolo import YoloBody
from utils.utils_bbox import DecodeBox
from utils.utils_map import file_lines_to_list

# Generate confusion matrix heatmap
def generate_confusion_matrix():
    print("Generating confusion matrix...")
    
    # Configuration parameters
    classes_path = 'model_data/voc_classes.txt'
    input_shape = [640, 640]
    phi = 's'
    save_dir = '.'
    val_annotation_path = '2007_val.txt'
    map_out_path = '.temp_confusion_matrix'
    
    # Get class information
    class_names, num_classes = get_classes(classes_path)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YoloBody(input_shape, num_classes, phi, pretrained=False)
    
    # Load best weights
    best_weights_path = os.path.join('logs', 'best_epoch_weights.pth')
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path))
        print(f"Best weights loaded: {best_weights_path}")
    
    model.to(device)
    model.eval()
    
    # Initialize decoding tool
    bbox_util = DecodeBox(num_classes, (input_shape[0], input_shape[1]))
    
    # Create validation set loader
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    
    val_dataset = YoloDataset(val_lines, input_shape, num_classes, epoch_length=1,
                            mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
    
    gen_val = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True,
                        drop_last=False, collate_fn=yolo_dataset_collate)
    
    # Create temporary directory
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, "ground-truth")):
        os.makedirs(os.path.join(map_out_path, "ground-truth"))
    if not os.path.exists(os.path.join(map_out_path, "detection-results")):
        os.makedirs(os.path.join(map_out_path, "detection-results"))
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for iteration, batch in enumerate(gen_val):
            if iteration >= len(val_lines):
                break
                
            images, targets = batch
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            outputs = bbox_util.decode_box(outputs)
            
            # Non-maximum suppression
            results = bbox_util.non_max_suppression(outputs, num_classes, input_shape,
                        (input_shape[0], input_shape[1]), False, conf_thres=0.1, nms_thres=0.5)
            
            # Get image path
            image_path = val_lines[iteration].split()[0]
            image_id = os.path.basename(image_path).split('.')[0]
            
            # Save ground truth labels to file
            gt_file = os.path.join(map_out_path, "ground-truth", f"{image_id}.txt")
            with open(gt_file, "w") as f:
                # Print debug info, check targets structure
                print(f"\nIteration {iteration}:")
                print(f"Targets type: {type(targets)}")
                print(f"Targets length: {len(targets) if hasattr(targets, '__len__') else 'N/A'}")
                
                if isinstance(targets, list):
                    print(f"Targets is list with {len(targets)} elements")
                    for i, target in enumerate(targets):
                        print(f"Target {i} type: {type(target)}")
                        print(f"Target {i} shape: {target.shape if hasattr(target, 'shape') else 'N/A'}")
                        print(f"Target {i} numel: {target.numel() if hasattr(target, 'numel') else 'N/A'}")
                        
                        if isinstance(target, torch.Tensor) and target.numel() > 0:
                            print(f"Target {i} dim: {target.dim()}")
                            if target.dim() == 2:
                                print(f"Target {i} shape: {target.shape}")
                                for j in range(target.shape[0]):
                                    # Print detailed info of each element
                                    print(f"  Element {j}: {target[j]}")
                                    if target[j].numel() >= 5:
                                        class_idx = int(target[j][-1])
                                        print(f"  Class index: {class_idx}")
                                        if class_idx < len(class_names):
                                            class_name = class_names[class_idx]
                                            print(f"  Class name: {class_name}")
                                            y_true.append(class_idx)
                            elif target.dim() == 1:
                                print(f"Target {i} elements: {target}")
                                if target.numel() >= 5:
                                    class_idx = int(target[-1])
                                    print(f"  Class index: {class_idx}")
                                    if class_idx < len(class_names):
                                        class_name = class_names[class_idx]
                                        print(f"  Class name: {class_name}")
                                        y_true.append(class_idx)
                elif isinstance(targets, torch.Tensor):
                    print(f"Targets is tensor with shape: {targets.shape}")
                    print(f"Targets elements: {targets}")
                    if targets.numel() >= 6:
                        if targets.dim() == 2:
                            for i in range(targets.shape[0]):
                                # Correctly parse label format: [0.0, class_idx, x, y, w, h]
                                class_idx = int(targets[i][1])
                                print(f"  Class index: {class_idx}")
                                if class_idx < len(class_names):
                                    class_name = class_names[class_idx]
                                    print(f"  Class name: {class_name}")
                                    y_true.append(class_idx)
                        elif targets.dim() == 1:
                            # Correctly parse label format: [0.0, class_idx, x, y, w, h]
                            class_idx = int(targets[1])
                            print(f"  Class index: {class_idx}")
                            if class_idx < len(class_names):
                                class_name = class_names[class_idx]
                                print(f"  Class name: {class_name}")
                                y_true.append(class_idx)
            
            # Save detection results to file
            dr_file = os.path.join(map_out_path, "detection-results", f"{image_id}.txt")
            with open(dr_file, "w") as f:
                if results[0] is not None:
                    for result in results[0]:
                        if len(result) > 5:
                            left, top, right, bottom, conf, class_idx = result
                            class_name = class_names[int(class_idx)]
                            f.write(f"{class_name} {conf} {left} {top} {right} {bottom}\n")
                            y_pred.append(int(class_idx))
    
    # Calculate confusion matrix
    print(f"y_true length: {len(y_true)}")
    print(f"y_pred length: {len(y_pred)}")
    print(f"y_true sample: {y_true[:10]}")
    print(f"y_pred sample: {y_pred[:10]}")
    
    # Ensure y_true and y_pred have the same length
    min_length = min(len(y_true), len(y_pred))
    if min_length > 0:
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    else:
        # If conditionally no labels, create an empty confusion matrix
        cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Print confusion matrix data
    print("Confusion Matrix:")
    print(cm)
    
    # Draw confusion matrix heatmap, ensure numbers are displayed completely
    plt.figure(figsize=(20, 16))  # Enlarge figure size
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=0, values_format='d')  # Use 'd' format to display integers
    
    # Adjust font size and layout
    plt.title('Confusion Matrix', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adjust colorbar and layout
    plt.tight_layout()
    
    # Save image
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {os.path.join(save_dir, 'confusion_matrix.png')}")
    
    # Clean up temporary files
    import shutil
    if os.path.exists(map_out_path):
        shutil.rmtree(map_out_path)
    
    print("Confusion matrix generation completed!")

if __name__ == "__main__":
    generate_confusion_matrix()