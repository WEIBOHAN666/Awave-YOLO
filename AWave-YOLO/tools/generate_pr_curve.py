import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.utils import get_classes
from utils.dataloader import YoloDataset, yolo_dataset_collate
from torch.utils.data import DataLoader
from nets.yolo import YoloBody
from utils.utils_bbox import DecodeBox
from utils.utils_map import get_map

# Generate PR curve and label AP

def generate_pr_curve():
    print("Generate PR curve...")
    
    # Configuration parameters
    classes_path = 'model_data/voc_classes.txt'
    input_shape = [640, 640]
    phi = 's'
    save_dir = '.'
    val_annotation_path = '2007_val.txt'
    map_out_path = '.temp_pr_curve'
    
    # Get category information
    class_names, num_classes = get_classes(classes_path)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YoloBody(input_shape, num_classes, phi, pretrained=False)
    
    # Load optimal weights
    best_weights_path = os.path.join('logs', 'best_epoch_weights.pth')
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path))
        print(f"The best weights have been loaded: {best_weights_path}")
    
    model.to(device)
    model.eval()
    
    # Initialize decoding tool
    bbox_util = DecodeBox(num_classes, (input_shape[0], input_shape[1]))
    
    # Create a validation set loader
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
    
    # Generate prediction results and ground truth label files
    with torch.no_grad():
        for iteration, batch in enumerate(gen_val):
            if iteration >= len(val_lines):
                break
                
            images, targets = batch
            images = images.to(device)
            
            # forward propagation
            outputs = model(images)
            outputs = bbox_util.decode_box(outputs)
            
            # non-maximum suppression
            results = bbox_util.non_max_suppression(outputs, num_classes, input_shape,
                        (input_shape[0], input_shape[1]), False, conf_thres=0.1, nms_thres=0.5)
            
            # Get image path
            image_path = val_lines[iteration].split()[0]
            image_id = os.path.basename(image_path).split('.')[0]
            
            # Save real labels to file
            gt_file = os.path.join(map_out_path, "ground-truth", f"{image_id}.txt")
            with open(gt_file, "w") as f:
                if isinstance(targets, list):
                    for target in targets:
                        if isinstance(target, torch.Tensor) and target.numel() > 0:
                            if target.dim() == 2:
                                for i in range(target.shape[0]):
                                    if target.shape[1] >= 6:
                                        class_idx = int(target[i, 1])
                                        if class_idx < len(class_names):
                                            class_name = class_names[class_idx]
                                            f.write(f"{class_name} {target[i, 2]} {target[i, 3]} {target[i, 4]} {target[i, 5]}\n")
                            elif target.dim() == 1:
                                if target.numel() >= 6:
                                    class_idx = int(target[1])
                                    if class_idx < len(class_names):
                                        class_name = class_names[class_idx]
                                        f.write(f"{class_name} {target[2]} {target[3]} {target[4]} {target[5]}\n")
                elif isinstance(targets, torch.Tensor) and targets.numel() > 0:
                    if targets.dim() == 2:
                        for i in range(targets.shape[0]):
                            if targets.shape[1] >= 6:
                                class_idx = int(targets[i, 1])
                                if class_idx < len(class_names):
                                    class_name = class_names[class_idx]
                                    f.write(f"{class_name} {targets[i, 2]} {targets[i, 3]} {targets[i, 4]} {targets[i, 5]}\n")
                    elif targets.dim() == 1:
                        if targets.numel() >= 6:
                            class_idx = int(targets[1])
                            if class_idx < len(class_names):
                                class_name = class_names[class_idx]
                                f.write(f"{class_name} {targets[2]} {targets[3]} {targets[4]} {targets[5]}\n")
            
            # Save prediction results to file
            dr_file = os.path.join(map_out_path, "detection-results", f"{image_id}.txt")
            with open(dr_file, "w") as f:
                if results[0] is not None:
                    for result in results[0]:
                        if len(result) > 5:
                            left, top, right, bottom, conf, class_idx = result
                            class_name = class_names[int(class_idx)]
                            f.write(f"{class_name} {conf} {left} {top} {right} {bottom}\n")
    
    # Calculate mAP and draw PR curve
    print("Calculate mAP and draw PR curve...")
    map_value = get_map(
        0.5,  # IoU threshold
        True,  # draw curve
        score_threhold=0.1,  # score threshold
        path=map_out_path  # path
    )
    
    print(f"mAP: {map_value:.4f}")
    print("PR curve has been generated in the map_out directory")
    
    # Clean temporary files
    import shutil
    if os.path.exists(map_out_path):
        shutil.rmtree(map_out_path)
    
    print("PR curve generation completed!")

if __name__ == "__main__":
    generate_pr_curve()
