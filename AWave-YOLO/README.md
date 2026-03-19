# AWave-YOLO: English Sign Language Teaching System based on Adaptive Damped Wave Propagation Equation

## Description
This repository contains the official implementation of the **AWave-YOLO** model. AWave-YOLO is a robust English sign language detection framework that incorporates an adaptive damped wave propagation equation to address the challenges of low recognition accuracy under complex lighting conditions and weak differentiation of similar hand gestures. It features a frequency-time decoupled vision semantic propagation framework and a Direction-Aware Wave Propagation Operator (DAWPO) module for physically interpretable global information interaction, achieving 99.65% mAP in sign language detection.

## Dataset Information
The model was trained and evaluated on a custom English sign language dataset constructed based on the *Concise American Sign Language Dictionary*. 

- **Samples**: The constructed dataset contains 1,658 effective samples in total (1,326 for training, 166 for validation, and 166 for testing).
- **Categories**: 26 English letter handshape categories (A-Z).
- **Reference**: R. A. Tennant and M. G. Brown, *The American sign language handshape dictionary*. Gallaudet University Press, 1998.

## Code Information
This repository is organized similarly to standard deep learning training templates. The core model architecture relies on the YOLO framework with our customized AWaveFormer backbone.
- `train.py` / `train_voc_full.py`: Main scripts for training the model.
- `predict.py`: Script for image, video, and FPS predictions.
- `eval.py`: Script used to evaluate the model's accuracy on the test set and calculate evaluation metrics.
- `prepare_data.py`: Used to prepare the VOC-style dataset formatting and generate training set path files.
- `tools/`: Contains auxiliary scripts for PR-curve generation, confusion matrix analysis, detecting and visualizations, and plotting.
- `nets/`: Contains core network implementations such as `yolo.py`, `waveformer_dct.py` and other backbone components.

## Method & Data Pre-Processing
- **Data Pre-processing**: There are no complex data pre-processing steps. All dataset images undergo a simple Letterbox transformation, adjusting the original resolution of 1920x1080 to a uniform 640x640 resolution to perfectly fit the model input format.
- **Model Framework**: Evaluated alongside ResNet50, YOLOv8s, ConvNeXt-T, etc., under identical training configurations. SGD optimizer is used with a multi-task joint loss function (comprising BCE loss, CIoU regression loss, and DFL).

## Evaluation Methods & Metrics
The model performance was verified through comprehensive evaluation including:
- **Evaluation Methods**: Validated via detailed **Ablation Studies** (verifying the contribution of the Semantic-aware Wave Modulation and Direction-Aware Wave Propagation modules) and **Comparative Experiments** with other advanced CNN/physics-based models (ResNet, ConvNeXt, vHeat, etc.).
- **Evaluation Metrics**: The primary evaluation metric is **mAP@0.5** (mean Average Precision). It represents the arithmetic mean of average precision (AP - the area under the Precision-Recall curve) for all 26 individual letter classes, measured at a fixed Intersection-over-Union (IoU) threshold of 0.5. Secondary metrics include model parameter size, FLOPs, and max GPU/CPU utilization to evaluate hardware adaptation and real-time efficiency.

## Instructions for Use

### 1. Prepare Dataset
Ensure your dataset XML and image files are located in the `VOCdevkit/VOC2007` directory, then run:
```bash
python prepare_data.py
```
This will generate the necessary txt training splits.

### 2. Train Model
To start training the AWave-YOLO model from scratch or from pre-trained weights, run:
```bash
python train.py
```
You can modify hyperparameters such as batch size, learning rate, and training epochs directly within the variables in `train.py`.

### 3. Evaluate Model
Once training is complete, ensure that the `model_path` variable in `yolo.py` points to your newly saved weights (stored in the logs folder). Then evaluate mAP by running:
```bash
python eval.py
```

### 4. Predict / Inference
To make predictions on images or calculate FPS, execute:
```bash
python predict.py
```
Follow the console prompts to input the path of the image you want to test. Auxiliary plot generation tools are available in the `tools/` folder.

## Dependencies
This code requires a Python running environment and common deep learning dependencies:
- Python >= 3.8
- PyTorch >= 2.0.0
- OpenCV (opencv-python)
- SciPy 
- CUDA >= 11.8 (for GPU acceleration)
- einops
- timm
- torch_dct

To install dependencies, please make sure you have created your environment and run:
```bash
pip install -r requirements.txt
```

*(Note: Ensure your environment matches the GPU capabilities for hardware-accelerated training.)*

## Acknowledgement
This project builds upon various open-source implementations of YOLO. We extend our gratitude to the contributors of those repositories.

## References / Citation

If you use this code or dataset in your research, please cite:

```
[Paper citation will be added upon publication]
```

The dataset was constructed with reference to the following dictionary. If you use the gesture standards from this work, please also cite:

> Tennant, R. A., & Brown, M. G. (1998). *The American Sign Language Handshape Dictionary*. Gallaudet University Press.  
> Available at: https://gupress.gallaudet.edu/bookpage/ASLHDpage.html

## Contributing

Contributions, issues, and feature requests are welcome. Please feel free to open a GitHub issue or submit a pull request. Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
MIT License.
