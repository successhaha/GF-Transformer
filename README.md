# GF-Transformer
Cross-Scale Guided Fusion Transformer for Disaster Assessment Using Satellite Imagery

# Requirements
   Install the necessary package with:
  
  -python 3\
  -pytorch 1.1.0+ and torchvision 0.3.0+\
  -Nvidia apex <https://github.com/NVIDA/apex>\
  -opencv-python\
  -imgaug

# Dataset
Dataset: https://www.xview2.org

# Train
we use the segformer as the backbone(mit_b3.pth).
downloda pretrained model:<https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA>

stage one:

CUDA_VISIBLE_DEVICES=0,1 python train_segformer.py/
python predict_transforer_loc.py/
stage two:
CUDA_VISIBLE_DEVICES=0,1 python train_segformer_cls.py/

# Data Processing Techniques

Models trained on different crops sizes from (448, 448) for heavy encoder to (736, 736) for light encoder.
Augmentations used for training:
 - Flip (often)
 - Rotation (often)
 - Scale (often)
 - Color shifts (rare)
 - Clahe / Blur / Noise (rare)
 - Saturation / Brightness / Contrast (rare)
 - ElasticTransformation (rare)

Inference goes on full image size (1024, 1024) with 4 simple test-time augmentations (original, filp left-right, flip up-down, rotation to 180).
