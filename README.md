# GF-Former
Cross-Scale Guided Fusion Transformer for Disaster Assessment Using Satellite Imagery\
<https://ieeexplore.ieee.org/document/10197573>
# Requirements
   Install the necessary package with:
  
  -python 3\
  -pytorch 1.9.0 and torchvision 0.10.0\
  -apex <https://github.com/NVIDA/apex>\
  -opencv-python\
  -imgaug\
  -timm

# Dataset
Dataset: https://www.xview2.org

# Train
Model：we use the segformer as the backbone(mit_b3.pth).\
Pretrained model: <https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA>

The first stage:
CUDA_VISIBLE_DEVICES=0 python train_segformer_loc.py

The second stage:
CUDA_VISIBLE_DEVICES=0 python train_segformer_cls.py

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

Contact the author e-mail：xiaowwhit@gmail.com or 2878570391@qq.com
