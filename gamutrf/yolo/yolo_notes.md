# Setup (AIR-T specific) 

## Create conda env  

`conda env create -f airstack-py36-yolo.yml`

- airstack-py36-yolo.yml 

```
name: airstack-py36-yolo
channels:
  - conda-forge
  - nvidia
  - defaults
  - file://opt/deepwave/conda-channels/airstack-conda

dependencies:
  - python=3.6
  - scipy
  - numpy
  - matplotlib
  - pip
  - soapysdr-module-airt
  - airstack-tensorrt
  - gnuradio
  - gr-wavelearner

  - pip:
    - https://archive.deepwavedigital.com/onnxruntime-gpu/onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
    - https://archive.deepwavedigital.com/pycuda/pycuda-2020.1-cp36-cp36m-linux_aarch64.whl
    - https://archive.deepwavedigital.com/torch/torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```
 
`conda activate airstack-py36-yolo`

## Install Yolov8

Clone repository 
`git clone https://github.com/ultralytics/ultralytics` 

Modify files
```
git diff setup.py
diff --git a/setup.py b/setup.py
index 129cebe..24317dd 100644
--- a/setup.py
+++ b/setup.py
@@ -21,7 +21,7 @@ def get_version():
 setup(
     name='ultralytics',  # name of pypi package
     version=get_version(),  # version of pypi package
-    python_requires='>=3.7',
+    python_requires='>=3.6',
     license='AGPL-3.0',
     description=('Ultralytics YOLOv8 for SOTA object detection, multi-object tracking, instance segmentation, '
                  'pose estimation and image classification.'),
```
```
git diff requirements.txt
diff --git a/requirements.txt b/requirements.txt
index 7a457dd..3b491ee 100644
--- a/requirements.txt
+++ b/requirements.txt
@@ -3,7 +3,7 @@

 # Base ----------------------------------------
 matplotlib>=3.2.2
-numpy>=1.22.2 # pinned by Snyk to avoid a vulnerability
+#numpy>=1.22.2 # pinned by Snyk to avoid a vulnerability^M
 opencv-python>=4.6.0
 pillow>=7.1.2
 pyyaml>=5.3.1
```
Install 
`pip install -e .`

## Install Onnx
`pip install onnx`


# Train 
`yolo detect train model=yolov8n.pt data=dataset.yaml epochs=500 project=results_yolo_combined_snr name=7_27_23`
- dataset.yaml
```
train: /home/ltindall/gamutRF/data/gamutrf/yolo_combined_snr/YOLODataset/images/train/
val: /home/ltindall/gamutRF/data/gamutrf/yolo_combined_snr/YOLODataset/images/val/
test: /home/ltindall/gamutRF/data/gamutrf/yolo_combined_snr/YOLODataset/images/test/
nc: 2
names: ['mini2_video', 'mini2_telem']
```

OR   

Define training parameters:
- /home/ltindall/RFClassification/params.yaml  
```
model_type: yolov8n.pt
pretrained: True
seed: 0
imgsz: 640
batch: 4
epochs: 100
optimizer: Adam # other choices=['SGD', 'Adam', 'AdamW', 'RMSProp']
lr0: 0.01  # learning rate
name: 'yolov8s_exp_v0' # experiment name
```

- /home/ltindall/RFClassification/data/roboflow/data.yaml    


`(airstack-py39) ltindall@icebreaker:~/RFClassification$ python yolov8_train.py`

<br>

# Yolov8 training output
## If project & name not defined then default output in 
`/home/ltindall/ultralytics/runs/`

<br>

# Convert .pt to .onnx
## Run
`(airstack-py36) ltindall@icebreaker:~$ yolo export model=/home/ltindall/yolov8s_weights_6_7.pt format=onnx device=0`   
## Creates
`/home/ltindall/yolov8s_weights_6_7.onnx`

<br>

# Convert .onnx to .engine/.plan
## Run
`(airt-py39) ltindall@icebreaker:~/gamutrf$ /usr/src/tensorrt/bin/trtexec --onnx=/home/ltindall/yolov8s_weights_6_7.onnx --saveEngine=/home/ltindall/yolov8s_weights_6_7.plan`   
## Creates
`/home/ltindall/yolov8s_weights_6_7.plan`

<br>

# BROKEN: convert .pt to .engine/.plan
> Note: Converting straight from PyTorch .pt to .engine/.plan has not been working. Reasons unknown. Instead convert .pt->.onnx->.plan 

`(airstack-py36) ltindall@icebreaker:~$ yolo export model=/home/ltindall/yolov8s_weights_6_7.pt format=engine device=0`   
--> /home/ltindall/yolov8s_weights_6_7.engine

<br>

# Run scanner with Yolov8
## Run
`(airt-py39) ltindall@icebreaker:~/gamutrf$ gamutrf-scan --sdr=SoapyAIRT --freq-start=5.6e9 --freq-end=5.8e9 --tune-step-fft 2048 --samp-rate=100e6 --nfft 256 --tuneoverlap=1 --inference_plan_file=/home/ltindall/yolov8s_weights_6_7.plan --inference_output_dir=inference_output`
