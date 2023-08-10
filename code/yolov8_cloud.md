```python
%pip install ultralytics
import ultralytics
ultralytics.checks()
```

    Ultralytics YOLOv8.0.92 🚀 Python-3.10.11 torch-2.0.0+cu118 CUDA:0 (Tesla T4, 15102MiB)
    Setup complete ✅ (2 CPUs, 12.7 GB RAM, 23.5/78.2 GB disk)



```python

!yolo export model=yolov8n.pt format=torchscript
```

    Ultralytics YOLOv8.0.92 🚀 Python-3.10.11 torch-2.0.0+cu118 CPU
    YOLOv8n summary (fused): 168 layers, 3151904 parameters, 0 gradients, 8.7 GFLOPs
    
    [34m[1mPyTorch:[0m starting from yolov8n.pt with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 84, 8400) (6.2 MB)
    
    [34m[1mTorchScript:[0m starting export with torch 2.0.0+cu118...
    [34m[1mTorchScript:[0m export success ✅ 3.3s, saved as yolov8n.torchscript (12.4 MB)
    
    Export complete (4.1s)
    Results saved to [1m/content[0m
    Predict:         yolo predict task=detect model=yolov8n.torchscript imgsz=640 
    Validate:        yolo val task=detect model=yolov8n.torchscript imgsz=640 data=coco.yaml 
    Visualize:       https://netron.app



```python
mkdir datasets
```

    mkdir: cannot create directory ‘datasets’: File exists



```python
!pip install roboflow

# %cd ../datasets/
from roboflow import Roboflow
rf = Roboflow(api_key="vJVZdwHbeGnEemGMJZ4K")
project = rf.workspace("cloudyclass").project("cloud_class")
dataset = project.version(1).download("folder")
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: roboflow in /usr/local/lib/python3.10/dist-packages (1.0.8)
    Requirement already satisfied: idna==2.10 in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.10)
    Requirement already satisfied: chardet==4.0.0 in /usr/local/lib/python3.10/dist-packages (from roboflow) (4.0.0)
    Requirement already satisfied: certifi==2022.12.7 in /usr/local/lib/python3.10/dist-packages (from roboflow) (2022.12.7)
    Requirement already satisfied: python-dotenv in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.0.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.27.1)
    Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.16.0)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from roboflow) (3.7.1)
    Requirement already satisfied: cycler==0.10.0 in /usr/local/lib/python3.10/dist-packages (from roboflow) (0.10.0)
    Requirement already satisfied: tqdm>=4.41.0 in /usr/local/lib/python3.10/dist-packages (from roboflow) (4.65.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.4.4)
    Requirement already satisfied: urllib3>=1.26.6 in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.26.15)
    Requirement already satisfied: Pillow>=7.1.2 in /usr/local/lib/python3.10/dist-packages (from roboflow) (8.4.0)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.8.2)
    Requirement already satisfied: wget in /usr/local/lib/python3.10/dist-packages (from roboflow) (3.2)
    Requirement already satisfied: pyparsing==2.4.7 in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.4.7)
    Requirement already satisfied: requests-toolbelt in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.0.0)
    Requirement already satisfied: numpy>=1.18.5 in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.22.4)
    Requirement already satisfied: opencv-python>=4.1.2 in /usr/local/lib/python3.10/dist-packages (from roboflow) (4.7.0.72)
    Requirement already satisfied: PyYAML>=5.3.1 in /usr/local/lib/python3.10/dist-packages (from roboflow) (6.0)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (1.0.7)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (4.39.3)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (23.1)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.10/dist-packages (from requests->roboflow) (2.0.12)
    loading Roboflow workspace...
    loading Roboflow project...
    Downloading Dataset Version Zip in cloud_class-1 to folder: 98% [209158144 / 212510276] bytes

    Extracting Dataset Version Zip to cloud_class-1 in folder:: 100%|██████████| 5603/5603 [00:02<00:00, 2692.10it/s]



```python
from ultralytics import YOLO
model = YOLO('yolov8n-cls.pt')  # load a pretrained YOLOv8n classification model
model.train(data='/content/cloud_class-1', epochs=60)  # train the model
# model('https://ultralytics.com/images/bus.jpg')  # predict on an image
```

    Downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt to yolov8n-cls.pt...
    100%|██████████| 5.28M/5.28M [00:00<00:00, 23.0MB/s]
    Ultralytics YOLOv8.0.92 🚀 Python-3.10.11 torch-2.0.0+cu118 CUDA:0 (Tesla T4, 15102MiB)
    [34m[1myolo/engine/trainer: [0mtask=classify, mode=train, model=yolov8n-cls.pt, data=/content/cloud_class-1, epochs=60, patience=50, batch=16, imgsz=224, save=True, save_period=-1, cache=False, device=None, workers=8, project=None, name=None, exist_ok=False, pretrained=False, optimizer=SGD, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=0, resume=False, amp=True, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, show=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, vid_stride=1, line_thickness=3, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, boxes=True, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, cfg=None, v5loader=False, tracker=botsort.yaml, save_dir=runs/classify/train
    Overriding model.yaml nc=1000 with nc=2
    
                       from  n    params  module                                       arguments                     
      0                  -1  1       464  ultralytics.nn.modules.Conv                  [3, 16, 3, 2]                 
      1                  -1  1      4672  ultralytics.nn.modules.Conv                  [16, 32, 3, 2]                
      2                  -1  1      7360  ultralytics.nn.modules.C2f                   [32, 32, 1, True]             
      3                  -1  1     18560  ultralytics.nn.modules.Conv                  [32, 64, 3, 2]                
      4                  -1  2     49664  ultralytics.nn.modules.C2f                   [64, 64, 2, True]             
      5                  -1  1     73984  ultralytics.nn.modules.Conv                  [64, 128, 3, 2]               
      6                  -1  2    197632  ultralytics.nn.modules.C2f                   [128, 128, 2, True]           
      7                  -1  1    295424  ultralytics.nn.modules.Conv                  [128, 256, 3, 2]              
      8                  -1  1    460288  ultralytics.nn.modules.C2f                   [256, 256, 1, True]           
      9                  -1  1    332802  ultralytics.nn.modules.Classify              [256, 2]                      
    YOLOv8n-cls summary: 99 layers, 1440850 parameters, 1440850 gradients, 3.4 GFLOPs
    Transferred 156/158 items from pretrained weights
    [34m[1mTensorBoard: [0mStart with 'tensorboard --logdir runs/classify/train', view at http://localhost:6006/
    [34m[1mAMP: [0mrunning Automatic Mixed Precision (AMP) checks with YOLOv8n...
    [34m[1mAMP: [0mchecks passed ✅
    [34m[1moptimizer:[0m SGD(lr=0.01) with parameter groups 26 weight(decay=0.0), 27 weight(decay=0.0005), 27 bias
    [34m[1malbumentations: [0mRandomResizedCrop(p=1.0, height=224, width=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1), HorizontalFlip(p=0.5), ColorJitter(p=0.5, brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=[0, 0]), Normalize(p=1.0, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0), ToTensorV2(always_apply=True, p=1.0, transpose_mask=False)
    Image sizes 224 train, 224 val
    Using 2 dataloader workers
    Logging results to [1mruns/classify/train[0m
    Starting training for 60 epochs...
    
          Epoch    GPU_mem       loss  Instances       Size
           1/60     0.524G     0.1694         16        224:   2%|▏         | 5/243 [00:02<00:56,  4.24it/s]Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
           1/60     0.524G      0.165         16        224:   3%|▎         | 7/243 [00:02<00:41,  5.69it/s]Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
           1/60     0.524G     0.1641         16        224:   5%|▍         | 11/243 [00:02<00:31,  7.32it/s]Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
           1/60     0.524G     0.1614         16        224:   5%|▌         | 13/243 [00:02<00:31,  7.37it/s]
    100%|██████████| 755k/755k [00:00<00:00, 67.4MB/s]
    
    100%|██████████| 755k/755k [00:00<00:00, 72.0MB/s]
           1/60     0.524G     0.1621         16        224:   6%|▌         | 14/243 [00:03<00:33,  6.83it/s]
    100%|██████████| 755k/755k [00:00<00:00, 27.5MB/s]
           1/60     0.528G     0.1083         14        224: 100%|██████████| 243/243 [00:31<00:00,  7.71it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:03<00:00,  5.00it/s]
                       all      0.959          1
    
          Epoch    GPU_mem       loss  Instances       Size
           2/60     0.371G    0.06803         14        224: 100%|██████████| 243/243 [00:29<00:00,  8.18it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.35it/s]
                       all      0.954          1
    
          Epoch    GPU_mem       loss  Instances       Size
           3/60     0.371G    0.06666         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.52it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.79it/s]
                       all      0.959          1
    
          Epoch    GPU_mem       loss  Instances       Size
           4/60     0.371G    0.06186         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.40it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.74it/s]
                       all      0.932          1
    
          Epoch    GPU_mem       loss  Instances       Size
           5/60     0.371G    0.06144         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.50it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.52it/s]
                       all      0.956          1
    
          Epoch    GPU_mem       loss  Instances       Size
           6/60     0.371G    0.05838         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.44it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  6.55it/s]
                       all      0.947          1
    
          Epoch    GPU_mem       loss  Instances       Size
           7/60     0.371G    0.05697         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.65it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:03<00:00,  4.60it/s]
                       all      0.954          1
    
          Epoch    GPU_mem       loss  Instances       Size
           8/60     0.371G    0.05772         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.39it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.75it/s]
                       all      0.952          1
    
          Epoch    GPU_mem       loss  Instances       Size
           9/60     0.371G     0.0526         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.42it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.65it/s]
                       all      0.956          1
    
          Epoch    GPU_mem       loss  Instances       Size
          10/60     0.371G    0.04821         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.46it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.48it/s]
                       all      0.956          1
    
          Epoch    GPU_mem       loss  Instances       Size
          11/60     0.371G    0.05199         14        224: 100%|██████████| 243/243 [00:29<00:00,  8.35it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.77it/s]
                       all      0.963          1
    
          Epoch    GPU_mem       loss  Instances       Size
          12/60     0.371G    0.05539         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.44it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:03<00:00,  4.93it/s]
                       all      0.959          1
    
          Epoch    GPU_mem       loss  Instances       Size
          13/60     0.371G    0.05079         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.56it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:03<00:00,  5.84it/s]
                       all      0.961          1
    
          Epoch    GPU_mem       loss  Instances       Size
          14/60     0.371G    0.05038         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.42it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.54it/s]
                       all      0.952          1
    
          Epoch    GPU_mem       loss  Instances       Size
          15/60     0.371G    0.05209         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.52it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.62it/s]
                       all      0.966          1
    
          Epoch    GPU_mem       loss  Instances       Size
          16/60     0.371G    0.04739         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.46it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  8.33it/s]
                       all      0.963          1
    
          Epoch    GPU_mem       loss  Instances       Size
          17/60     0.371G    0.04691         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.46it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.02it/s]
                       all      0.963          1
    
          Epoch    GPU_mem       loss  Instances       Size
          18/60     0.371G    0.04356         14        224: 100%|██████████| 243/243 [00:27<00:00,  8.88it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:03<00:00,  4.56it/s]
                       all      0.952          1
    
          Epoch    GPU_mem       loss  Instances       Size
          19/60     0.371G    0.04329         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.45it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.79it/s]
                       all      0.963          1
    
          Epoch    GPU_mem       loss  Instances       Size
          20/60     0.371G    0.04822         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.48it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.90it/s]
                       all      0.961          1
    
          Epoch    GPU_mem       loss  Instances       Size
          21/60     0.371G    0.04805         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.42it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.98it/s]
                       all      0.963          1
    
          Epoch    GPU_mem       loss  Instances       Size
          22/60     0.371G    0.05068         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.46it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  8.24it/s]
                       all      0.957          1
    
          Epoch    GPU_mem       loss  Instances       Size
          23/60     0.371G    0.04468         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.45it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  6.76it/s]
                       all      0.963          1
    
          Epoch    GPU_mem       loss  Instances       Size
          24/60     0.371G    0.04495         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.40it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:03<00:00,  4.99it/s]
                       all      0.972          1
    
          Epoch    GPU_mem       loss  Instances       Size
          25/60     0.371G    0.04238         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.45it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  8.03it/s]
                       all       0.97          1
    
          Epoch    GPU_mem       loss  Instances       Size
          26/60     0.371G    0.04532         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.47it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  8.23it/s]
                       all      0.961          1
    
          Epoch    GPU_mem       loss  Instances       Size
          27/60     0.371G    0.04306         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.46it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.91it/s]
                       all      0.972          1
    
          Epoch    GPU_mem       loss  Instances       Size
          28/60     0.371G    0.04182         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.50it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.41it/s]
                       all       0.97          1
    
          Epoch    GPU_mem       loss  Instances       Size
          29/60     0.371G    0.04043         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.42it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.61it/s]
                       all      0.972          1
    
          Epoch    GPU_mem       loss  Instances       Size
          30/60     0.371G    0.03976         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.64it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:03<00:00,  5.12it/s]
                       all      0.972          1
    
          Epoch    GPU_mem       loss  Instances       Size
          31/60     0.371G    0.03947         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.66it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  6.06it/s]
                       all       0.97          1
    
          Epoch    GPU_mem       loss  Instances       Size
          32/60     0.371G    0.03712         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.49it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  8.33it/s]
                       all      0.963          1
    
          Epoch    GPU_mem       loss  Instances       Size
          33/60     0.371G    0.03997         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.58it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  8.00it/s]
                       all      0.972          1
    
          Epoch    GPU_mem       loss  Instances       Size
          34/60     0.371G    0.03816         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.60it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  8.14it/s]
                       all      0.979          1
    
          Epoch    GPU_mem       loss  Instances       Size
          35/60     0.371G    0.03641         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.43it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.94it/s]
                       all      0.966          1
    
          Epoch    GPU_mem       loss  Instances       Size
          36/60     0.371G    0.04081         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.56it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.14it/s]
                       all      0.975          1
    
          Epoch    GPU_mem       loss  Instances       Size
          37/60     0.371G    0.03743         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.60it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:03<00:00,  4.62it/s]
                       all      0.966          1
    
          Epoch    GPU_mem       loss  Instances       Size
          38/60     0.371G    0.03877         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.58it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.14it/s]
                       all      0.975          1
    
          Epoch    GPU_mem       loss  Instances       Size
          39/60     0.371G    0.03553         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.58it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.83it/s]
                       all      0.973          1
    
          Epoch    GPU_mem       loss  Instances       Size
          40/60     0.371G    0.03637         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.59it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  8.00it/s]
                       all      0.973          1
    
          Epoch    GPU_mem       loss  Instances       Size
          41/60     0.371G    0.03523         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.48it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.65it/s]
                       all      0.975          1
    
          Epoch    GPU_mem       loss  Instances       Size
          42/60     0.371G    0.03341         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.46it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  8.17it/s]
                       all      0.973          1
    
          Epoch    GPU_mem       loss  Instances       Size
          43/60     0.371G    0.03448         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.50it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  6.57it/s]
                       all      0.972          1
    
          Epoch    GPU_mem       loss  Instances       Size
          44/60     0.371G     0.0307         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.66it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:03<00:00,  5.24it/s]
                       all      0.975          1
    
          Epoch    GPU_mem       loss  Instances       Size
          45/60     0.371G    0.03208         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.64it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.53it/s]
                       all      0.972          1
    
          Epoch    GPU_mem       loss  Instances       Size
          46/60     0.371G    0.03194         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.54it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  8.12it/s]
                       all       0.98          1
    
          Epoch    GPU_mem       loss  Instances       Size
          47/60     0.371G    0.02944         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.49it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.92it/s]
                       all      0.979          1
    
          Epoch    GPU_mem       loss  Instances       Size
          48/60     0.371G    0.02993         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.56it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.79it/s]
                       all      0.979          1
    
          Epoch    GPU_mem       loss  Instances       Size
          49/60     0.371G    0.02875         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.58it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  8.09it/s]
                       all      0.977          1
    
          Epoch    GPU_mem       loss  Instances       Size
          50/60     0.371G    0.02922         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.50it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:03<00:00,  4.99it/s]
                       all      0.984          1
    
          Epoch    GPU_mem       loss  Instances       Size
          51/60     0.371G    0.02827         14        224: 100%|██████████| 243/243 [00:27<00:00,  8.73it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:03<00:00,  5.52it/s]
                       all      0.979          1
    
          Epoch    GPU_mem       loss  Instances       Size
          52/60     0.371G    0.02876         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.47it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  8.00it/s]
                       all      0.982          1
    
          Epoch    GPU_mem       loss  Instances       Size
          53/60     0.371G    0.02829         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.42it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.88it/s]
                       all       0.98          1
    
          Epoch    GPU_mem       loss  Instances       Size
          54/60     0.371G    0.02759         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.40it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.51it/s]
                       all      0.984          1
    
          Epoch    GPU_mem       loss  Instances       Size
          55/60     0.371G    0.02501         14        224: 100%|██████████| 243/243 [00:29<00:00,  8.36it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  8.09it/s]
                       all      0.977          1
    
          Epoch    GPU_mem       loss  Instances       Size
          56/60     0.371G    0.02577         14        224: 100%|██████████| 243/243 [00:29<00:00,  8.36it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:03<00:00,  5.62it/s]
                       all      0.982          1
    
          Epoch    GPU_mem       loss  Instances       Size
          57/60     0.371G    0.02451         14        224: 100%|██████████| 243/243 [00:29<00:00,  8.38it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:03<00:00,  5.57it/s]
                       all      0.988          1
    
          Epoch    GPU_mem       loss  Instances       Size
          58/60     0.371G    0.02497         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.41it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.80it/s]
                       all      0.982          1
    
          Epoch    GPU_mem       loss  Instances       Size
          59/60     0.371G    0.02492         14        224: 100%|██████████| 243/243 [00:29<00:00,  8.32it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  7.85it/s]
                       all      0.984          1
    
          Epoch    GPU_mem       loss  Instances       Size
          60/60     0.371G    0.02526         14        224: 100%|██████████| 243/243 [00:28<00:00,  8.39it/s]
                   classes   top1_acc   top5_acc: 100%|██████████| 18/18 [00:02<00:00,  6.17it/s]
                       all      0.982          1
    
    60 epochs completed in 0.531 hours.
    Optimizer stripped from runs/classify/train/weights/last.pt, 3.0MB
    Optimizer stripped from runs/classify/train/weights/best.pt, 3.0MB
    Results saved to [1mruns/classify/train[0m



```python
from google.colab import drive
/home
/content/runs
drive.mount('/content/drive')
```

    Mounted at /content/drive


# New Section


```python
# Validate multiple models
for x in 'nsmlx':
  !yolo val model=yolov8n-cls.pt data="/content/cloud_class-1"
```

    Ultralytics YOLOv8.0.92 🚀 Python-3.10.11 torch-2.0.0+cu118 CUDA:0 (Tesla T4, 15102MiB)
    YOLOv8n-cls summary (fused): 73 layers, 2715880 parameters, 0 gradients, 4.3 GFLOPs
    Traceback (most recent call last):
      File "/usr/local/bin/yolo", line 8, in <module>
        sys.exit(entrypoint())
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/cfg/__init__.py", line 391, in entrypoint
        getattr(model, mode)(**overrides)  # default args from model
      File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/engine/model.py", line 301, in val
        validator(model=self.model)
      File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/engine/validator.py", line 135, in __call__
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/v8/classify/val.py", line 64, in get_dataloader
        dataset = self.build_dataset(dataset_path)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/v8/classify/val.py", line 59, in build_dataset
        dataset = ClassificationDataset(root=img_path, imgsz=self.args.imgsz, augment=False)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/data/dataset.py", line 214, in __init__
        super().__init__(root=root)
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 309, in __init__
        super().__init__(
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 145, in __init__
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 189, in make_dataset
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 61, in make_dataset
        directory = os.path.expanduser(directory)
      File "/usr/lib/python3.10/posixpath.py", line 232, in expanduser
        path = os.fspath(path)
    TypeError: expected str, bytes or os.PathLike object, not NoneType
    Sentry is attempting to send 2 pending events
    Waiting up to 2 seconds
    Press Ctrl-C to quit
    Ultralytics YOLOv8.0.92 🚀 Python-3.10.11 torch-2.0.0+cu118 CUDA:0 (Tesla T4, 15102MiB)
    YOLOv8n-cls summary (fused): 73 layers, 2715880 parameters, 0 gradients, 4.3 GFLOPs
    Traceback (most recent call last):
      File "/usr/local/bin/yolo", line 8, in <module>
        sys.exit(entrypoint())
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/cfg/__init__.py", line 391, in entrypoint
        getattr(model, mode)(**overrides)  # default args from model
      File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/engine/model.py", line 301, in val
        validator(model=self.model)
      File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/engine/validator.py", line 135, in __call__
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/v8/classify/val.py", line 64, in get_dataloader
        dataset = self.build_dataset(dataset_path)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/v8/classify/val.py", line 59, in build_dataset
        dataset = ClassificationDataset(root=img_path, imgsz=self.args.imgsz, augment=False)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/data/dataset.py", line 214, in __init__
        super().__init__(root=root)
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 309, in __init__
        super().__init__(
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 145, in __init__
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 189, in make_dataset
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 61, in make_dataset
        directory = os.path.expanduser(directory)
      File "/usr/lib/python3.10/posixpath.py", line 232, in expanduser
        path = os.fspath(path)
    TypeError: expected str, bytes or os.PathLike object, not NoneType
    Ultralytics YOLOv8.0.92 🚀 Python-3.10.11 torch-2.0.0+cu118 CUDA:0 (Tesla T4, 15102MiB)
    YOLOv8n-cls summary (fused): 73 layers, 2715880 parameters, 0 gradients, 4.3 GFLOPs
    Traceback (most recent call last):
      File "/usr/local/bin/yolo", line 8, in <module>
        sys.exit(entrypoint())
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/cfg/__init__.py", line 391, in entrypoint
        getattr(model, mode)(**overrides)  # default args from model
      File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/engine/model.py", line 301, in val
        validator(model=self.model)
      File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/engine/validator.py", line 135, in __call__
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/v8/classify/val.py", line 64, in get_dataloader
        dataset = self.build_dataset(dataset_path)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/v8/classify/val.py", line 59, in build_dataset
        dataset = ClassificationDataset(root=img_path, imgsz=self.args.imgsz, augment=False)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/data/dataset.py", line 214, in __init__
        super().__init__(root=root)
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 309, in __init__
        super().__init__(
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 145, in __init__
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 189, in make_dataset
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 61, in make_dataset
        directory = os.path.expanduser(directory)
      File "/usr/lib/python3.10/posixpath.py", line 232, in expanduser
        path = os.fspath(path)
    TypeError: expected str, bytes or os.PathLike object, not NoneType
    Ultralytics YOLOv8.0.92 🚀 Python-3.10.11 torch-2.0.0+cu118 CUDA:0 (Tesla T4, 15102MiB)
    YOLOv8n-cls summary (fused): 73 layers, 2715880 parameters, 0 gradients, 4.3 GFLOPs
    Traceback (most recent call last):
      File "/usr/local/bin/yolo", line 8, in <module>
        sys.exit(entrypoint())
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/cfg/__init__.py", line 391, in entrypoint
        getattr(model, mode)(**overrides)  # default args from model
      File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/engine/model.py", line 301, in val
        validator(model=self.model)
      File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/engine/validator.py", line 135, in __call__
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/v8/classify/val.py", line 64, in get_dataloader
        dataset = self.build_dataset(dataset_path)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/v8/classify/val.py", line 59, in build_dataset
        dataset = ClassificationDataset(root=img_path, imgsz=self.args.imgsz, augment=False)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/data/dataset.py", line 214, in __init__
        super().__init__(root=root)
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 309, in __init__
        super().__init__(
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 145, in __init__
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 189, in make_dataset
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 61, in make_dataset
        directory = os.path.expanduser(directory)
      File "/usr/lib/python3.10/posixpath.py", line 232, in expanduser
        path = os.fspath(path)
    TypeError: expected str, bytes or os.PathLike object, not NoneType
    Ultralytics YOLOv8.0.92 🚀 Python-3.10.11 torch-2.0.0+cu118 CUDA:0 (Tesla T4, 15102MiB)
    YOLOv8n-cls summary (fused): 73 layers, 2715880 parameters, 0 gradients, 4.3 GFLOPs
    Traceback (most recent call last):
      File "/usr/local/bin/yolo", line 8, in <module>
        sys.exit(entrypoint())
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/cfg/__init__.py", line 391, in entrypoint
        getattr(model, mode)(**overrides)  # default args from model
      File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/engine/model.py", line 301, in val
        validator(model=self.model)
      File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/engine/validator.py", line 135, in __call__
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/v8/classify/val.py", line 64, in get_dataloader
        dataset = self.build_dataset(dataset_path)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/v8/classify/val.py", line 59, in build_dataset
        dataset = ClassificationDataset(root=img_path, imgsz=self.args.imgsz, augment=False)
      File "/usr/local/lib/python3.10/dist-packages/ultralytics/yolo/data/dataset.py", line 214, in __init__
        super().__init__(root=root)
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 309, in __init__
        super().__init__(
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 145, in __init__
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 189, in make_dataset
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
      File "/usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py", line 61, in make_dataset
        directory = os.path.expanduser(directory)
      File "/usr/lib/python3.10/posixpath.py", line 232, in expanduser
        path = os.fspath(path)
    TypeError: expected str, bytes or os.PathLike object, not NoneType



```python
!git clone https://github.com/ultralytics/ultralytics -b updates
%pip install -qe ultralytics
!pytest ultralytics/tests
```

    Cloning into 'ultralytics'...
    fatal: Remote branch updates not found in upstream origin
    [31mERROR: ultralytics is not a valid editable requirement. It should either be a path to a local project or a VCS URL (beginning with bzr+http, bzr+https, bzr+ssh, bzr+sftp, bzr+ftp, bzr+lp, bzr+file, git+http, git+https, git+ssh, git+git, git+file, hg+file, hg+http, hg+https, hg+ssh, hg+static-http, svn+ssh, svn+http, svn+https, svn+svn, svn+file).[0m[31m
    [0m[1m============================= test session starts ==============================[0m
    platform linux -- Python 3.10.11, pytest-7.2.2, pluggy-1.0.0
    rootdir: /content
    plugins: anyio-3.6.2
    collected 0 items                                                              [0m
    
    [33m============================ [33mno tests ran[0m[33m in 0.00s[0m[33m =============================[0m
    [31mERROR: file or directory not found: ultralytics/tests
    [0m



```python

```
