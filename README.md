# YOLOv5-dataloaders-for-industrial-purpose
## dataloaders.py for industrial purpose. 

### When training YOLOv5 model with original dataloaders.py, you may notice images which have incomplete Ground-Truth-Boxes are used.
![train_batch1](https://github.com/SwHaraday/YOLOv5-dataloaders-for-industrial-purpose/assets/49700734/ac146ee3-9ade-4160-9b7e-93ae53fb497a)
### This may lead to unexpected over-detections, especially inspecting industrial products.
### Here is modified dataloaders.py which keeps Ground-Truth-Boxes as you defined even with Mosaic augmentation.

### Make 'keep Ground-Truth-Box' function selectable from 'train.py' by adding --kgtb option.
- yv5_60 for YOLOv5 version 6.0
- yv5_70 for YOLOv5 version 7.0

## Now 'kgtb=True' option is available to use on YOLOv8
- yv8 for YOLOv8
