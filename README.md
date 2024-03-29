# YOLOv5-dataloaders-for-industrial-purpose
dataloaders.py for industrial purpose. 

When training YOLOv5 model with original dataloaders.py, you may notice images which have incomplete Ground-Truth-Boxes are used.
This may lead to unexpected over-detections, especially inspecting industrial products.
Here is modified dataloaders.py which keeps Ground-Truth-Boxes as you defined even with Mosaic augmentation.
