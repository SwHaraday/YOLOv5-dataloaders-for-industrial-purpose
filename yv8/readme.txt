To Do
1.Rename original ..ultralytics\data\augment.py
2.Copy and paste THIS augment.py to ..ultralytics\data\
3.Rename original ..ultralytics\cfg\default.yaml
4.Copy and paste THIS default.yaml to ..ultralytics\cfg\
5.rename original ..ultralytics\engine\trainer.py
6.Copy and paste THIS trainer.py to ..ultralytics\engine\

In augment.py,
- def v8_transforms, class Mosiac and def _mosaic4 are modified.
- class keepGTB are added just after class Mosaic.

In default.yaml,
- kgtb: False are added as args.
- Other than HSV and Fliplr are stopped.

In trainer.py,
- line 'self.args.kgtb = False' inserted in def _close_dataloader_mosaic().

Usage:
yolo detect train data=hoge.yaml model=yolov8n.pt epochs=100 batch=16 device=0 kgtb=True

You can perform original training, unless 'kgtb=True' option is used.
Only 640*640 training w/kGTB is available at this moment.

**important notice**
YOLOv8 does not allow to use training image with no bboxes.
If you are UNLUCKY, top-left-bottom-right which contains at-least 1 bbox and keeps shape
can not be achieved within more than 10,000times trials. 
That case, (0,0)to(640,640) cropped image which is same as no-mosaic will be used
as training data. 

202406 SwHaraday