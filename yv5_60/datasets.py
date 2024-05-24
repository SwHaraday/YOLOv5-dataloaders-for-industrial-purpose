# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
class Load4TISCams,class LoadV4TISCams にデバイスロスト検出機能を追加　20230619 原田
train時にGround-Truth-Boxを維持する機能を追加　train.pyで--kgtbをしていすること　20240404 原田
"""

import glob
import hashlib
import json
import os, sys
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='', shuffle=False, kgtb=False):
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augmentation
                                      hyp=hyp,  # hyperparameters
                                      rect=rect,  # rectangular batches
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      kgtb=kgtb,   # keep Ground-Truth-Box
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // WORLD_SIZE, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn), dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path

        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)
        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        cv2.waitKey(0)
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]
        img0 = np.expand_dims(img0, axis=0) # CHW > BCHW 
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    # YOLOv5 local webcam dataloader, i.e. `python detect.py --source 0`
    def __init__(self, pipe='0', img_size=640, stride=32):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference 最新のstream_loader.pyから登用
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        s = f'webcam {self.count}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None, s

    def __len__(self):
        return 0


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference 最新のstream_loader.pyから登用
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
         
        if os.path.isfile(sources) and sources.endswith(".txt"): # ソースに指定されたファイルがテキストの時だけ
            with open(sources) as f: # 行を分解してリスト化する
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        print(sources)
        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if 'youtube.com/' in s or 'youtu.be/' in s:  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            LOGGER.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(1 / self.fps[i])  # wait time
        cap.release() # 基本に忠実にループを抜けたらリリースしておく

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == 27: #ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years

# ★自作の4カメラ読み込みクラス
# usbカメラを最大4つまで繋いで、4つの画像を★タイル状に★一つに纏めた上でYOLOV5のモデルに
# 送るために作成した。Yolov5のutilsディレクトリのdatasets.pyに埋め込んで
# 使用するように作っている。20220131　原田
class Load4Streams:
    # originalのLoadStreamsを弄ってスレッドを使って4つのカメラ画像を一つにしてmodelに送り込むためのクラス
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        global flag
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference 最新のstream_loader.pyから登用
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.flag = True # 複数開いたカメラスレッドを閉じるためのフラグ
        self.w = 640
        self.h = 480

        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip()) and x[0] != '#']
        else:
            sources = [sources]

        print(sources)
        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * 4, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        # カメラの立ち上がり方次第でエラーを起こすことあるので、予め赤色の画面をカメラの数だけ用意しておく
        for i, s in enumerate(sources):  # index, source
            self.imgs[i] = np.full((self.h, self.w, 3), (0, 0, 255), dtype=np.uint8)

        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s + cv2.CAP_DSHOW)
            #assert cap.isOpened(), f'{st}Failed to open {s}'
            w = self.w #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = self.h #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            if cap.isOpened():
                _, self.imgs[i] = cap.read()  # guarantee first frame
                self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=False)
                # threadsは、daemon=Trueで複数起動すると終了時にカメラを開放しなくなる。そのためdaemon=False（デフォ）とした。
                LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
                self.threads[i].start()
                #print('** ', self.threads) # debug print
            else:
                print(f'{st}Failed to open Cam {s}')
                self.imgs[i] = np.full((self.h, self.w, 3), (128, 128, 128), dtype=np.uint8)

        LOGGER.info('')  # newline
		
        self.rect = True #np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        #if not self.rect:
            #LOGGER.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f and self.flag: # flagもループの条件に加えている
            start_t = time.perf_counter()
            n += 1
            #_, self.imgs[i] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            end_t = time.perf_counter()
            print(str(i) + '　elapse time = {:.3f} Seconds'.format((end_t - start_t))) 
            time.sleep(1 / self.fps[i])  # wait time
        cap.release() # 無限ループから抜けたらカメラインスタンスを開放するのを忘れないこと！

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        #if not all(x.isAlive() for x in self.threads) or cv2.waitKey(1) == 27: #ord('q'):  # q to quit
        if cv2.waitKey(1) == 27: #ord('q'):  # q to quit 
            self.flag = False # 画像取込の無限ループを抜けるためフラグを書き換える
            cv2.destroyAllWindows()          
            raise StopIteration

        #h, w, _ = self.imgs[0].shape # 画像のサイズを取込んでおく
        # ここで4つの画像を合成する
        if len(self.sources) == 1:
            self.imgs[1] = np.full((self.h, self.w, 3), (128, 128, 128), dtype=np.uint8)
        self.concimg = cv2.hconcat([self.imgs[0], self.imgs[1]])
        if len(self.sources) > 2:
            if len(self.sources) == 3:
                self.imgs[3] = np.full((self.h, self.w, 3), (128, 128, 128), dtype=np.uint8)
            conc2 = cv2.hconcat([self.imgs[2], self.imgs[3]])
            self.concimg = cv2.vconcat([self.concimg, conc2])
            self.concimg = cv2.resize(self.concimg, (800, 600), interpolation = cv2.INTER_AREA)
        else:
            self.concimg = cv2.resize(self.concimg, (800, 300), interpolation = cv2.INTER_AREA)
        self.obi = np.full((20, 800, 3), (255, 255, 255), dtype=np.uint8)
        self.concimg = cv2.vconcat([self.concimg, self.obi])
        self.concimg = np.expand_dims(self.concimg, axis=0) # CHW > BCHW 

        # Letterbox
        img0 = self.concimg.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW 
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years
# ★からここまで…

# ★★　縦ならび　！
# usbカメラを最大4つまで繋いで、4つの画像を縦に並べて一つに纏めた上でYOLOV5のモデルに
# 送るために作成した。Yolov5のutilsディレクトリのdatasets.pyに埋め込んで
# 使用するように作っている。20221206　原田
class LoadV4Streams:
    # originalのLoadStreamsを弄ってスレッドを使って4つのカメラ画像を一つにしてmodelに送り込むためのクラス
    def __init__(self, sources='Vstreams.txt', img_size=640, stride=32, auto=True):
        global flag
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference 最新のstream_loader.pyから登用
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.flag = True # 複数開いたカメラスレッドを閉じるためのフラグ

        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip()) and x[0] != '#']
        else:
            sources = [sources]

        print(sources)
        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * 4, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        self.w = 640 #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = 160 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        full_h = 480 # クロップしない場合の縦画素数
        self.start_h = int((full_h - self.h) / 2)
        # 予め赤色の画面をカメラの数だけ用意しておく
        for i, s in enumerate(sources):  # index, source
            self.imgs[i] = np.full((self.h, self.w, 3), (0, 0, 255), dtype=np.uint8)

        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s + cv2.CAP_DSHOW)
            #assert cap.isOpened(), f'{st}Failed to open {s}'

            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            if cap.isOpened():
                _, im = cap.read()  # guarantee first frame
                self.imgs[i] = im[self.start_h:(self.start_h + self.h), 0:self.w] # crop
                self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=False)
                # threadsは、daemon=Trueで複数起動すると終了時にカメラを開放しなくなる。そのためdaemon=False（デフォ）とした。
                LOGGER.info(f"{st} Success ({self.frames[i]} frames {self.w}x{self.h} at {self.fps[i]:.2f} FPS)")
                self.threads[i].start()
                #print('** ', self.threads) # debug print
            else:
                print(f'{st}Failed to open Cam {s}')
                self.imgs[i] = np.full((self.h, self.w, 3), (128, 128, 128), dtype=np.uint8)

        LOGGER.info('')  # newline
        
        self.rect = True #np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        #if not self.rect:
            #LOGGER.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f and self.flag: # flagもループの条件に加えている
            n += 1
            #_, self.imgs[i] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:

                    self.imgs[i] = im[self.start_h:(self.start_h + self.h), 0:self.w] # 取り込んだ画像の高さ方向で中心部分だけを使う
                else:
                    LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(1 / self.fps[i])  # wait time
        cap.release() # 無限ループから抜けたらカメラインスタンスを開放するのを忘れないこと！

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        #if not all(x.isAlive() for x in self.threads) or cv2.waitKey(1) == 27: #ord('q'):  # q to quit
        if cv2.waitKey(1) == 27: #ord('q'):  # q to quit 
            self.flag = False # 画像取込の無限ループを抜けるためフラグを書き換える
            cv2.destroyAllWindows()          
            raise StopIteration

        h, w, _ = self.imgs[0].shape # 画像のサイズを取込んでおく
        # ここで4つの画像を合成する
        if len(self.sources) == 1:
            self.imgs[1] = np.full((self.h, self.w, 3), (128, 128, 128), dtype=np.uint8)
        self.concimg = cv2.vconcat([self.imgs[0], self.imgs[1]])
        if len(self.sources) > 2:
            if len(self.sources) == 3:
                self.imgs[3] = np.full((self.h, self.w, 3), (128, 128, 128), dtype=np.uint8)
            conc2 = cv2.vconcat([self.imgs[2], self.imgs[3]])
            self.concimg = cv2.vconcat([self.concimg, conc2])
            self.concimg = cv2.resize(self.concimg, (self.w, 4*self.h), interpolation = cv2.INTER_AREA)
        else:
            self.concimg = cv2.resize(self.concimg, (self.w, 2*self.h), interpolation = cv2.INTER_AREA)
        self.obi = np.full((20, self.w, 3), (255, 255, 255), dtype=np.uint8)
        self.concimg = cv2.vconcat([self.concimg, self.obi])
        self.concimg = np.expand_dims(self.concimg, axis=0) # CHW > BCHW 

        # Letterbox
        img0 = self.concimg.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW 
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years
# ★からここまで…

# ★★自作のTIS産業用カメラ用のLoad4streamsバージョン
# 20220413　原田
# 20220617 運用から20日経過後、カメラからの入力画像が止まったため取り込みの確認と
# 取込めなかった場合前と同じ画像でなくブルーバックとして異常が分かりやすいようにした。 原田
class Load4TISCams:
    # originalのLoadStreamsを弄ってスレッドを使ってTISのカメラから画像を取込むためのクラス
    def __init__(self, sources='4TISCams.txt', img_size=640, stride=32, auto=True):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference 最新のstream_loader.pyから登用
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.flag = True
        self.rbt_flag = False # デバイスロストなどで自動的に自分を止める（再起動要否の目印）フラグ
        self.bubun = 40 # 前と新しい画像の比較に使う四角形部分の一辺のピクセル数 ★必ず偶数にすること！！！
        self.bad_cam = "" # デバイスロストしたカメラの位置情報を渡す変数

        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip()) and x[0] != '#']
        else:
            sources = [sources]

        print(sources)
        n = len(sources)
        try:
            # TISカメラのためにimportする
            import ctypes
            import tisgrabber as tis
        except:
            print('tisgrabber is not installed. Please check !')
            sys.exit(0)
        self.imgs, self.frames, self.threads = [None] * 4, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto

        self.cnt = 0 # maenoとnowの同一画像検出の回数カウンタ
        self.maeno = [None] * 4 # 比較用画像を保存する変数
        self.now = [None] * 4
        for i in range(4): # 初めに画像比較用の前の画像に当たるものを用意しておく
            self.maeno[i] = np.full((self.bubun, self.bubun, 3), (0, 255, 0), dtype=np.uint8)        

        self.fps = 70
        self.w = 640
        self.h = 480 # temporary definition
        vformat = "RGB24 ({0}x{1})".format(self.w, self.h) # カメラのビデオフォーマットを指定する定数
        
        ic = ctypes.cdll.LoadLibrary("./tisgrabber_x64.dll") # TISおまじない1
        tis.declareFunctions(ic) # TISおまじない2
        ic.IC_InitLibrary(0) # TISおまじない3
        hGrabber = [None] * 4 # カメラインスタンスを格納するリストを定義しておく
        # カメラの立上り順によるエラーを回避するために予め赤色の画面をカメラの数だけ用意しておく
        for i in range(4):  # index, source
            self.imgs[i] = np.full((self.h, self.w, 3), (0, 0, 255), dtype=np.uint8)

        for i, s in enumerate(sources):  # index, source
            self.imgs[i] = np.full((self.h, self.w, 3), (128, 128, 128), dtype=np.uint8) # ダミーとして最初に灰色画面を用意
            self.frames[i] = float('inf')  # infinite stream fallback
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            s = str(s)
            hGrabber[i] = ic.IC_CreateGrabber()
            ic.IC_OpenDevByUniqueName(hGrabber[i], tis.T(s)) # シリアルナンバーの指定も可能
            ic.IC_SetVideoFormat(hGrabber[i], tis.T(vformat))
            if (ic.IC_IsDevValid(hGrabber[i])): # カメラが開けたら
                # カメラの露光時間、FPS、ホワイトバランス、ゲインなどを設定する 
                # fps: - 549 と Exposure ：0.000001 - 30.0              
                ic.IC_SetFrameRate(hGrabber[i], ctypes.c_float(self.fps))
                ic.IC_SetPropertySwitch(hGrabber[i], tis.T("Exposure"), tis.T("Auto"), 0)
                ic.IC_SetPropertyAbsoluteValue(hGrabber[i], tis.T("Exposure"), tis.T("Value"), ctypes.c_float(0.004))
                #Brightness : 0 - 4095 Default 240
                ic.IC_SetPropertyValue(hGrabber[i], tis.T("Brightness"), tis.T("Value"),ctypes.c_int(240))
                #Gain :0.0 - 48.0 Default 1.0
                ic.IC_SetPropertySwitch(hGrabber[i], tis.T("Gain"), tis.T("Auto"), 0)
                ic.IC_SetPropertyAbsoluteValue(hGrabber[i], tis.T("Gain"), tis.T("Value"), ctypes.c_float(10.0))
                #WhiteBalance ： 各色 0.0 - 3.984375 ※IC Captureなどで実写を見て調整
                ic.IC_SetPropertySwitch(hGrabber[i], tis.T("WhiteBalance"), tis.T("Auto"), 0)
                ic.IC_SetPropertyAbsoluteValue(hGrabber[i], tis.T("WhiteBalance"), tis.T("White Balance Red"), ctypes.c_float(1.66))
                ic.IC_SetPropertyAbsoluteValue(hGrabber[i], tis.T("WhiteBalance"), tis.T("White Balance Green"), ctypes.c_float(1.00))              
                ic.IC_SetPropertyAbsoluteValue(hGrabber[i], tis.T("WhiteBalance"), tis.T("White Balance Blue"), ctypes.c_float(2.48))
                # ここまででカメラパラメータ設定は終了
                
                # Start the live video stream, but show no own live video window. We will use OpenCV for this.
                ic.IC_StartLive(hGrabber[i], 0) # 引数を「１」にするとライブ画像が開く。OpenCVでの描画をするので「０」とする。
                #print('★★ic.IC_SnapImage(hGrabber[',i, ']: ', ic.IC_SnapImage(hGrabber[i])) #debugprint

                # 連続取り込みのスレッドを起動する
                self.threads[i] = Thread(target=self.update, args=([i, hGrabber[i], s, ic, ctypes, tis]), daemon=False)
                LOGGER.info(f"{st} Success ({self.frames[i]} frames {self.w}x{self.h} at {self.fps:.2f} FPS)")
                self.threads[i].start()
                    #print('** ', self.threads) # debug print
                #else: # カメラは開いているのに画像が取得できない時
                #    print(f'{st}Failed to get images from opened Cam {s}')
                #    self.imgs[i] = np.full((self.h, self.w, 3), (128, 128, 128), dtype=np.uint8)
            else: # カメラが開けない時
                print(f'{st}Failed to open Cam {s}')
                
        self.rect = True  # dummy code. rect inference if all shapes equal


    def update(self, i, hGrabber, stream, ic, ctypes, tis):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        Width = ctypes.c_long()
        Height = ctypes.c_long()
        BitsPerPixel = ctypes.c_int()
        colorformat = ctypes.c_int()
        while (ic.IC_IsDevValid(hGrabber)) and n < f and self.flag:
            # かなり長い記述になるが以下self.imgs[i] = im までで画像をOpenCVに渡せる形で取得している
            if ic.IC_SnapImage(hGrabber) == tis.IC_SUCCESS:
                # Query values of image description
                ic.IC_GetImageDescription(hGrabber, Width, Height, BitsPerPixel, colorformat)
                # Calculate the buffer size
                bpp = int(BitsPerPixel.value / 8.0)
                buffer_size = Width.value * Height.value * BitsPerPixel.value
                n += 1
                imagePtr = ic.IC_GetImagePtr(hGrabber)
                imagedata = ctypes.cast(imagePtr, ctypes.POINTER(ctypes.c_ubyte * buffer_size))
                # Create the numpy array
                im = np.ndarray(buffer=imagedata.contents, dtype=np.uint8, shape=(Height.value, Width.value, bpp))
                im = cv2.flip(im, 0)
                self.imgs[i] = im					
                time.sleep(1 / self.fps)  # wait time
            
            else: # 画像が上手く取り込めなかったときの処理。メッセージを出してブルーバックにする。
                LOGGER.warning('WARNING: 画像が正常に取込めていません。　確認の上、プログラムを再起動して下さい。')
                self.imgs[i] =  np.full((Height.value, Width.value, 3), (255, 0, 0), dtype=np.uint8)
                #cap.open(stream)  # re-open stream if signal was lost         

        # 何らかの理由でループを抜けてしまった場合もブルーバック画像とする。ここに来るのはEscで意識的に止めた時とic.IC_IsDevValid(hGrabber)がFalseの時。
        print('画像取込のループを抜けました。 Cam:', i)
        self.imgs[i] =  np.full((Height.value, Width.value, 3), (255, 0, 0), dtype=np.uint8)
        ic.IC_StopLive(hGrabber)
        ic.IC_ReleaseGrabber(hGrabber)        

    def __iter__(self):
        return self

    def __next__(self):
        #if not all(x.isAlive() for x in self.threads) or cv2.waitKey(1) == 27: #ord('q'):  # q to quit
        if cv2.waitKey(1) == 27 or self.rbt_flag:#ord('q'):  # q to quit 
            self.flag = False
            cv2.destroyAllWindows()
            raise StopIteration

        self.now[0] = self.imgs[0][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)]
        self.now[1] = self.imgs[1][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)]   
        self.now[2] = self.imgs[2][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)]
        self.now[3] = self.imgs[3][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)] 
        #if (self.now[0] == self.maeno[0]).all() or (self.now[1] == self.maeno[1]).all() or (self.now[2] == self.maeno[2]).all() or (self.now[3] == self.maeno[3]).all():
        if (self.now[3] == self.maeno[3]).all(): # debug用　カメラが4台無い時に使う
            self.cnt +=1
            if self.cnt >= self.fps * 3 : # 画像が更新されないという判断が数秒続いたら…
                self.flag = False
                if (self.now[0] == self.maeno[0]).all():
                    self.bad_cam = "左上"
                elif (self.now[1] == self.maeno[1]).all():
                    self.bad_cam = "右上"                            
                elif (self.now[2] == self.maeno[2]).all():
                    self.bad_cam = "右下"
                elif (self.now[3] == self.maeno[3]).all():
                    self.bad_cam = "左下"
                self.rbt_flag = True # 終了後、自分を再起動するフラグを立てる
        else:
            self.cnt = 0 # 比較結果が異なればカウンタをリセット

        # ここで4つの画像を合成する
        self.concimg = cv2.hconcat([self.imgs[0], self.imgs[1]])
        conc2 = cv2.hconcat([self.imgs[3], self.imgs[2]])
        self.concimg = cv2.vconcat([self.concimg, conc2])
        self.concimg = cv2.resize(self.concimg, (800, 600), interpolation = cv2.INTER_AREA)
        self.obi = np.full((20, 800, 3), (255, 255, 255), dtype=np.uint8)
        self.concimg = cv2.vconcat([self.concimg, self.obi])
        self.concimg = np.expand_dims(self.concimg, axis=0) # CHW > BCHW 

        self.maeno[0] = self.now[0] # 比較用画像の入れ替え
        self.maeno[1] = self.now[1] # 比較用画像の入れ替え
        self.maeno[2] = self.now[2] # 比較用画像の入れ替え
        self.maeno[3] = self.now[3] # 比較用画像の入れ替え

        # Letterbox
        img0 = self.concimg.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW 
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, self.rbt_flag, self.bad_cam

    #def __len__(self):
    #    return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years
# ★★からここまで…
#


#
# ★★自作のTIS産業用カメラ用のLoad4TISCamsをヨコに細長い画像を縦に組合せるバージョン
# 20221206　撚線の山越え検出用に着手　原田
#
class LoadV4TISCams:
    # originalのLoadStreamsを弄ってスレッドを使ってTISのカメラから画像を取込むためのクラス
    def __init__(self, sources='V4TISCams.txt', img_size=640, stride=32, auto=True):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference 最新のstream_loader.pyから登用
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.flag = True
        self.rbt_flag = False # デバイスロストなどで自動的に自分を止める（再起動要否の目印）フラグ
        self.bubun = 40 # 前と新しい画像の比較に使う四角形部分の一辺のピクセル数 ★必ず偶数にすること！！！
        self.bad_cam = "" # デバイスロストしたカメラの位置情報を渡す変数

        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip()) and x[0] != '#']
        else:
            sources = [sources]

        print(sources)
        n = len(sources)
        try:
            # TISカメラのためにimportする
            import ctypes
            import tisgrabber as tis
        except:
            print('tisgrabber is not installed. Please check !')
            sys.exit(0)
        self.imgs, self.fps, self.frames, self.threads = [None] * 4, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto

        self.cnt = 0 # maenoとnowの同一画像検出の回数カウンタ
        self.maeno = [None] * 4 # 比較用画像を保存する変数
        self.now = [None] * 4
        for i in range(4): # 初めに画像比較用の前の画像に当たるものを用意しておく
            self.maeno[i] = np.full((self.bubun, self.bubun, 3), (0, 255, 0), dtype=np.uint8)  

        self.fps = 70
        self.w = 720 #640
        self.h = 180 #160 # temporary definition
        vformat = "RGB24 ({0}x{1})".format(self.w, self.h) # カメラのビデオフォーマットを指定する定数　WDR機能を使うのでRGB64とした。
        
        ic = ctypes.cdll.LoadLibrary("./tisgrabber_x64.dll") # TISおまじない1
        tis.declareFunctions(ic) # TISおまじない2
        ic.IC_InitLibrary(0) # TISおまじない3
        hGrabber = [None] * 4 # カメラインスタンスを格納するリストを定義しておく
        # カメラの立上り順によるエラーを回避するために予め赤色の画面をカメラの数だけ用意しておく
        for i in range(4):  # index, source
            self.imgs[i] = np.full((self.h, self.w, 3), (0, 0, 255), dtype=np.uint8)

        for i, s in enumerate(sources):  # index, source
            self.frames[i] = float('inf')  # infinite stream fallback
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            s = str(s)
            hGrabber[i] = ic.IC_CreateGrabber()
            ic.IC_OpenDevByUniqueName(hGrabber[i], tis.T(s)) # シリアルナンバーの指定も可能
            ic.IC_SetVideoFormat(hGrabber[i], tis.T(vformat))
            if (ic.IC_IsDevValid(hGrabber[i])): # カメラが開けたら
                #ic.IC_printItemandElementNames(hGrabber[i])
                # カメラの露光時間、FPS、ホワイトバランス、ゲインなどを設定する 

                # WDR（ダイナミックレンジを広げて明るくする）をセットしてみる　※撚線機の画質改善のため
                ic.IC_SetPropertySwitch(hGrabber[i], tis.T("Tone Mapping"), tis.T("Enable"), 1)
                ic.IC_SetPropertySwitch(hGrabber[i], tis.T("Tone Mapping"), tis.T("Auto"), 0)
                ic.IC_SetPropertyAbsoluteValue(hGrabber[i], tis.T("Tone Mapping"), tis.T("Intensity"), ctypes.c_float(0.5))
                ic.IC_SetPropertyAbsoluteValue(hGrabber[i], tis.T("Tone Mapping"), tis.T("Global Brightness Factor"), ctypes.c_float(0.0))
                #ic.IC_SetPropertySwitch(hGrabber[i], tis.T("Tone Mapping"), tis.T("Enable"), 0)

                #Gamma: 0.1-5.0 default 1.0
                ic.IC_SetPropertyAbsoluteValue(hGrabber[i], tis.T("Gamma"), tis.T("Value"), ctypes.c_float(0.7))

                # fps: - 549 と Exposure ：0.000001 - 30.0              
                ic.IC_SetFrameRate(hGrabber[i], ctypes.c_float(self.fps))
                ic.IC_SetPropertySwitch(hGrabber[i], tis.T("Exposure"), tis.T("Auto"), 0)
                ic.IC_SetPropertyAbsoluteValue(hGrabber[i], tis.T("Exposure"), tis.T("Value"), ctypes.c_float(0.004))
                #Brightness : 0 - 4095 Default 240
                ic.IC_SetPropertyValue(hGrabber[i], tis.T("Brightness"), tis.T("Value"),ctypes.c_int(240))
                #Gain :0.0 - 48.0 Default 1.0
                ic.IC_SetPropertySwitch(hGrabber[i], tis.T("Gain"), tis.T("Auto"), 0)
                ic.IC_SetPropertyAbsoluteValue(hGrabber[i], tis.T("Gain"), tis.T("Value"), ctypes.c_float(25.0))
                #WhiteBalance ： 各色 0.0 - 3.984375 ※IC Captureなどで実写を見て調整
                ic.IC_SetPropertySwitch(hGrabber[i], tis.T("WhiteBalance"), tis.T("Auto"), 0)
                ic.IC_SetPropertyAbsoluteValue(hGrabber[i], tis.T("WhiteBalance"), tis.T("White Balance Red"), ctypes.c_float(1.66))
                ic.IC_SetPropertyAbsoluteValue(hGrabber[i], tis.T("WhiteBalance"), tis.T("White Balance Green"), ctypes.c_float(1.00))              
                ic.IC_SetPropertyAbsoluteValue(hGrabber[i], tis.T("WhiteBalance"), tis.T("White Balance Blue"), ctypes.c_float(2.48))
                # ここまででカメラパラメータ設定は終了
                
                # Start the live video stream, but show no own live video window. We will use OpenCV for this.
                ic.IC_StartLive(hGrabber[i], 0) # 引数を「１」にするとライブ画像が開く。OpenCVでの描画をするので「０」とする。
                #print('★★ic.IC_SnapImage(hGrabber[',i, ']: ', ic.IC_SnapImage(hGrabber[i])) #debugprint

                # 連続取り込みのスレッドを起動する
                self.threads[i] = Thread(target=self.update, args=([i, hGrabber[i], s, ic, ctypes, tis]), daemon=False)
                LOGGER.info(f"{st} Success ({self.frames[i]} frames {self.w}x{self.h} at {self.fps:.2f} FPS)")
                self.threads[i].start()
                    #print('** ', self.threads) # debug print
                #else: # カメラは開いているのに画像が取得できない時
                #    print(f'{st}Failed to get images from opened Cam {s}')
                #    self.imgs[i] = np.full((self.h, self.w, 3), (128, 128, 128), dtype=np.uint8)
            else: # カメラが開けない時
                print(f'{st}Failed to open Cam {s}')
                
        self.rect = True  # dummy code. rect inference if all shapes equal


    def update(self, i, hGrabber, stream, ic, ctypes, tis):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        Width = ctypes.c_long()
        Height = ctypes.c_long()
        BitsPerPixel = ctypes.c_int()
        colorformat = ctypes.c_int()
        while (ic.IC_IsDevValid(hGrabber)) and n < f and self.flag:
            # かなり長い記述になるが以下self.imgs[i] = im までで画像をOpenCVに渡せる形で取得している
            if ic.IC_SnapImage(hGrabber) == tis.IC_SUCCESS:
                # Query values of image description
                ic.IC_GetImageDescription(hGrabber, Width, Height, BitsPerPixel, colorformat)
                # Calculate the buffer size
                bpp = int(BitsPerPixel.value / 8.0)
                buffer_size = Width.value * Height.value * BitsPerPixel.value
                n += 1
                imagePtr = ic.IC_GetImagePtr(hGrabber)
                imagedata = ctypes.cast(imagePtr, ctypes.POINTER(ctypes.c_ubyte * buffer_size))
                # Create the numpy array
                im = np.ndarray(buffer=imagedata.contents, dtype=np.uint8, shape=(Height.value, Width.value, bpp))
                im = cv2.flip(im, 0)
                self.imgs[i] = im					
                #time.sleep(1 / self.fps)  # wait time
            
            else: # 画像が上手く取り込めなかったときの処理。メッセージを出してブルーバックにする。
                LOGGER.warning('WARNING: 画像が正常に取込めていません。　確認の上、プログラムを再起動して下さい。')
                self.imgs[i] =  np.full((Height.value, Width.value, 3), (255, 0, 0), dtype=np.uint8)
                #cap.open(stream)  # re-open stream if signal was lost         

        # 何らかの理由でループを抜けてしまった場合もブルーバック画像とする。ここに来るのはEscで意識的に止めた時とic.IC_IsDevValid(hGrabber)がFalseの時。
        print('画像取込のループを抜けました。 Cam:', i)
        self.imgs[i] =  np.full((Height.value, Width.value, 3), (255, 0, 0), dtype=np.uint8)
        ic.IC_StopLive(hGrabber)
        ic.IC_SetPropertySwitch(hGrabber, tis.T("Tone Mapping"), tis.T("Enable"), 0)
        ic.IC_ReleaseGrabber(hGrabber)        

    def __iter__(self):
        return self

    def __next__(self):
        #if not all(x.isAlive() for x in self.threads) or cv2.waitKey(1) == 27: #ord('q'):  # q to quit
        if cv2.waitKey(1) == 27 or self.rbt_flag:#ord('q'):  # q to quit 
            self.flag = False
            cv2.destroyAllWindows()
            raise StopIteration

        # 比較用画像の切り出し
        self.now[0] = self.imgs[0][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)]
        self.now[1] = self.imgs[1][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)]   
        self.now[2] = self.imgs[2][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)]
        self.now[3] = self.imgs[3][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)] 
        #if (self.now[0] == self.maeno[0]).all() or (self.now[1] == self.maeno[1]).all() or (self.now[2] == self.maeno[2]).all() or (self.now[3] == self.maeno[3]).all():
        if (self.now[3] == self.maeno[3]).all(): # debug用　カメラが4台無い時に使う
            self.cnt +=1
            if self.cnt >= self.fps * 3 : # 画像が更新されないという判断が数秒続いたら…
                self.flag = False
                if (self.now[0] == self.maeno[0]).all():
                    self.bad_cam = "一番上"
                elif (self.now[1] == self.maeno[1]).all():
                    self.bad_cam = "二番目"                            
                elif (self.now[2] == self.maeno[2]).all():
                    self.bad_cam = "三番目"
                elif (self.now[3] == self.maeno[3]).all():
                    self.bad_cam = "一番下"
                self.rbt_flag = True # 終了後、自分を再起動するフラグを立てる
        else:
            self.cnt = 0 # 比較結果が異なればカウンタをリセット
            
        # ここで4つの画像を合成する
        self.concimg = cv2.vconcat([self.imgs[0], self.imgs[1]])
        conc2 = cv2.vconcat([self.imgs[2], self.imgs[3]])
        self.concimg = cv2.vconcat([self.concimg, conc2])
        self.concimg = cv2.resize(self.concimg, (self.w, 4*self.h), interpolation = cv2.INTER_AREA)
        self.obi = np.full((20, self.w, 3), (255, 255, 255), dtype=np.uint8)
        self.concimg = cv2.vconcat([self.concimg, self.obi])
        self.concimg = np.expand_dims(self.concimg, axis=0) # CHW > BCHW 

        self.maeno[0] = self.now[0] # 比較用画像の入れ替え
        self.maeno[1] = self.now[1] # 比較用画像の入れ替え
        self.maeno[2] = self.now[2] # 比較用画像の入れ替え
        self.maeno[3] = self.now[3] # 比較用画像の入れ替え

        # Letterbox
        img0 = self.concimg.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW 
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, self.rbt_flag, self.bad_cam

    #def __len__(self):
    #    return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years
# ★★からここまで…
#
#
#

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', kgtb=False):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None
        self.kgtb = kgtb # keep Ground-Truth-Box

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # same version
            assert cache['hash'] == get_hash(self.label_files + self.img_files)  # same hash
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    im = self.imgs[i]
    if im is None:  # not cached in ram
        npy = self.img_npy[i]
        if npy and npy.exists():  # load npy
            im = np.load(npy)
        else:  # read image
            path = self.img_files[i]
            im = cv2.imread(path)  # BGR
            assert im is not None, f'Image Not Found {path}'
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    else:
        return self.imgs[i], self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized

# ★★ グランドトゥルースボックスが分断されることで意図しない形状の山越えを過検出している可能性があった。
# ★★　そのためグランドトゥルースボックスが分断されない位置でモザイク済みの□1280を□640でクロップする関数を作成した。
def random_crop_keepGTB(self, im, labels):
    # 画像の大きさを取っておく
    h0, w0, c = im.shape
    n = len(labels)
    # w方向のクロップ範囲を決める 
    flag = [True] * n        
    while any(flag):
        flag = [True] * n
        i = 0
        left = random.randint(0, self.img_size - 1)
        right = left + (w0 // 2)
        for label in labels:
            if label[1] <= left <= label[3] or label[1] <= right <= label[3]:
                flag[i] = True
            else:
                flag[i] = False
            i += 1
    # h方向のクロップ範囲を決める
    flag = [True] * n
    while any(flag):
        flag = [True] * n
        i = 0
        top = random.randint(0, self.img_size - 1)
        bottom = top + (w0 // 2)
        for label in labels:
            if label[2] <= top <= label[4] or label[2] <= bottom <= label[4]:
                flag[i] = True
            else:
                flag[i] = False
            i += 1
    # スライスで画像をクロップ
    im = im[top:bottom, left:right]
    
    # クロップ範囲外のlabelを除去する
    ll = [] # テンポラリの空リスト
    for label in labels:
        if label[3] <= left or label[1] >= right or label[4] <= top or label[2] >= bottom:
            pass # クロップ範囲外なので残すリストには追加しない
        else:
            # ここで変換して残すリストに追加
            label[1] = label[1] - left
            label[3] = label[3] - left
            label[2] = label[2] - top
            label[4] = label[4] - top
            ll.append(label)

    labels = np.array(ll)           
    #print('LTRB & labels:', left, top, right, bottom, labels) #debugprint
    return im, labels


def load_mosaic(self, index):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    labels4, segments4 = [], []
    s = self.img_size
    if self.kgtb:
        # mosaicの中心を動かさないために固定する
        yc, xc = s, s    
    else:
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y

    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    if self.kgtb:
        # 以下のcopy_pasteとrandom_perspectiveの代わりにグランドトゥルースボックスを分断せずにクロップする関数を使う
        img4, labels4 = random_crop_keepGTB(self, img4, labels4)     
    else:
        # Augment # を実行するとグランドトゥルースボックスが分断されてしまうためコメントアウトしている
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
        img4, labels4 = random_perspective(img4, labels4, segments4,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

    return img4, labels4


def load_mosaic9(self, index):
    # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img9, labels9


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../datasets/coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../datasets/coco128'):  # from utils.datasets import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../datasets/coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # add image to txt file


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any([len(x) > 8 for x in l]):  # is segment
                    classes = np.array([x[0] for x in l], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                    l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                l = np.array(l, dtype=np.float32)
            nl = len(l)
            if nl:
                assert l.shape[1] == 5, f'labels require 5 columns, {l.shape[1]} columns detected'
                assert (l >= 0).all(), f'negative label values {l[l < 0]}'
                assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
                _, i = np.unique(l, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    l = l[i]  # remove duplicates
                    if segments:
                        segments = segments[i]
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                l = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 5), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """ Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.datasets import *; dataset_stats('../datasets/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

    def unzip(path):
        # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
        if str(path).endswith('.zip'):  # path is data.zip
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            ZipFile(path).extractall(path=path.parent)  # unzip
            dir = path.with_suffix('')  # dataset directory == zip name
            return True, str(dir), next(dir.rglob('*.yaml'))  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    def hub_ops(f, max_dim=1920):
        # HUB ops for 1 image 'f': resize and save at reduced quality in /dataset-hub for web/app viewing
        f_new = im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, 'JPEG', quality=75, optimize=True)  # save
        except Exception as e:  # use OpenCV
            print(f'WARNING: HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(f_new), im)

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_yaml(yaml_path), errors='ignore') as f:
        data = yaml.safe_load(f)  # data dict
        if zipped:
            data['path'] = data_dir  # TODO: should this be dir.resolve()?
    check_dataset(data, autodownload)  # download dataset if missing
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}  # statistics dictionary
    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # load dataset
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        x = np.array(x)  # shape(128x80)
        stats[split] = {'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()},
                        'image_stats': {'total': dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()),
                                        'per_class': (x > 0).sum(0).tolist()},
                        'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in
                                   zip(dataset.img_files, dataset.labels)]}

        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files), total=dataset.n, desc='HUB Ops'):
                pass

    # Profile
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)  # save stats *.json
            t2 = time.time()
            with open(file) as f:
                x = json.load(f)  # load hyps dict
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

    # Save, print and return
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)  # save stats.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats

# ★★自作のFLIRカメラ用のLoad4streamsバージョン
# 20220315　原田
# flirのカメラに関してはThreadingするためには厳密な同期制御が必要な様で
# ここにある様な記述では動作しないことが確認された。よって勿体ないで今ある4台は使いたい
# と思ったが断念することにする。プログラムの記述についてはこのまま残しておく。
# 20220622 原田
class Load4FlirCams:
    # originalのLoadStreamsを弄ってスレッドを使ってFlirのカメラから画像を取込むためのクラス
    def __init__(self, sources='4FlirCams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.flag = True

        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        print(sources)
        n = len(sources)
        try:
            import PySpin
        except:
            print('PySpin is not installed. Please check !')
            sys.exit(0)
        self.imgs, self.fps, self.frames, self.threads = [None] * 4, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        cap = [None] * 4 # カメラインスタンスを格納するリストを定義しておく
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            #s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            s = str(s)
            cap[i] = cam_list.GetBySerial(s)
            #assert cap.isOpened(), f'{st}Failed to open {s}'
            cap[i].Init() 
            #cap.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            #cap.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)
            cap[i].BeginAcquisition()
            w = cap[i].Width() # should be 640
            h = cap[i].Height() # should be 480
            self.fps[i] = 70.0  # 30 FPS fallback
            self.frames[i] = float('inf')  # infinite stream fallback
            #print('w h:', type(w),type(h))
            if cap[i].IsValid():
                im = cap[i].GetNextImage()  # guarantee first frame
                self.imgs[i] = im.GetNDArray()
                #self.imgs[i] = im.GetData().reshape(w, h, 3)
                print('im.shape:', self.imgs[i].shape)
                print('im.dtype:', self.imgs[i].dtype)
                #cv2.imshow('test ', self.imgs[i])
                cv2.imwrite('c:/users/aiuse/desktop/t1.jpg', self.imgs[i])
                #im.release()
                self.threads[i] = Thread(target=self.update, args=([i, cap[i], s]), daemon=False)
                LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
                self.threads[i].start()
            
                #print('** ', self.threads) # debug print
            else:
                print(f'{st}Failed to open Cam {s}')
                self.imgs[i] = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)

        LOGGER.info('')  # newline
        # ここで4つの画像を合成する
        '''
        if len(self.sources) == 1:
            self.imgs[1] = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
            print('imgs1.shape:', self.imgs[1].shape)
            print('imgs1.dtype:', self.imgs[1].dtype)
            cv2.imwrite('c:/users/aiuse/desktop/t2.jpg', self.imgs[0])
        self.concimg = cv2.hconcat([self.imgs[0], self.imgs[1]])
        if len(self.sources) > 2:
            if len(self.sources) == 3:
                self.imgs[3] = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
            conc2 = cv2.hconcat([self.imgs[2], self.imgs[3]])
            self.concimg = cv2.vconcat([self.concimg, conc2])
            self.concimg = cv2.resize(self.concimg, (800, 600), interpolation = cv2.INTER_AREA)
        else:
            self.concimg = cv2.resize(self.concimg, (800, 300), interpolation = cv2.INTER_AREA)
        self.obi = np.full((20, 800, 3), (255, 255, 255), dtype=np.uint8)
        self.concimg = cv2.vconcat([self.concimg, self.obi])
        '''
        #self.concimg = np.expand_dims(self.concimg, axis=0) # CHW > BCHW
        self.concimg = np.expand_dims(self.imgs[i], axis=0) # CHW > BCHW
        
        # check for common shapes
        #s0 = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        s0 = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.concimg])
        self.rect = np.unique(s0, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            LOGGER.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cam, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        #while cap.IsValid() and n < f and self.flag:
        print('updateのwhileの前です')
        while n < f and self.flag:
            print('updateのwhileの直後です')
            n += 1
            if n % read == 0:
                print('updateのif n % read == 0の直後です')
                im = cam.GetNextImage()
                print('updateのGetNextImageの直後です')
                #if success:
                self.imgs[i] = im
                cv2.imwrite('c:/users/aiuse/desktop/t2.jpg', self.imgs[i])
                '''
                else:
                
                    LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                '''
                    #cap.open(stream)  # re-open stream if signal was lost
            #im.release()
            time.sleep(1 / self.fps[i])  # wait time
        cap.EndAcquisition()
        cap.DeInit
        del cap        

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        #if not all(x.isAlive() for x in self.threads) or cv2.waitKey(1) == 27: #ord('q'):  # q to quit
        if cv2.waitKey(1) == 27:#ord('q'):  # q to quit 
            self.flag = False
            cv2.destroyAllWindows()
            raise StopIteration
            
        # ここで4つの画像を合成する
        '''
        if len(self.sources) == 1:
            self.imgs[1] = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
        self.concimg = cv2.hconcat([self.imgs[0], self.imgs[1]])
        if len(self.sources) > 2:
            if len(self.sources) == 3:
                self.imgs[3] = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
            conc2 = cv2.hconcat([self.imgs[2], self.imgs[3]])
            self.concimg = cv2.vconcat([self.concimg, conc2])
            self.concimg = cv2.resize(self.concimg, (800, 600), interpolation = cv2.INTER_AREA)
        else:
            self.concimg = cv2.resize(self.concimg, (800, 300), interpolation = cv2.INTER_AREA)
        self.obi = np.full((20, 800, 3), (255, 255, 255), dtype=np.uint8)
        self.concimg = cv2.vconcat([self.concimg, self.obi])
        '''
        #self.concimg = np.expand_dims(self.concimg, axis=0) # CHW > BCHW 
        self.concimg = np.expand_dims(self.imgs[i], axis=0) # CHW > BCHW 

        # Letterbox
        img0 = self.concimg.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW 
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years
# ★★からここまで…