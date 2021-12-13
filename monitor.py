
import queue
import threading

import time
import os
import sys
import glob
import cv2
import torch
import inspect
import numpy as np

from pathlib import Path
from numpy import random

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_synchronized





# added for threading

help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff',
               'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg',
               'm4v', 'wmv', 'mkv']  # acceptable video suffixes


def frame_writer(q):
    while True:
        (save_frame, im0) = q.get()
        if save_frame == "over":
            break
        cv2.imwrite(save_frame, im0)


def plot_one_box_local(x, img0, instance_path, color=None, label=None, line_thickness=3):
    # def plot_one_box_local(q, x, img0, instance_path, color=None, label=None, line_thickness=3):
    # original_clone = img0[c1[1]:c2[1], c1[0]:c2[0]].copy()
    # q.put((instance_path, original_clone))
    # Plots one bounding box on image img0
    tl = line_thickness or round(
        0.002 * (img0.shape[0] + img0.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    instance_image = img0[c1[1]:c2[1], c1[0]:c2[0]].copy()

    cv2.imwrite(instance_path, instance_image)

    cv2.rectangle(img0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img0, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img0, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def detect(model, dataset, output_dir, threshold):
    imgsz = 640
    webcam = False
    save_dir = output_dir
    instances_dir = os.path.join(save_dir, 'instances')
    # print(instances_dir)

    if not os.path.exists(save_dir):  # output dir
        os.mkdir(save_dir)
        os.mkdir(instances_dir)

    # Initialize
    # set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(model, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors

    names = model.module.names if hasattr(model, 'module') else model.names

    # Sending classes info to UI module
    yield(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    q = queue.Queue(4)
    NWRITERS = 4
    threads = []
    for _ in range(NWRITERS):
        t = threading.Thread(target=frame_writer, args=(q,))
        t.start()
        threads.append(t)
    t0 = time.time()

    for index, (path, img, im0s, vid_cap, cur_frame) in enumerate(dataset):
        image_path = os.path.join(save_dir, f'{cur_frame:08d}')
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, threshold, 0.45, classes=False, agnostic=False)
        t2 = time_synchronized()

        frame_info = {'frame': index + 1, 'total_instance': 0}
        instances = []
        for i, det in enumerate(pred):  # detections per image

            # Process detections
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                frame_info['classes'] = {}

                # save frame info
                for c in det[:, -1].unique():
                    class_name = names[int(c)]
                    sum_cls_instances = (det[:, -1] == c).sum().item()
                    mean_cls_confidence = det[det[:, -1]
                                              == c][:, -2].mean().item()
                    frame_info['classes'][class_name] = [
                        sum_cls_instances, mean_cls_confidence]

                # Write results
                for index, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    instance_path = f"{instances_dir}/{cur_frame:08d}_{index}.jpg"
                    label = f'{names[int(cls)]} {conf:.2f}'
                    # print(instance_path)
                    # label = f'{names[int(cls)]} {index_instance}'

                    plot_one_box_local(xyxy, im0, instance_path, label=label,
                                       color=colors[int(cls)], line_thickness=3)

                    instances.append(instance_path)
            q.put((image_path + '.jpg', im0))
        # if index == 0 or len(bboxes):
        frame_info['instances'] = instances
        frame_info['total_instance'] = len(instances)
        yield(frame_info)
        # print(f'{s}Done. ({t2 - t1:.3f}s)')

    for _ in range(NWRITERS):
        q.put(("over", "over"))
    for thread in threads:
        thread.join()

    print(f'Done. ({time.time() - t0:.3f}s)')


class LoadImages:  # for inference
    def __init__(self, path, offset=0, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0], offset)  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            # video inference log commented out
            # print(
            #     f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, self.frame

    def new_video(self, path, offset):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.cap.set(1, offset)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
