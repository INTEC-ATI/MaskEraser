###############################################################################
# MIT License
#
# Copyright (c) 2021, Jun So Intec Inc. All rights reserved.
#
# Author & Contact: Jun So (so_jun@intec.co.jp)
###############################################################################
import argparse
from PIL import Image
import tkinter
import PIL.Image
import PIL.ImageTk
import numpy as np
import cv2
from math import ceil
import os

import torch
from torchvision import transforms
from unet import PConvUNet
from datetime import datetime
from models.retinaface import RetinaFace
from itertools import product as product


def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGR)
    return new_image


def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGB)
    new_image = Image.fromarray(new_image)
    return new_image


def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

# https://github.com/biubug6/Pytorch_Retinaface/blob/master/layers/functions/prior_box.py
class PriorBox(object):
    def __init__(self, cfg, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class make_mask_RetinaFace():
    def __init__(self, mask_device):
        self.cfg_mnet = {
            'name': 'mobilenet0.25',
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'gpu_train': True,
            'batch_size': 32,
            'ngpu': 1,
            'epoch': 250,
            'decay1': 190,
            'decay2': 220,
            'image_size': 640,
            'pretrain': True,
            'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
            'in_channel': 32,
            'out_channel': 64
        }

        trained_model = './weight/mobilenet0.25_Final.pth'  # help='Trained state_dict file path to open')

        torch.set_grad_enabled(False)
        net = RetinaFace(cfg=self.cfg_mnet, phase='test', checkpoint='./weight/mobilenetV1X0.25_pretrain.tar')
        # load_model
        pretrained_dict = torch.load(trained_model, map_location=lambda storage, loc: storage.cuda(mask_device))
        pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        net.load_state_dict(pretrained_dict, strict=False)

        net.eval()
        self.net = net.to(mask_device)

        self.offset_rate = 0.1
        self.mask_rate = 0.5#0.65
        self.mask_device = mask_device

    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        # print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def find_face(self, img_raw):
        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.mask_device)
        scale = scale.to(self.mask_device)

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg_mnet, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.mask_device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg_mnet['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        confidence_threshold = 0.5
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # NMS
        nms_threshold = 0.4
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= nms_threshold)[0]
            order = order[inds + 1]
        boxes = boxes[keep]
        scores = scores[keep]

        vis_thres = 0.6
        face_coord = []
        for box, score in zip(boxes, scores):
            if score < vis_thres:
                continue
            b = list(map(int, box))
            face_coord.append([b[0], b[1], b[2], b[3]])
        return face_coord


    def get_mask(self, gt_img):
        img = pil2cv(gt_img)

        margin = 50
        padding = cv2.copyMakeBorder(img, margin, margin, margin, margin, cv2.BORDER_CONSTANT)

        face_coord = self.find_face(padding)

        detect_face = 10
        face_images = []
        face_masks = []
        face_points = []
        if face_coord is not None:
            for coord in face_coord:
                if len(coord) < 4:
                    continue

                if detect_face > 0:
                    for x1, y1, x2, y2 in [coord]:
                        x1, y1, x2, y2 = ceil(x1), ceil(y1), ceil(x2), ceil(y2)

                        # face array
                        face_baseX1, face_baseY1 = x1 - margin, y1 - margin
                        face_baseX2, face_baseY2 = x2 - margin, y2 - margin
                        face_baseW, face_baseH = face_baseX2 - face_baseX1, face_baseY2 - face_baseY1
                        offset_x, offset_y = int(face_baseW * self.offset_rate), int(face_baseH * self.offset_rate)
                        face_baseX1 = max(0, face_baseX1 - offset_x)
                        face_baseX2 = max(0, face_baseX2 + offset_x)
                        face_baseY1 = max(0, face_baseY1 - offset_y)
                        face_baseY2 = max(0, face_baseY2 + offset_y)

                        img_temp = img[face_baseY1:face_baseY2, face_baseX1:face_baseX2, :]
                        face_images.append(cv2pil(img_temp).convert('RGB'))
                        face_points.append([face_baseY1, face_baseY2, face_baseX1, face_baseX2])
                        face_mask = np.full(img_temp.shape[:2], 255, dtype=img_temp.dtype)
                        cv2.rectangle(face_mask,
                                      (0, int(img_temp.shape[0] * self.mask_rate)),
                                      (img_temp.shape[1], img_temp.shape[0]),
                                      (0, 0, 0), -1)
                        face_masks.append(cv2pil(face_mask).convert("L").convert("RGB"))

                    detect_face -= 1

        return face_masks, face_images, face_points


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(STD) + torch.Tensor(MEAN)
    x = x.transpose(1, 3)
    return x


class App:
    def __init__(self, args, window):
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
        self.output = args.output

        self.model = PConvUNet(layer_size=7, input_channels=3, upsampling_mode='nearest').to(device)
        ckpt_dict = torch.load(args.snapshot)
        self.model.load_state_dict(ckpt_dict['model'], strict=False)
        self.model.eval()

        self.make_mask = make_mask_RetinaFace(device)

        self.window = window
        self.window.title("MaskeEraser")

        # 保存用-----------------------------------------------------------------------------------
        # self.vcap = cv2.VideoCapture('/home/intec/nvme/動画/production ID 3960181.mp4')
        # self.width = int(self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # self.height = int(self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # size = (self.width, self.height)
        # fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # フレームレート(1フレームの時間単位はミリ秒)の取得
        # frame_rate = int(self.vcap.get(cv2.CAP_PROP_FPS))
        # self.writer = cv2.VideoWriter('/home/intec/nvme/動画/production ID 3960181_outmask.mp4', fmt, frame_rate, size)

        self.vcap = cv2.VideoCapture(0)
        #フォーマット・解像度・FPSの設定
        self.width = 640
        self.height = 360
        # self.vcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.vcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
        self.vcap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.vcap.set(cv2.CAP_PROP_FPS, 25)

        # カメラモジュールの映像を表示するキャンバスを用意する
        self.canvas = tkinter.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # ボタン
        self.close_btn = tkinter.Button(window, text="終了")
        self.close_btn.pack(fill='x', padx=20, side='right')
        self.close_btn.configure(command=self.close)

        self.satuei_btn = tkinter.Button(window, text="撮影")
        self.satuei_btn.pack(fill='x', padx=20, side='right')
        self.satuei_btn.configure(command=self.satuei)

        self.delay = 30
        self.update()

        self.window.mainloop()

    def update(self):
        _, frame = self.vcap.read()

        # frame = cv2.imread("/home/intec/nvme/DeepLearning/img_celeba/train/000055.jpg")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_masks, face_images, face_points = self.make_mask.get_mask(frame)

        for mask, face_image, face_point in zip(face_masks, face_images, face_points):
            gt_img = img_transform(face_image)
            mask = mask_transform(mask)

            image = gt_img * mask
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)

            with torch.no_grad():
                output, _ = self.model(image.to(torch.device('cuda')), mask.to(torch.device('cuda')))
            output = output.to(torch.device('cpu'))
            output = mask * image + (1 - mask) * output

            ndarr = unnormalize(output)[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            img = pil2cv(im)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (face_image.width, face_image.height))
            frame[face_point[0]:face_point[1], face_point[2]:face_point[3], :] = img

        # 保存用-----------------------------------------------------------------------------------
        # self.writer.write(frame)

        self.save_img = PIL.Image.fromarray(frame)
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)

    def satuei(self):
        jst_now = datetime.now()
        self.output = os.path.join(self.output, f'{jst_now.strftime("%Y%m%d%H%M%S")}.jpg')
        self.save_img.save(self.output)

    def close(self):
        self.window.destroy()
        self.vcap.release()
        # 保存用-----------------------------------------------------------------------------------
        # self.writer.release()


if __name__ == '__main__':

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--snapshot', type=str, default='./weight/100000.pth')
    parser.add_argument('--output', type=str, default='./picture')
    args = parser.parse_args()

    size = (args.image_size, args.image_size)
    img_transform = transforms.Compose([transforms.Resize(size=size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=MEAN, std=STD)])
    mask_transform = transforms.Compose([transforms.Resize(size=size),
                                         transforms.ToTensor()])

    App(args, tkinter.Tk())
