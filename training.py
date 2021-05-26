###############################################################################
# MIT License
#
# Copyright (c) 2021, Jun So Intec Inc. All rights reserved.
#
# Author & Contact: Jun So (so_jun@intec.co.jp)
###############################################################################
import argparse
import os
from PIL import Image
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
from math import ceil
import random

import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torch.utils import data
from torchvision import transforms

from facenet_pytorch import MTCNN
from unet import VGG16FeatureExtractor, PConvUNet, InpaintingLoss


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
LAMBDA_DICT = {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}


class InfiniteSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, num_samples):
        super().__init__(self)
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed(0)
                order = np.random.permutation(self.num_samples)
                i = 0


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


class datafeed(torch.utils.data.Dataset):
    def __init__(self, img_root, split='train'):
        super(datafeed, self).__init__()
        if split == 'train':
            self.paths = glob('{:s}/train/*.jpg'.format(img_root), recursive=True)
        else:
            self.paths = glob('{:s}/val/*.jpg'.format(img_root))
            random.shuffle(self.paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index]).convert('RGB')
        temp_img = mask_transform(gt_img)
        # temp_img = np.array(gt_img, dtype=np.uint8)
        gt_img = img_transform(gt_img)
        return gt_img, temp_img

    def __len__(self):
        return len(self.paths)


class make_mask_mtcnn():
    def __init__(self, mask_device):
        self.mtcnn = MTCNN(select_largest=False, device=mask_device)
        self.offset_rate = 0.1
        self.mask_rate = 0.65
        self.mask_device = mask_device

    def get_mask(self, gt_img, mode='train'):
        if mode == 'train':
            img = gt_img.to('cpu').detach().numpy().copy().transpose(1, 2, 0)
            img = (img*255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = pil2cv(gt_img)

        margin = 50
        padding = cv2.copyMakeBorder(img, margin, margin, margin, margin, cv2.BORDER_CONSTANT)
        image = Image.fromarray(padding)

        # gives the face co-ordinates
        face_coord, _ = self.mtcnn.detect(image)
        detect_face = 1
        face_images = []
        face_masks = []
        face_points = []
        if face_coord is not None:
            for coord in face_coord:
                if detect_face > 0:
                    for x1, y1, x2, y2 in [coord]:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # face array
                        face_baseX1, face_baseY1 = x1 - margin, y1 - margin
                        face_baseX2, face_baseY2 = x2 - margin, y2 - margin
                        face_baseW, face_baseH = face_baseX2 - face_baseX1, face_baseY2 - face_baseY1
                        offset_x, offset_y = int(face_baseW * self.offset_rate), int(face_baseH * self.offset_rate)

                        face_baseX1 = max(0, face_baseX1 - offset_x)
                        face_baseX2 = max(0, face_baseX2 + offset_x)
                        face_baseY1 = max(0, face_baseY1 - offset_y)
                        face_baseY2 = max(0, face_baseY2 + offset_y)

                        img = img[face_baseY1:face_baseY2, face_baseX1:face_baseX2, :]
                        # print(img.shape)
                        face_images.append(cv2pil(img).convert('RGB'))
                        face_points.append([face_baseY1, face_baseY2, face_baseX1, face_baseX2])
                        face_mask = np.full(img.shape[:2], 255, dtype=img.dtype)

                        cv2.rectangle(face_mask,
                                      (0, int(img.shape[0] * self.mask_rate)),
                                      (img.shape[1], img.shape[0]),
                                      (0, 0, 0), -1)
                        face_masks.append(cv2pil(face_mask).convert("L").convert("RGB"))

                        detect_face -= 1
                        # cv2.imshow("Frame", img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
        else:
            # ra
            height, width = img.shape[:2]

            face_baseX1 = int(random.uniform(margin, width))
            face_baseY1 = int(random.uniform(margin, height))
            face_baseX2 = int(random.uniform(face_baseX1 + margin, width))
            face_baseY2 = int(random.uniform(face_baseY1 + margin, height))

            # face array
            # face_baseX1, face_baseY1 = x1 - margin, y1 - margin
            # face_baseX2, face_baseY2 = x2 - margin, y2 - margin
            # face_baseW, face_baseH = face_baseX2 - face_baseX1, face_baseY2 - face_baseY1
            # offset_x, offset_y = int(face_baseW * self.offset_rate), int(face_baseH * self.offset_rate)
            # print(img.shape)
            # img = img[face_baseY1:face_baseY2, face_baseX1:face_baseX2, :]
            face_images.append(cv2pil(img).convert('RGB'))
            face_points.append([0, height, 0, width])
            face_mask = np.full(img.shape[:2], 255, dtype=img.dtype)

            cv2.rectangle(face_mask,
                          (face_baseX1, face_baseY1),
                          (face_baseX2, face_baseY2),
                          (0, 0, 0), -1)
            # cv2.imshow("Frame", face_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            face_masks.append(cv2pil(face_mask).convert("L").convert("RGB"))

        return face_masks, face_images, face_points


    def get_masks(self, gt_imgs, device):
        masks = []
        imgs = []

        for gt_img in gt_imgs:
            face_masks, face_images, face_points = self.get_mask(gt_img, mode='train')
            img = img_transform(face_images[0])
            mask = mask_transform(face_masks[0])
            imgs.append(img.to(device))
            masks.append(mask.to(device))
        imgs = torch.stack(imgs, 0)
        masks = torch.stack(masks, 0)
        return imgs, masks


def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name)
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['n_iter']


def save_ckpt(ckpt_name, models, optimizers, n_iter):
    ckpt_dict = {'n_iter': n_iter}
    for prefix, model in models:
        cpu_device = torch.device('cpu')
        state_dict = model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(cpu_device)
        # return state_dict
        ckpt_dict[prefix] = state_dict

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)

    return ckpt_name


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(STD) + torch.Tensor(MEAN)
    x = x.transpose(1, 3)
    return x


def evaluate(model, dataset, make_mask, device, filename):
    gt = [dataset[i][0].to(device) for i in range(8)]
    temp_img = [dataset[i][1].to(device) for i in range(8)]
    gt = torch.stack(gt)
    temp_img = torch.stack(temp_img)

    # _, mask = make_mask.get_masks(temp_img, device)

    gt, mask = make_mask.get_masks(temp_img, device)

    image = gt * mask

    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    mask = mask.to(torch.device('cpu'))
    image = image.to(torch.device('cpu'))
    gt = gt.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, filename)


def train(args, device, mask_device):

    if not os.path.exists(args.save_dir):
        os.makedirs('{:s}/images'.format(args.save_dir))
        os.makedirs('{:s}/model'.format(args.save_dir))

    model = PConvUNet(layer_size=7, input_channels=3, upsampling_mode='nearest').to(device)

    dataset_train = datafeed(args.root, 'train')
    dataset_val = datafeed(args.root, 'val')

    iterator_train = iter(data.DataLoader(dataset_train,
                                          batch_size=args.batch_size,
                                          sampler=InfiniteSampler(len(dataset_train)),
                                          num_workers=args.n_threads))

    print(len(dataset_train))

    start_iter = 0
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

    make_mask = make_mask_mtcnn(mask_device)

    model_path = ''
    for i in tqdm(range(start_iter, args.max_iter)):
        model.train()

        gt, temp_img = [x.to(device) for x in next(iterator_train)]
        gt, mask = make_mask.get_masks(temp_img, device)

        image = gt * mask

        output, _ = model(image, mask)
        loss_dict = criterion(image, mask, output, gt)

        # 可視化用
        # grid = make_grid(unnormalize(image.to(torch.device('cpu'))))
        # save_image(grid, "/home/intec/nvme/DeepLearning/pytorch-inpainting-with-partial-conv-master/res/images/test.jpg")

        loss = 0.0
        for key, coef in LAMBDA_DICT.items():
            value = coef * loss_dict[key]
            loss += value
            if (i + 1) % args.save_model_interval == 0:
                print('loss_{:s}'.format(key), value.item(), i + 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            model_path = save_ckpt('{:s}/model/{:d}.pth'.format(args.save_dir, i + 1), [('model', model)], [('optimizer', optimizer)], i + 1)
            model.eval()
            evaluate(model, dataset_val, make_mask, device, '{:s}/images/test_{:d}.jpg'.format(args.save_dir, i + 1))

    return model_path


def main(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')

    # スナップショットが指定されていなければ、学習開始
    if args.mode == 'train':
        # mask_device = torch.device('cpu')
        mask_device = device
        args.snapshot = train(args, device, mask_device)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # common options---------------------------------------
    parser.add_argument('--save_dir', type=str, default='/home/intec/nvme/DeepLearning/pytorch-inpainting-with-partial-conv-master/res')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--mode', type=str, default='train')

    # training options--------------------------------------
    parser.add_argument('--root', type=str, default='/home/intec/nvme/DeepLearning/img_celeba')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max_iter', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_threads', type=int, default=0)
    parser.add_argument('--save_model_interval', type=int, default=1)

    args = parser.parse_args()

    size = (args.image_size, args.image_size)
    img_transform = transforms.Compose([transforms.Resize(size=size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=MEAN, std=STD)])
    mask_transform = transforms.Compose([transforms.Resize(size=size),
                                         transforms.ToTensor()])




    device = torch.device('cpu')
    model = PConvUNet(layer_size=7, input_channels=3, upsampling_mode='nearest').to(device)
    ckpt_dict = torch.load('/home/intec/nvme/SIGNATE/無題のフォルダー/res_batch16_layer7/model/100000.pth')
    model.load_state_dict(ckpt_dict['model'], strict=False)
    # model.load_state_dict(torch.load('/home/intec/nvme/DeepLearning/pytorch-inpainting-with-partial-conv-master/res/100000.pth'))

    model.eval()
    # evaluate(model, dataset_val, make_mask, device, '{:s}/images/test_{:d}.jpg'.format(args.save_dir, i + 1))
    # frame = cv2.imread("/home/intec/nvme/DeepLearning/img_celeba/train/000055.jpg")
    # gt = [dataset[i][0].to(device) for i in range(8)]
    # temp_img = [dataset[i][1].to(device) for i in range(8)]

    gt_img = Image.open("/home/intec/nvme/DeepLearning/img_celeba/train/000055.jpg").convert('RGB')
    temp_img = mask_transform(gt_img).unsqueeze(0)
    gt_img = img_transform(gt_img).unsqueeze(0)


    # gt = torch.stack(gt_img)
    # temp_img = torch.stack(temp_img)
    make_mask = make_mask_mtcnn(device)
    gt, mask = make_mask.get_masks(temp_img, device)

    image = gt * mask

    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    mask = mask.to(torch.device('cpu'))
    image = image.to(torch.device('cpu'))
    gt = gt.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, '/home/intec/nvme/DeepLearning/img_celeba/test.jpg')

    main(args)

