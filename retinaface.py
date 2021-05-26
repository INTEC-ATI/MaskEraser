from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
# from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time


# def check_keys(model, pretrained_state_dict):
#     ckpt_keys = set(pretrained_state_dict.keys())
#     model_keys = set(model.state_dict().keys())
#     used_pretrained_keys = model_keys & ckpt_keys
#     unused_pretrained_keys = ckpt_keys - model_keys
#     missing_keys = model_keys - ckpt_keys
#     print('Missing keys:{}'.format(len(missing_keys)))
#     print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
#     print('Used keys:{}'.format(len(used_pretrained_keys)))
#     assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
#     return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    # check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    cfg_mnet = {
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

    cfg_re50 = {
        'name': 'Resnet50',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 24,
        'ngpu': 4,
        'epoch': 100,
        'decay1': 70,
        'decay2': 90,
        'image_size': 840,
        'pretrain': True,
        'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
        'in_channel': 256,
        'out_channel': 256
    }

    trained_model = '/home/intec/Downloads/Pytorch_Retinaface-master/Resnet50_Final.pth'  #help='Trained state_dict file path to open')
    network = 'resnet50' #help='Backbone network mobile0.25 or resnet50')
    cpu = False

    torch.set_grad_enabled(False)
    cfg = None
    if network == "mobile0.25":
        cfg = cfg_mnet
    elif network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, trained_model, cpu)
    net.eval()
    # print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if cpu else "cuda")
    net = net.to(device)

    # resize = 1

    # testing begin
    # for i in range(100):
    v = cv2.VideoCapture(0)
    # フォーマット・解像度・FPSの設定
    # v.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    v.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    v.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    v.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    v.set(cv2.CAP_PROP_FPS, 25)
    # v = cv2.VideoCapture("/home/intec/Videos/【白い怪物、爆誕。】最高にイカつい自作PCが完成しました。.mp4")

    while (v.isOpened()):
        ret, img_raw = v.read()
        if (ret == False):
            break


        # image_path = "/home/intec/Downloads/ivan.jpg"
        # img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        confidence_threshold = 0.02
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        top_k = 5000
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        nms_threshold = 0.4
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        keep_top_k = 750
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # show image
        # if args.save_image:
        vis_thres = 0.6
        for b in dets:
            if b[4] < vis_thres:
                continue
            # text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            # cx = b[0]
            # cy = b[1] + 12
            # cv2.putText(img_raw, text, (cx, cy),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        # save image

        name = "/home/intec/Downloads/test.jpg"
        # cv2.imwrite(name, img_raw)
        cv2.imshow("test", img_raw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    v.release()
    cv2.destroyAllWindows()