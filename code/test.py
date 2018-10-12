import sys
import cv2
import torch
import numpy as np
import os
from os.path import realpath, dirname, join

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import cxy_wh_2_rect, rect_2_cxy_wh, _compile_results, save_video
import pdb
import glob
import json
import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def main(imagedir, gtdir):
    # load net
    net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
    net = SiamRPNBIG()
    net.load_state_dict(torch.load(net_file))
    net.eval().cuda()

    # warm up
    for i in range(10):
        net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
        net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())
       
    # start to track
    # get the first frame groundtruth
    gt_file = os.path.join(gtdir, 'gt.txt')
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    gt = []
    for line in lines:
        line = line.split(' ')
        gt.append([int(float(x)) for x in line])
    init_bbox = gt[0] # top-left x y,w,h
    target_pos, target_sz = rect_2_cxy_wh(init_bbox) # top-left x y,w,h --> center x y,w,h
    
    image_list = glob.glob(os.path.join(imagedir, '*.jpg'))
    image_list.sort()
    im = cv2.imread(image_list[0])  # HxWxC      

    state = SiamRPN_init(im, target_pos, target_sz, net)  # init tracker
    bboxes = [] 
    for i in range(1, len(gt)):
        im = cv2.imread(image_list[i])   # HxWxC
        state = SiamRPN_track(state, im)  # track
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])    # center x y,w,h --> top-left x y,w,h
        bboxes.append(res.tolist())
            
    _, precision, precision_auc, iou = _compile_results(gt[1:], bboxes)
    print(' -- Precision ' + "(20 px)"  + ': ' + "%.2f" % precision +\
            ' -- Precision AUC: ' + "%.2f" % precision_auc + \
            ' -- IOU: ' + "%.2f" % iou + ' --')
            
    isSavebbox = True
    if isSavebbox:
        print('saving bbox...')
        res_bbox_file = os.path.join('results_bbox.json')
        json.dump(bboxes, open(res_bbox_file, 'w'), indent=2)
    
    isSavevideo = True
    if isSavevideo:
        print('saving video...')
        save_video(image_list, bboxes)
    print('done')

if __name__ == '__main__':
    imagedir = os.path.join('../MOT17-02-DPM/img1')
    gtdir = os.path.join('../MOT17-02-DPM/gt')
    sys.exit(main(imagedir, gtdir))
