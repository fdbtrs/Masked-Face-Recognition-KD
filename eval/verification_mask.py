"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import datetime
import os
import pickle
import cv2
import logging
import sys

import mxnet as mx
import numpy as np
import sklearn
import torch
from mxnet import ndarray as nd
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from torchvision import transforms

from config import config as cfg

landmarks = np.array([[30.2946, 51.6963],
                      [65.5318, 51.5014],
                      [48.0252, 71.7366],
                      [33.5493, 92.3655],
                      [62.7299, 92.2041]
                      ], dtype=np.float32 )


mask_img = cv2.imread("mask_img.png", cv2.IMREAD_UNCHANGED)

class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print(true_accept, false_accept)
    # print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

@torch.no_grad()
def load_bin(path, image_size, both_masked=True, load_color=True):
    
    dataset = path.split("/")[-1][:-4]
    print(dataset)
    
    color_path = "/home/fboutros/Masked-Face-Recognition-KD/eval/" + dataset + "_both_" + str(both_masked) + ".txt"

    # load data
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
            
    # load color
    if load_color == True:
        try:
            file = open(color_path, 'rb')
            mask_colors = file.readlines()
            print(len(mask_colors))
            file.close()

        except (FileNotFoundError, IOError):
            print("Wrong file or file path")
            
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    
    color_list = []
    
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        
        # mask2mask
        if both_masked == True:
            
            # get and save new colors
            if load_color == False:
                img, col = simulatedMask(img.asnumpy())
                img = mx.nd.array(img)
                tmp = str(idx) + "," + str(col)
                color_list.append(tmp)
                #img, col = mx.nd.array(simulatedMask(img.asnumpy()))
            
            # load pre-set color
            else:
                tmp = str(mask_colors[idx])
                color = tmp.split("[")[1]
                color = color.split("]")[0]
                color = np.fromstring(color, dtype=int, sep=" ")
                img, _ = simulatedMask(img.asnumpy(), color)
                img = mx.nd.array(img)

        # mask2nomask
        if both_masked == False:
            
            # get and save new colors
            if load_color == False:
                if (idx>=len(issame_list)):
                    img, col = simulatedMask(img.asnumpy())
                    img = mx.nd.array(img)
                    tmp = str(idx) + "," + str(col)
                    color_list.append(tmp)
                    
            else:
                if (idx>=len(issame_list)):
                    tmp = str(mask_colors[idx - len(issame_list)])
                    color = tmp.split("[")[1]
                    color = color.split("]")[0]
                    color = np.fromstring(color, dtype=int, sep=" ")
                    img, _ = simulatedMask(img.asnumpy(), color)
                    img = mx.nd.array(img)
            
            # add mask on one
            #if (idx>=len(issame_list)):
            #    img, _ = mx.nd.array(simulatedMask(img.asnumpy()))
                
                
        
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 1000 == 0:
            print('loading bin', idx)
    print(data_list[0].shape)
    
    # only saved when not loaded
    if load_color == False:
        with open(color_path, 'w') as f:
            for item in color_list:
                f.write("%s\n" % item)
            f.close()
    
    return data_list, issame_list

def _getFeatureBlob(input_blob,model):
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        model.forward(db, is_train=False)
        emb = model.get_outputs()[0].asnumpy()
        return  emb

trans = transforms.Compose([

        transforms.Normalize(
            mean=[0., 0., 0.],
            std=[1., 1., 1.]),
    ])
@torch.no_grad()
def test(data_set, backbone, batch_size, nfolds=10,model_name="MagFace"):
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            time0 = datetime.datetime.now()
            if (model_name=="ArcFace"):
                 _embeddings= _getFeatureBlob(_data, backbone )
            elif (model_name=="MagFace"):
                img = ((_data / 255) ) 
                net_out: torch.Tensor = backbone(img.to("cuda:3"))
                #net_out: torch.Tensor = backbone(img.cuda())
                _embeddings = net_out.detach().cpu().numpy()
              
            else:
             img = ((_data / 255) - 0.5) / 0.5
             net_out: torch.Tensor = backbone(img.to("cuda:3"))
             #net_out: torch.Tensor = backbone(img.cuda())
             _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list


def dumpR(data_set,
          backbone,
          batch_size,
          name='',
          data_extra=None,
          label_shape=None):
    print('dump verification embedding..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba

            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
            time0 = datetime.datetime.now()
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra),
                                     label=(_label,))
            model.forward(db, is_train=False)
            net_out = model.get_outputs()
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    actual_issame = np.asarray(issame_list)
    outname = os.path.join('temp.bin')
    with open(outname, 'wb') as f:
        pickle.dump((embeddings, issame_list),
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL)


def simulatedMask(img, color=False):
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        #rs = np.random.randint(-40,40)
        #rx = np.random.randint(-10,10)
        rs = 0
        rx = 0
    
        #keypoints of mask image
        src_pts = np.array([np.array([678+rx,464+rs]), 
                            np.array([548+rx,614+rs]), 
                            np.array([991+rx,664+rs]), 
                            np.array([1009+rx,64+rs]), 
                            np.array([557+rx,64+rs])], dtype="float32")

        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         mask_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = cymk_to_rgb(transformed_mask)
        
        if type(color) is not np.ndarray:
            random_value = np.random.randint(0,150,3)
        else:
            random_value = color
            
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img, random_value
    
def cymk_to_rgb(img):
        cyan = img[:,:,0] 
        magenta = img[:,:,1] 
        yellow = img[:,:,2] 
        black = img[:,:,3]
        
        scale = 100
        red = 255*(1.0-(cyan+black)/float(scale))
        green = 255*(1.0-(magenta+black)/float(scale))
        blue = 255*(1.0-(yellow+black)/float(scale))
            
        rgbimg = np.stack((red, green, blue))
        rgbimg = np.moveaxis(rgbimg, 0, 2)
        return rgbimg
    
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='do verification')
#     # general
#     parser.add_argument('--data-dir', default='', help='')
#     parser.add_argument('--model',
#                         default='../model/softmax,50',
#                         help='path to load model.')
#     parser.add_argument('--target',
#                         default='lfw,cfp_ff,cfp_fp,agedb_30',
#                         help='test targets.')
#     parser.add_argument('--gpu', default=0, type=int, help='gpu id')
#     parser.add_argument('--batch-size', default=32, type=int, help='')
#     parser.add_argument('--max', default='', type=str, help='')
#     parser.add_argument('--mode', default=0, type=int, help='')
#     parser.add_argument('--nfolds', default=10, type=int, help='')
#     args = parser.parse_args()
#     image_size = [112, 112]
#     print('image_size', image_size)
#     ctx = mx.gpu(args.gpu)
#     nets = []
#     vec = args.model.split(',')
#     prefix = args.model.split(',')[0]
#     epochs = []
#     if len(vec) == 1:
#         pdir = os.path.dirname(prefix)
#         for fname in os.listdir(pdir):
#             if not fname.endswith('.params'):
#                 continue
#             _file = os.path.join(pdir, fname)
#             if _file.startswith(prefix):
#                 epoch = int(fname.split('.')[0].split('-')[1])
#                 epochs.append(epoch)
#         epochs = sorted(epochs, reverse=True)
#         if len(args.max) > 0:
#             _max = [int(x) for x in args.max.split(',')]
#             assert len(_max) == 2
#             if len(epochs) > _max[1]:
#                 epochs = epochs[_max[0]:_max[1]]
#
#     else:
#         epochs = [int(x) for x in vec[1].split('|')]
#     print('model number', len(epochs))
#     time0 = datetime.datetime.now()
#     for epoch in epochs:
#         print('loading', prefix, epoch)
#         sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
#         # arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
#         all_layers = sym.get_internals()
#         sym = all_layers['fc1_output']
#         model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
#         # model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
#         model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0],
#                                           image_size[1]))])
#         model.set_params(arg_params, aux_params)
#         nets.append(model)
#     time_now = datetime.datetime.now()
#     diff = time_now - time0
#     print('model loading time', diff.total_seconds())
#
#     ver_list = []
#     ver_name_list = []
#     for name in args.target.split(','):
#         path = os.path.join(args.data_dir, name + ".bin")
#         if os.path.exists(path):
#             print('loading.. ', name)
#             data_set = load_bin(path, image_size)
#             ver_list.append(data_set)
#             ver_name_list.append(name)
#
#     if args.mode == 0:
#         for i in range(len(ver_list)):
#             results = []
#             for model in nets:
#                 acc1, std1, acc2, std2, xnorm, embeddings_list = test(
#                     ver_list[i], model, args.batch_size, args.nfolds)
#                 print('[%s]XNorm: %f' % (ver_name_list[i], xnorm))
#                 print('[%s]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], acc1, std1))
#                 print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], acc2, std2))
#                 results.append(acc2)
#             print('Max of [%s] is %1.5f' % (ver_name_list[i], np.max(results)))
#     elif args.mode == 1:
#         raise ValueError
#     else:
#         model = nets[0]
#         dumpR(ver_list[0], model, args.batch_size, args.target)
