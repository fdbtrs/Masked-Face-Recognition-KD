import numbers
import os
import queue as Queue
import threading
import cv2

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

landmarks = np.array([[30.2946, 51.6963],
                      [65.5318, 51.5014],
                      [48.0252, 71.7366],
                      [33.5493, 92.3655],
                      [62.7299, 92.2041]
                      ], dtype=np.float32 )


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
            
        self.mask_img = cv2.imread("mask_img.png", cv2.IMREAD_UNCHANGED)

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        masked_sample=self.mask_images(sample)
        if self.transform is not None:
            sample = self.transform(sample)
            masked_sample=self.transform(sample)
        return masked_sample,sample, label
    
    def mask_images(self,img):
        
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        rs = np.random.randint(-40,40)
        rx = np.random.randint(-10,10)
    
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
                         self.mask_img,
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
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def __len__(self):
        return len(self.imgidx)

    def cymk_to_rgb(self, img):
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
        
