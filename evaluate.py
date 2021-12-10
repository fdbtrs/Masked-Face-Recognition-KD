import logging
import os

#import cv2
import sys
import torch

#import backbones.mixnetm as mx
from backbones import iresnet100
from utils.utils_callbacks import CallBackVerification
from utils.utils_logging import init_logging

sys.path.append('/root/xy/work_dir/xyface/')
from config import config as cfg



if __name__ == "__main__":
    
    load_color = True
    both_masked = False
    
    gpu_id = 3
    log_root = logging.getLogger()
    init_logging(log_root, 0, cfg.output,logfile="test_student2_single.log")
    callback_verification = CallBackVerification(1, 0, cfg.val_targets, cfg.rec, load_color=load_color, both_masked=both_masked)
    output_folder='/home/fboutros/Masked-Face-Recognition-KD/eval'
    weights=os.listdir(output_folder)

    with torch.no_grad():
        for w in weights:
            if "backbone" in w:
                print("Evaluating:", w)
                backbone = iresnet100(num_features=cfg.embedding_size).to(f"cuda:{gpu_id}")
                backbone.load_state_dict(torch.load(os.path.join(output_folder,w)))
                model = torch.nn.DataParallel(backbone, device_ids=[gpu_id])
                callback_verification(int(w.split("backbone")[0]),backbone)

