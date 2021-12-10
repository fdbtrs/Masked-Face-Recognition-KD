from easydict import EasyDict as edict

config = edict()
config.dataset = "emore"  # MS1MV2
config.embedding_size = 512
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 1.0
config.output = "output/resume"
config.scale=1.0
config.global_step=0
config.s=64.0
config.m=0.5


# for KD
config.teacher_pth = "/home/fboutros/arcface_torch/output/emore_random_resnet"
config.teacher_global_step = 295672
config.teacher_network="resnet100"

# if use pretrained model (not for resume!)
config.student_pth = "output/resume_0.01"
config.student_global_step = 227440
config.net_name="resnet100"


config.w=3000
if config.dataset == "emore":
    config.rec = "/data/fboutros/faces_emore"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch = 26
    config.warmup_epoch = -1
    config.val_targets =["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step = 5686
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14, 20, 25] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "webface":
    config.rec = "data/faces_webface_112x112"
    config.num_classes = 10572
    config.num_image = 501195
    config.num_epoch = 34
    config.warmup_epoch = -1
    config.val_targets =  ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [20, 28, 32] if m - 1 <= epoch])
    config.lr_func = lr_step_func
