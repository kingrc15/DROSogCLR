import os
import argparse
import wandb

import train
import lincls

parser = argparse.ArgumentParser(description="SogCLR Training")
parser.add_argument("--data", metavar="DIR", default="./data/", help="path to dataset")

parser.add_argument(
    "--lr",
    "--learning-rate",
    default=1.0,
    type=float,
    metavar="LR",
    help="initial (base) learning rate",
    dest="lr",
)
parser.add_argument(
    "--t", default=0.3, type=float, help="softmax temperature (default: 0.3)"
)

parser.add_argument(
    "--kl_weight", default=0.001, type=float, help="softmax temperature (default: 0.3)"
)

parser.add_argument(
    "--use_cov", default=False, type=bool, help="softmax temperature (default: 0.3)"
)

parser.add_argument(
    "--gpu", default=0, type=int, help="softmax temperature (default: 0.3)"
)

args = parser.parse_args()

if args.gpu == 7:
    gpu = 3
else:
    gpu = args.gpu

# kl_weight = 1
# lr = 0.1
# dro_lr = 0.0001
# temperature = 0.07

wandb.init(project="MLOpt", config=args)


lincls.main(
    os.path.join(
        "./saved_models/",
        train.main(args.t, args.lr, args.kl_weight, gpu),
        "checkpoint_0399.pth.tar",
    ),
    gpu,
)

# lr_str = str(lr).split(".")

# os.system(
#     f"python train.py --gpu {gpu} --t {temperature} --lr {lr} --dro_lr {dro_lr} --kl_weight {kl_weight} --p_method Linear --sim l2"
# )
# os.system(
#     f"python lincls.py --pretrained ./saved_models/20221013_cifar10_resnet50_sogclr-128-2048_bz_64_E200_WR10_lr_0.250_linear_wd_0.0001_t_0.3_g_0.9_lars_Linear_lr_0_0001_KL_1_0/checkpoint_0170.pth.tar --gpu {gpu}"
# )

# 20221013_cifar10_resnet50_sogclr-128-2048_bz_64_E200_WR10_lr_0.250_linear_wd_0.0001_t_0.3_g_0.9_lars_Linear_lr_0_0001_KL_1_0
# 20221013_cifar10_resnet50_sogclr-128-2048_bz_64_E200_WR10_lr_0.025_linear_wd_0.0001_t_1.0_g_0.9_lars_Linear_lr_0_0001_KL_1_0
# 20221013_cifar10_resnet50_sogclr-128-2048_bz_64_E200_WR10_lr_0.250_linear_wd_0.0001_t_0.3_g_0.9_lars_Linear_lr_0_0001_KL_1_
