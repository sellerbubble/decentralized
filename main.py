import os
import copy
import torch
import socket
import datetime
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from datasets import load_dataset
from networks import load_model, load_valuemodel
from workers.worker_vision import Worker_Vision, Worker_Vision_AMP, DQNAgent
from utils.scheduler import Warmup_MultiStepLR
from utils.utils import (
    set_seed,
    add_identity,
    generate_P,
    save_model,
    evaluate_and_log,
    update_center_model,
    update_dsgd,
    update_dqn_chooseone,
    update_csgd,
)
from easydict import EasyDict
import wandb
from utils.dirichlet import dirichlet_split_noniid
from torchvision.datasets import CIFAR10
import numpy as np

from fire import Fire

# torch.set_num_threads(4)

nfs_dataset_path1 = "/mnt/nfs4-p1/ckx/datasets/"
nfs_dataset_path2 = "/nfs4-p1/ckx/datasets/"


def main(
    dataset_path="datasets",
    dataset_name="CIFAR10",
    image_size=56,
    batch_size=512,
    n_swap=None,
    mode="csgd",
    shuffle="fixed",
    size=16,
    port=29500,
    backend="gloo",
    model="ResNet18_M",
    pretrained=1,
    lr=0.1,
    wd=0.0,
    gamma=0.1,
    alpha=0.3,
    momentum=0.0,
    warmup_step=0,
    epoch=6000,
    early_stop=6000,
    milestones=[2400, 4800],
    seed=666,
    device=0,
    amp=False,
    sample=0,
    n_components=0,
    nonIID=True,
    project_name="decentralized",
):
    sub_dict_keys = [
        "dataset_name",
        "image_size",
        "batch_size",
        "mode",
        "model",
        "pretrained",
        "alpha",
        "epoch",
        "nonIID",
    ]
    args = EasyDict(locals().copy())
    # set_seed(args)
    set_seed(seed, torch.cuda.device_count())
    dir_path = os.path.dirname(__file__)
    args = add_identity(args, dir_path)
    sub_dict_str = "_".join([key + str(args[key]) for key in sub_dict_keys])
    # 登录Wandb账户
    wandb.login(key="831b4bf90cf69dcf8cae62953d13595412ce439d")

    # 初始化Wandb项目
    run = wandb.init(project=project_name)
    wandb.config.update(args)

    run.name = sub_dict_str
    run.save()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # check nfs dataset path

    log_id = (
        datetime.datetime.now().strftime("%b%d_%H:%M:%S")
        + "_"
        + socket.gethostname()
        + "_"
        + args.identity
    )
    writer = SummaryWriter(log_dir=os.path.join(args.runs_data_dir, log_id))

    probe_train_loader, probe_valid_loader, _, classes = load_dataset(
        root=args.dataset_path,
        name=args.dataset_name,
        image_size=args.image_size,
        train_batch_size=args.batch_size,
        valid_batch_size=args.batch_size,
    )
    worker_list = []
    if nonIID:
        train_set = CIFAR10(
            root=args.dataset_path, train=True, transform=None, download=True
        )
        label = np.array(train_set.targets)
        split = dirichlet_split_noniid(
            train_labels=label, alpha=args.alpha, n_clients=args.size
        )
    else:
        split = [
            1.0 / args.size for _ in range(args.size)
        ]  # split 是一个列表 代表每个model分的dataset

    for rank in range(args.size):
        train_loader, _, _, classes = load_dataset(
            root=args.dataset_path,
            name=args.dataset_name,
            image_size=args.image_size,
            train_batch_size=args.batch_size,
            distribute=True,
            rank=rank,
            split=split,
            seed=args.seed,
        )
        model = load_model(args.model, classes, pretrained=args.pretrained).to(
            args.device
        )
        if args.mode == "dqn_chooseone":
            value_model = load_valuemodel(1, 50, args.size)

        optimizer = SGD(
            model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum
        )
        # scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        scheduler = Warmup_MultiStepLR(
            optimizer,
            warmup_step=args.warmup_step,
            milestones=args.milestones,
            gamma=args.gamma,
        )
        # worker是训练器
        if args.amp:
            worker = Worker_Vision_AMP(
                model, rank, optimizer, scheduler, train_loader, args.device
            )
        else:
            if args.mode == "dqn_chooseone":
                worker = DQNAgent(
                    model, value_model, rank, optimizer, scheduler, train_loader, args
                )
            else:
                worker = Worker_Vision(
                    model, rank, optimizer, scheduler, train_loader, args.device
                )
        worker_list.append(worker)

    # 定义 中心模型 center_model
    center_model = copy.deepcopy(worker_list[0].model)
    for name, param in center_model.named_parameters():
        for worker in worker_list[1:]:
            param.data += worker.model.state_dict()[name].data
        param.data /= args.size

    P = generate_P(args.mode, args.size)

    for iteration in range(args.early_stop):
        epoch = iteration // len(train_loader)

        if iteration % len(train_loader) == 0:
            for worker in worker_list:
                worker.update_iter()

        if args.mode == "csgd":
            update_csgd(worker_list, center_model)
        elif args.mode == "dqn_chooseone":
            update_dqn_chooseone(worker_list)
        else:  # dsgd
            update_dsgd(worker_list, P, args)

        center_model = update_center_model(worker_list)

        if iteration % 50 == 0:
            train_acc, train_loss, valid_acc, valid_loss = evaluate_and_log(
                center_model,
                probe_train_loader,
                probe_valid_loader,
                iteration,
                epoch,
                writer,
                args,
                wandb,
            )

        if iteration == args.early_stop - 1:
            save_model(center_model, train_acc, epoch, args, log_id)
            break

    writer.close()
    print("Ending")


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # ## dataset
    # parser.add_argument("--dataset_path", type=str, default='datasets')
    # parser.add_argument("--dataset_name", type=str, default='CIFAR10',
    #                                         choices=['CIFAR10','CIFAR100','TinyImageNet'])
    # parser.add_argument("--image_size", type=int, default=56, help='input image size')
    # parser.add_argument("--batch_size", type=int, default=512)
    # parser.add_argument('--n_swap', type=int, default=None)

    # # mode parameter
    # parser.add_argument('--mode', type=str, default='csgd', choices=['csgd', 'ring', 'meshgrid', 'exponential', 'dqn_chooseone'])
    # parser.add_argument('--shuffle', type=str, default="fixed", choices=['fixed', 'random'])
    # parser.add_argument('--size', type=int, default=16)
    # parser.add_argument('--port', type=int, default=29500)
    # parser.add_argument('--backend', type=str, default="gloo")
    # # deep model parameter
    # parser.add_argument('--model', type=str, default='ResNet18_M',
    #                     choices=['ResNet18', 'AlexNet', 'DenseNet121', 'AlexNet_M','ResNet18_M', 'ResNet34_M', 'DenseNet121_M'])
    # parser.add_argument("--pretrained", type=int, default=1)

    # # optimization parameter
    # parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    # parser.add_argument('--wd', type=float, default=0.0,  help='weight decay')
    # parser.add_argument('--gamma', type=float, default=0.1)
    # parser.add_argument('--momentum', type=float, default=0.0)
    # parser.add_argument('--warmup_step', type=int, default=0)
    # parser.add_argument('--epoch', type=int, default=6000)
    # parser.add_argument('--early_stop', type=int, default=6000, help='w.r.t., iterations')
    # parser.add_argument('--milestones', type=int, nargs='+', default=[2400, 4800])
    # parser.add_argument('--seed', type=int, default=666)
    # parser.add_argument("--device", type=int, default=0)
    # parser.add_argument("--amp", action='store_true', help='automatic mixed precision')

    # args = parser.parse_args()

    # args = add_identity(args, dir_path)
    # # print(args)
    # main(args)
    if os.path.exists(nfs_dataset_path1):
        dataset_path = nfs_dataset_path1
    elif os.path.exists(nfs_dataset_path2):
        dataset_path = nfs_dataset_path2
    Fire(main)
