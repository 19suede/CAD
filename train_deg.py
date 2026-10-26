import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import importlib
import numpy as np
from tqdm import tqdm
from dataset import PansharpeningDataset
from utils import create_logger, seed_all, crop_patches, count_param
from init_config import init_config

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.cuda.empty_cache()

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='IKONOS')
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--logger_path', default='./log/log_deg_')
    parser.add_argument('--save_checkpoint_path', default='checkpoints', type=str)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--deg-model', default='model_deg_ab1', type=str,
                        help='Degradation model module, e.g., model_deg_ab1')
    parser.add_argument('--suffix', default='ab1', type=str,
                        help='Optional suffix for checkpoint/log naming, e.g., ab1')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow overwriting existing checkpoint file')

    args = parser.parse_args()
    return args

def train_one_epoch(G, dataloader, opt, sch, logger, epoch, args):
    G.train()
    total_loss = 0
    spa_loss = 0
    spe_loss = 0

    for batch in tqdm(dataloader):
        pan = batch['pan'].to(args.device)
        ms = batch['ms'].to(args.device)

        loss, spa_l, spe_l, balance = G(pan, ms)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        spa_loss += spa_l.item()
        spe_loss += spe_l.item()

    logger.info(f"Epoch {epoch}: Total_loss={total_loss/len(dataloader):.4f} | spa_loss={spa_loss/len(dataloader):.4f} | spe_loss={spe_loss/len(dataloader):.4f}")

    sch.step()

def train(args):
    seed_all(42)
    suffix = f"_{args.suffix}" if args.suffix else ""
    args.logger_path = args.logger_path + args.dataset + suffix + ".txt"
    logger = create_logger(args)
    config = init_config(f"./config_{args.dataset}.yaml")

    print(args.device)
    train_dataset = PansharpeningDataset(config.train_path, norm=True, dr=config.dr)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    deg_lib = importlib.import_module(args.deg_model)
    DegClass = getattr(deg_lib, 'Degradation')
    Deg = DegClass(config.lms_dim)
    Deg.to(args.device)

    opt = torch.optim.Adam(Deg.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=1.0)
    count_param(Deg)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(Deg, dataloader, opt, sch, logger, epoch, args)

        if epoch % 10 == 0:
            model_name = f"Deg{suffix}"
            save_checkpoint(model=Deg, epoch=epoch, args=args, model_name=model_name)

def save_checkpoint(model, epoch, args, optimizer=None, scheduler=None, loss=None, model_name=None):
    path = os.path.join(args.save_checkpoint_path, args.dataset)
    os.makedirs(path, exist_ok=True)
    state_dict = {}
    state_dict["model_state_dict"] = model.state_dict()
    state_dict["epoch"] = epoch
    if optimizer:
        state_dict["optimizer_state_dict"] =  optimizer.state_dict()
    if scheduler:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()
    if loss:
        state_dict["loss"] = loss

    if model_name:
        output_path = os.path.join(path, f"{model_name}_epoch{epoch}.pth")
        if os.path.exists(output_path) and not args.overwrite:
            raise FileExistsError(f"Checkpoint already exists: {output_path}")
        torch.save(state_dict, output_path)
    else:
        output_path = os.path.join(path, f"epoch{epoch}.pth")
        if os.path.exists(output_path) and not args.overwrite:
            raise FileExistsError(f"Checkpoint already exists: {output_path}")
        torch.save(state_dict, output_path)

if __name__ == '__main__':
    args = set_args()
    train(args)
