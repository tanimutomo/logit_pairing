import os
import sys
import torch
import torch.nn as nn

from advertorch.attacks import LinfPGDAttack 
from options import Parser
from utils import Timer
from dataset import load_dataset
from trainer import Trainer
from models import LeNet, ResNetv2_20


def main():
    opt = Parser(train=False).get()
    
    # dataset and data loader
    _, val_loader, adv_val_loader, _, num_classes = \
            load_dataset(opt.dataset, opt.batch_size, opt.data_root,
                         False, 0.0, opt.num_val_samples,
                         workers=4)

    # model
    if opt.arch == 'lenet':
        model = LeNet(num_classes)
    elif opt.arch == 'resnet':
        model = ResNetv2_20(num_classes)
    else:
        raise NotImplementedError

    # move model to device
    model.to(opt.device)

    # load trained weight
    model.load_state_dict(torch.load(opt.weight_path))

    # criterion
    criterion = nn.CrossEntropyLoss()

    # advertorch attacker
    if opt.attack == 'pgd':
        attacker = LinfPGDAttack(
            model, loss_fn = criterion, eps=opt.eps/255,
            nb_iter=opt.num_steps, eps_iter=opt.eps_iter/255,
            rand_init=True, clip_min=opt.clip_min, 
            clip_max=opt.clip_max, targeted=False
            )
    else:
        raise NotImplementedError

    # timer
    timer = Timer(opt.num_epochs, 0)

    # trainer
    trainer = Trainer(opt, model, criterion, attacker)
    
    # validation
    val_losses, val_acc1s, val_acc5s = \
        trainer.validate(val_loader)
    aval_losses, aval_acc1s, aval_acc5s = \
        trainer.adv_validate(adv_val_loader)


if __name__ == '__main__':
    main()
