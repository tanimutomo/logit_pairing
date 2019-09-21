from comet_ml import Experiment

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from advertorch.attacks import LinfPGDAttack

from options import get_options
from utils import report_epoch_status, Timer, init_he
from dataset import load_dataset
from trainer import Trainer
from model import LeNet


def main():
    opt = get_options()
    
    if opt.comet:
        experiment = Experiment()
        experiment.set_name(opt.exp_name)
        experiment.log_parameters(opt.__dict__)
        experiment.add_tag('{}e'.format(opt.num_epochs))
        experiment.add_tags(opt.add_tags)
        for flag, name in zip(
                [opt.ct, opt.at, opt.alp, opt.clp, opt.lsq],
                ['ct', 'at', 'alp', 'clp', 'lsq']):
            if flag:
                experiment.add_tag(name)
    else:
        experiment = None

    # device
    device = torch.device('cuda:{}'.format(opt.gpu_id)
                          if torch.cuda.is_available() and opt.cuda
                          else 'cpu')

    # dataset and data loader
    train_loader, val_loader, adv_val_loader, _, num_classes = \
            load_dataset(opt.dataset, opt.batch_size, opt.data_root,
                         opt.noise, opt.noise_std, opt.val_samples,
                         workers=4)

    # model
    if opt.model == 'lenet':
        model = LeNet(num_classes).to(device)
    else:
        raise NotImplementedError

    # weight init
    if opt.weight_init == 'he':
        model.apply(init_he)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # advertorch attacker
    if opt.attacker == 'pgd':
        attacker = LinfPGDAttack(
            model, loss_fn = criterion, eps=opt.eps,
            nb_iter=opt.num_steps, eps_iter=opt.eps_iter,
            rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False
        )
    else:
        raise NotImplementedError

    # optimizer
    if opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), opt.lr,
                               eps=1e-6, weight_decay=opt.wd)
    else:
        raise NotImplementedError

    # scheduler
    if opt.scheduler_step:
        scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=opt.scheduler_step,
                gamma=opt.scheduler_gamma)
    else:
        scheduler = None

    # timer
    timer = Timer(opt.num_epochs, 0)

    # trainer
    trainer = Trainer(opt, device, model, criterion, attacker, optimizer,
                      experiment)
    
    # epoch iteration
    for epoch in range(1, opt.num_epochs+1):
        experiment.set_epoch(epoch)
        if scheduler:
            scheduler.step(epoch - 1) # scheduler's epoch is 0-indexed.

        # training
        train_losses, train_acc1s, train_acc5s = \
                trainer.train(train_loader)

        # validation
        val_losses, val_acc1s, val_acc5s = \
                trainer.validate(val_loader)
        if opt.adv_val_freq != -1 and epoch % opt.adv_val_freq == 0:
            aval_losses, aval_acc1s, aval_acc5s = \
                trainer.adv_validate(adv_val_loader)
        else:
            aval_losses, aval_acc1s, aval_acc5s = \
                    dict(), dict(), dict()

        losses = dict(**train_losses, **val_losses, **aval_losses)
        acc1s = dict(**train_acc1s, **val_acc1s, **aval_acc1s)
        acc5s = dict(**train_acc5s, **val_acc5s, **aval_acc5s)
        report_epoch_status(losses, acc1s, acc5s, trainer.num_loss,
                            epoch, opt, timer, experiment)

    save_path = os.path.join('ckpt', 'models', opt.exp_name + 'pth')
    trainer.save_model(save_path)

if __name__ == '__main__':
    main()
