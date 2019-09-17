import sys
import torch
import torch.nn as nn
import torch.optim as optim

from advertorch.attacks import LinfPGDAttack

from utils import get_options, report_epoch_status, Timer, init_he
from dataset import load_dataset
from trainer import Trainer
from model import LeNet


def main():
    opt = get_options()

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
    trainer = Trainer(opt, device, model, criterion, attacker, optimizer)
    
    # epoch iteration
    for epoch in range(1, opt.num_epochs+1):
        if scheduler:
            scheduler.step(epoch - 1) # scheduler's epoch is 0-indexed.

        train_losses, train_acc1s, train_acc5s = trainer.train(train_loader)
        val_losses, val_acc1s, val_acc5s = trainer.validate(val_loader)

        losses = dict(**train_losses, **val_losses)
        acc1s = dict(**train_acc1s, **val_acc1s)
        acc5s = dict(**train_acc5s, **val_acc5s)
        report_epoch_status(losses, acc1s, acc5s, trainer.num_loss,
                            epoch, opt.num_epochs, opt, timer)

    save_path = os.path.join('ckpt', 'models', opt.save_name + 'pth')
    trainer.save_model(save_path)

if __name__ == '__main__':
    main()
