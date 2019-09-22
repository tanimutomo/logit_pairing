import argparse
import sys
import time
import datetime
import torch.nn as nn


def init_he(m):
    if type(m) in [nn.Linear or nn.Conv2d]:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)


def report_epoch_status(losses, acc1s, acc5s, num_loss,
                        epoch, opt, timer, experiment):
    opt.total = True
    opt.val = True
    opt.aval = opt.adv_val_freq != -1 and epoch % opt.adv_val_freq == 0
    log = '\r\033[{}A\033[J'.format(num_loss+2) \
          + 'epoch [{:d}/{:d}]'.format(epoch, opt.num_epochs)

    # loss
    log += '\n[loss] '
    for name in ['ct', 'at', 'alp', 'clp', 'lsq',
                 'total', 'val', 'aval']:
        if getattr(opt, name):
            log += '{} {:.4f} / '.format(name, losses[name].avg)
            experiment.log_metric(name + '-loss',
                                  losses[name].avg)

    # acc1 log
    log += '\n[acc1] '
    for name in ['ct', 'at', 'val', 'aval']:
        if getattr(opt, name):
            log += '{} {:.2f}% / '.format(name, acc1s[name].avg)
            experiment.log_metric(name + '-acc1',
                                  acc1s[name].avg)

    # acc5 log
    log += '\n[acc5] '
    for name in ['ct', 'at', 'val', 'aval']:
        if getattr(opt, name):
            log += '{} {:.2f}% / '.format(name, acc5s[name].avg)

    timer.step()

    # time log
    log += '\n[time] elapsed: {} / '.format(timer.get_elapsed_time()) \
                 + 'estimated: {}\n'.format(timer.get_estimated_time())

    sys.stdout.write(log)
    sys.stdout.flush()


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.cnt = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt


class Timer():
	def __init__(self, num_steps, start_step=0):
		self.num_steps = num_steps
		self.start_step = start_step
		self.current_step = start_step
		self.start_time = time.time()
		self.elapsed_time = time.time()

	def step(self):
		self.current_step += 1

	def set_current_step(self, step):
		self.current_step = step

	def get_elapsed_time(self):
		self.elapsed_time = time.time() - self.start_time
		return str(datetime.timedelta(seconds=int(self.elapsed_time)))

	def get_estimated_time(self):
		self.elapsed_time = time.time() - self.start_time
		remaining_step = self.num_steps - self.current_step

		if self.current_step == self.start_step:
			return str(datatime.timedelta(seconds=int(0)))
		estimated_time = self.elapsed_time * remaining_step / (self.current_step - self.start_step)
		return str(datetime.timedelta(seconds=int(estimated_time)))

