import argparse
import sys
import time
import datetime
import torch.nn as nn


def get_options():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # device
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--cuda', action='store_true')
    # data
    parser.add_argument('--data_root', type=str, required=True, help='root of dataset')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--noise', action='store_true', help='gaussian noise')
    parser.add_argument('--noise_std', type=float, default=0.5)
    parser.add_argument('--val_samples', type=int, default=1000)
    # model
    parser.add_argument('--model', type=str, default='lenet')
    parser.add_argument('--weight_init', type=str, default='he')
    # loss
    parser.add_argument('--ct', action='store_true')
    parser.add_argument('--ct_lambda', type=float, default=0.0)
    parser.add_argument('--at', action='store_true')
    parser.add_argument('--at_lambda', type=float, default=0.0)
    parser.add_argument('--alp', action='store_true')
    parser.add_argument('--alp_lambda', type=float, default=0.0)
    parser.add_argument('--clp', action='store_true')
    parser.add_argument('--clp_lambda', type=float, default=0.0)
    parser.add_argument('--lsq', action='store_true')
    parser.add_argument('--lsq_lambda', type=float, default=0.0)
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    # attack
    parser.add_argument('--attacker', type=str, default='pgd')
    parser.add_argument('--eps', type=float, default=0.3)
    parser.add_argument('--eps_iter', type=float, default=0.01)
    parser.add_argument('--num_steps', type=int, default=40)
    # optimization
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler_step', type=int, default=0)
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=30)
    # others
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--adv_val_freq', type=int, default=5)
    parser.add_argument('--save_name', type=str, required=True)

    opt = parser.parse_args()
    return opt


def init_he(m):
    if type(m) in [nn.Linear or nn.Conv2d]:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)


def report_epoch_status(losses, acc1s, acc5s, num_loss,
                        epoch, epochs, opt, timer):
    log = '\r\033[{}A\033[J'.format(num_loss+1) \
          + 'epoch [{:d}/{:d}]'.format(epoch, epochs)

    # loss
    log += '\n[loss] '
    for flag, name in zip([opt.ct, opt.at, opt.alp, opt.clp, opt.lsq,
                           True, True, True],
                          ['ct', 'at', 'alp', 'clp', 'lsq',
                           'total', 'val', 'aval']):
        if flag:
            log += '{} {:.4f} / '.format(name, losses[name].avg)

    # acc1 log
    log += '\n[acc1] '
    for flag, name in zip([opt.ct, opt.at, True, True],
                          ['ct', 'at', 'val', 'aval']):
        if flag:
            log += '{} {:.2f}% / '.format(name, acc1s[name].avg)

    # acc5 log
    log += '\n[acc5] '
    for flag, name in zip([opt.ct, opt.at, True, True],
                          ['ct', 'at', 'val', 'aval']):
        if flag:
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

