import argparse
import random

class Parser():
    default_options = dict(
        mnist=dict(batch_size=200,
                   num_epochs=500,
                   model='lenet',
                   eps=76.5,
                   eps_iter=2.55,
                   num_steps=40,
                   clip_min=0.0,
                   clip_max=1.0,
                   noise_std=0.5),

        cifar10=dict(batch_size=128,
                     num_epochs=100,
                     model='resnet',
                     eps=16.0,
                     eps_iter=2.0,
                     num_steps=10,
                     clip_min=0.0,
                     clip_max=1.0,
                     noise_std=0.06)
        )

    def __init__(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        # device
        parser.add_argument('--gpu_id', type=int, default=0, help='id of gpu')
        parser.add_argument('--cuda', action='store_true', help='use cuda')

        # data
        parser.add_argument('--data_root', type=str, required=True, help='root of dataset')
        parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10'],
                            help='dataset name')
        parser.add_argument('--batch_size', type=int, help='batch size')
        parser.add_argument('--noise', action='store_true', help='gaussian noise')
        parser.add_argument('--noise_std', type=float, help='std of gaussian')
        parser.add_argument('--val_samples', type=int, default=1000, help='number of samples for validation')

        # model
        parser.add_argument('--model', type=str, choices=['lenet', 'resnet'], help='model name')
        parser.add_argument('--weight_init', type=str, default='he', help='method of weight initialization')

        # loss
        parser.add_argument('--ct', action='store_true', help='use clean example training')
        parser.add_argument('--ct_lambda', type=float, default=0.0, help='coef for clean example training')
        parser.add_argument('--at', action='store_true', help='use adversarial example training')
        parser.add_argument('--at_lambda', type=float, default=0.0, help='coef of adversarial example training')
        parser.add_argument('--alp', action='store_true', help='use adversarial logit pairing')
        parser.add_argument('--alp_lambda', type=float, default=0.0, help='coef of adversarial logit pairing')
        parser.add_argument('--clp', action='store_true', help='use clean logit pairing')
        parser.add_argument('--clp_lambda', type=float, default=0.0, help='coef of clean logit pairing')
        parser.add_argument('--lsq', action='store_true', help='use logit squeezing')
        parser.add_argument('--lsq_lambda', type=float, default=0.0, help='coef of logit squeezing')
        parser.add_argument('--lsq_lambda_grad', action='store_true', help='use gradual coef for logit squeezing')
        parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

        # attack
        parser.add_argument('--attacker', type=str, default='pgd', help='name of adversarial attack method')
        parser.add_argument('--eps', type=float, help='epsilon for lp-norm attack')
        parser.add_argument('--eps_iter', type=float, help='epsilon for each attack step')
        parser.add_argument('--num_steps', type=int, help='number of steps for attack')
        parser.add_argument('--clip_min', type=float, help='minimum value for cliping AEs')
        parser.add_argument('--clip_max', type=float, help='miximum value for cliping AEs')

        # optimization
        parser.add_argument('--optimizer', type=str, default='Adam', help='name of optimization method')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--scheduler_step', type=int, default=0, help='step for lr-scheduler')
        parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='gamma for lr-scheduler')
        parser.add_argument('--num_epochs', type=int, help='number of epochs')

        # others
        parser.add_argument('--comet', action='store_true', help='use comet for training log')
        parser.add_argument('--print_freq', type=int, default=10, help='frequency of printing logs')
        parser.add_argument('--adv_val_freq', type=int, default=1, help='frequency of adversarial validation')
        parser.add_argument('--add_names', type=str, nargs='*', default=[], help='additional experiment name')
        parser.add_argument('--add_tags', type=str, nargs='*', default=[], help='additinal tags for comet')

        # debug
        parser.add_argument('--report_itr_loss', type=str, nargs='*', default=[], 
                            choices=['ct', 'at', 'alp', 'clp', 'lsq'],
                            help='report loss each iteration in the training phase')

        self.opt = parser.parse_args()

    def get(self):
        # Set None options in default_options
        for name, value in self.default_options[self.opt.dataset].items():
            if not getattr(self.opt, name):
                setattr(self.opt, name, value)


        # Add tags and set experiment name
        base_tags = ['{}e'.format(self.opt.num_epochs),
                     self.opt.dataset]
        for name in ['ct', 'at', 'alp', 'clp', 'lsq']:
            if getattr(self.opt, name):
                base_tags.append(name)
                self.opt.add_names.append(name)
        self.opt.add_tags.extend(base_tags)

        self.opt.add_names.append(str(random.randint(100, 999)))
        self.opt.exp_name = '_'.join(self.opt.add_names)

        return self.opt
