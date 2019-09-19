import argparse


def get_options():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # device
    parser.add_argument('--gpu_id', type=int, default=0, help='id of gpu')
    parser.add_argument('--cuda', action='store_true', help='use cuda')

    # data
    parser.add_argument('--data_root', type=str, required=True, help='root of dataset')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--noise', action='store_true', help='gaussian noise')
    parser.add_argument('--noise_std', type=float, default=0.5, help='std of gaussian')
    parser.add_argument('--val_samples', type=int, default=1000, help='number of samples for validation')

    # model
    parser.add_argument('--model', type=str, default='lenet', help='model name')
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
    parser.add_argument('--eps', type=float, default=0.3, help='epsilon for lp-norm attack')
    parser.add_argument('--eps_iter', type=float, default=0.01, help='epsilon for each attack step')
    parser.add_argument('--num_steps', type=int, default=40, help='number of steps for attack')

    # optimization
    parser.add_argument('--optimizer', type=str, default='Adam', help='name of optimization method')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--scheduler_step', type=int, default=0, help='step for lr-scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='gamma for lr-scheduler')
    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs')

    # others
    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--comet', action='store_true', help='use comet for training log')
    parser.add_argument('--print_freq', type=int, default=10, help='frequency of printing logs')
    parser.add_argument('--adv_val_freq', type=int, default=5, help='frequency of adversarial validation')

    # debug
    parser.add_argument('--report_itr_losses', type=str, nargs='*', 
                        choices=['ct', 'at', 'alp', 'clp', 'lsq'],
                        help='report loss each iteration in the training phase')

    opt = parser.parse_args()
    return opt


