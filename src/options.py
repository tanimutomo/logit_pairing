import argparse


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
    parser.add_argument('--comet', action='store_true')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--adv_val_freq', type=int, default=5)
    parser.add_argument('--exp_name', type=str, required=True)

    opt = parser.parse_args()
    return opt


