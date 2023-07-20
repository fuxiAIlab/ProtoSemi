def parse_args(parser):
    """
    Dataset configs
    """
    parser.add_argument(
        "--dataset_dir", type=str, default="../data", help="path to dataset directory"
    )
    parser.add_argument(
        "--result_dir", type=str, default="./results", help="path to save directory"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="path to checkpoint directory",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="path to log directory"
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="./tensorboards",
        help="path to tensorboard directory",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="clean",
        help="clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100",
    )
    parser.add_argument(
        "--noise_path",
        type=str,
        default=None,
        help="paths of CIFAR-10_human.pt and CIFAR-100_human.pt",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="cifar10 or cifar100",
        choices=["cifar10", "cifar100"],
    )
    parser.add_argument(
        "--is_human",
        action="store_true",
        default=False,
        help="whether to use human-annotated data or symmetrized data",
    )
    """
    Training configs
    """
    parser.add_argument(
        "--train_batch_size", type=int, default=128, help="train batch size"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=64, help="test batch size"
    )
    parser.add_argument(
        "--proto_batch_size",
        type=int,
        default=200,
        help="batch size in prototype embedding generation",
    )
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--warmups", type=int, default=None, help="number of warm up epochs"
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", help="optimizer", choices=["sgd"]
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--scheduler", type=str, default="cos", help="the scheduler for learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="weight decay"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="how many subprocesses to use for data loading",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=50,
        help="print frequency in the training process",
    )
    """
    Method configs
    """
    parser.add_argument(
        "--sample_split",
        type=str,
        default=None,
        help="method to divide clean and noisy samples",
        choices=["pes", "proto"],
    )
    parser.add_argument(
        "--ssl",
        type=str,
        default=None,
        help="semi-supervised learning method to enhance",
        choices=["mixmatch"],
    )
    parser.add_argument(
        "--mixmatch_alpha",
        type=float,
        default=4,
        help="the beta parameter in mixmatch training",
    )
    parser.add_argument(
        "--mixmatch_t",
        type=float,
        default=0.5,
        help="the temperature parameter in mixmatch training",
    )
    parser.add_argument(
        "--mixmatch_lambda_u",
        type=float,
        default=5.0,
        help="weight for unsupervised loss in mixmatch training",
    )
    parser.add_argument(
        "--cos_up_bound",
        type=float,
        default=0.9,
        help="hyperparameter to select the samples that may belong to this type",
    )
    parser.add_argument(
        "--cos_low_bound",
        type=float,
        default=0.3,
        help="hyperparameter to filter the samples that must not belong to this type",
    )
    parser.add_argument(
        "--proto_epochs",
        type=int,
        default=1,
        help="the epochs to adjust in protosplit",
    )

    args = parser.parse_args()
    return args
