import logging
import argparse
import time
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")

from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from myutils import get_elapse_time, accuracy, get_file_time
from models import ResNet34
from data_preprocess.datasets import (
    input_dataset,
    train_cifar100_transform,
    train_cifar10_transform,
)
from configs import parse_args


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def bulid_model(num_classes):
    model = ResNet34(num_classes=num_classes)
    return model


def train(train_dataset, train_loader, model, optimizer, epoch, args):
    train_total = 0
    train_correct = 0

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        batch_size = len(ind)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        logits, _ = model(images)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec
        loss = F.cross_entropy(logits, labels, reduce=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = float(train_correct) / float(train_total)
    return train_acc, loss


def evaluate(model, test_loader):
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits, _ = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100 * float(correct) / float(total)
    return acc


def main():
    parser = argparse.ArgumentParser()
    args = parse_args(parser=parser)
    t0 = time.time()
    file_time = get_file_time(t0, args)

    # read noise files
    if args.noise_path is None:
        if args.dataset == "cifar10":
            args.noise_path = "../data/CIFAR-N/CIFAR-10_human.pt"
            train_transform = train_cifar10_transform
            if args.warmups is None:
                args.warmups = 10
            args.mixmatch_lambda_u = 5
        elif args.dataset == "cifar100":
            args.noise_path = "../data/CIFAR-N/CIFAR-100_human.pt"
            train_transform = train_cifar100_transform
            if args.warmups is None:
                args.warmups = 35
            args.mixmatch_lambda_u = 75
        else:
            raise NameError(f"Undefined dataset {args.dataset}")

    noise_type_map = {
        "clean": "clean_label",
        "worst": "worse_label",
        "aggre": "aggre_label",
        "rand1": "random_label1",
        "rand2": "random_label2",
        "rand3": "random_label3",
        "clean100": "clean_label",
        "noisy100": "noisy_label",
    }
    args.noise_type = noise_type_map[args.noise_type]

    cifar_n_label = torch.load(args.noise_path)
    clean_labels = cifar_n_label["clean_label"]
    noisy_labels = cifar_n_label[args.noise_type]

    # Run relevant files configuration
    if not os.path.exists(args.log_dir):
        os.path.mkdir(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.path.mkdir(args.result_dir)
    if not os.path.exists(args.tensorboard_dir):
        os.path.mkdir(args.tensorboard_dir)

    result_file = open(
        os.path.join(
            args.result_dir,
            "{}_{}_{}_{}_{}_{}.txt".format(
                args.dataset,
                args.noise_type,
                args.sample_split,
                args.ssl,
                args.cos_up_bound,
                args.cos_low_bound,
            ),
        ),
        "a+",
    )
    result_file.writelines(str(args) + "\n")

    log_file_name = os.path.join(args.log_dir, "{}.txt".format(file_time))
    handler = logging.FileHandler(log_file_name)
    logger.addHandler(handler)

    tb_writer = SummaryWriter(
        "{}/{}".format(
            args.tensorboard_dir,
            "{}_{}_{}_{}_{}_{}".format(
                args.dataset,
                args.noise_type,
                args.sample_split,
                args.ssl,
                args.cos_up_bound,
                args.cos_low_bound,
            ),
        )
    )

    # Start training
    logger.info(args)
    logger.info("Preparing data...")
    train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        noise_type=args.noise_type,
        noise_path=args.noise_path,
        is_human=args.is_human,
    )
    args.num_classes = num_classes
    logging.info("Train labels examples:{}".format(train_dataset.train_labels[:20]))
    logging.info("Building model...")
    model = bulid_model(num_classes=num_classes)
    model.cuda()
    # if args.optimizer == "sgd":
    #     alpha_plan = [0.1] * 60 + [0.01] * 40
    #     optimizer = torch.optim.SGD(
    #         model.parameters(),
    #         lr=args.lr,
    #         weight_decay=args.weight_decay,
    #         momentum=args.momentum,
    #     )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if args.scheduler == "cos":
        scheduler = CosineAnnealingLR(optimizer, args.epochs, args.lr / 100)
    elif args.scheduler == "multistep":
        scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    if args.sample_split == "pes":
        from sample_splits import pes_split as sample_split
    elif args.sample_split == "proto":
        from sample_splits import proto_split as sample_split

    if args.ssl == "mixmatch":
        from myssl import mixmatch_train as ssl_train

    logger.info("Start Training...")
    best_test_acc, best_test_epoch = 0, 0

    for epoch in range(args.epochs):
        args.epoch_now = epoch
        if epoch < args.warmups:
            if epoch == 0:
                logger.info("Start Warmup...")
            model.train()
            train_acc, train_loss = train(
                train_dataset, train_loader, model, optimizer, epoch, args
            )
            tb_writer.add_scalar("train_acc", train_acc, epoch)
            tb_writer.add_scalar("train_loss", train_loss, epoch)
            logger.info(
                "Epoch: {}, train accuracy: {:.2f}%, train loss: {:.2f}".format(
                    epoch, train_acc, train_loss
                )
            )
        else:
            if epoch == args.warmups:
                logger.info("Start Semi-supervised Learning...")
            model.eval()
            labeled_trainloader, unlabeled_trainloader, class_weights = sample_split(
                model=model,
                train_data=train_dataset.train_data,
                transform_train=train_transform,
                clean_targets=clean_labels,
                noisy_targets=noisy_labels,
                args=args,
            )
            model.train()
            losses, losses_lx, losses_lu = ssl_train(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                labeled_trainloader=labeled_trainloader,
                unlabeled_trainloader=unlabeled_trainloader,
                class_weights=class_weights,
                args=args,
            )
            tb_writer.add_scalar("losses", losses, epoch)
            tb_writer.add_scalar("losses_lx", losses_lx, epoch)
            tb_writer.add_scalar("losses_lu", losses_lu, epoch)
            logger.info(
                "Epoch: {}, losses: {:.4f}, losses_lx: {:.4f}, losses_lu: {:.6f}".format(
                    epoch, losses, losses_lx, losses_lu
                )
            )

        model.eval()
        test_acc = evaluate(test_loader=test_loader, model=model)
        logger.info("Epoch: {}, test accuracy: {:.2f}".format(epoch, test_acc))
        tb_writer.add_scalar("test_acc", test_acc, epoch)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_epoch = epoch
        scheduler.step()
    tb_writer.close()
    logger.info(
        "Best test accuracy: {:.2f}, best test epoch: {}".format(
            best_test_acc, best_test_epoch
        )
    )
    logger.info("Finsh and take {}".format(get_elapse_time(t0)))
    result_file.write(
        "Best test accuracy: {:.2f}, best test epoch: {}\n".format(
            best_test_acc, best_test_epoch
        )
    )
    result_file.write("Finsh and take {}\n".format(get_elapse_time(t0)))
    result_file.write("\n")


main()
