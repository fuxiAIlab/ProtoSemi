import torch
import torch.nn.functional as F
import numpy as np
import logging

from myutils import AverageMeter

logger = logging.getLogger(__name__)


def linear_rampup(current, lambda_u, warm_up, rampup_length):
    # reference from PES
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u * float(current)


def mixmatch_train(
    epoch,
    model,
    optimizer,
    labeled_trainloader,
    unlabeled_trainloader,
    class_weights,
    args,
):
    # reference from PES
    if epoch >= args.epochs / 2:
        args.mixmatch_alpha = 0.75

    losses = AverageMeter("Loss", ":6.2f")
    losses_lx = AverageMeter("Loss_Lx", ":6.2f")
    losses_lu = AverageMeter("Loss_lu", ":6.5f")

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = int(50000 / args.train_batch_size)
    for batch_idx in range(num_iter):
        try:
            inputs_x1, inputs_x2, targets_x = labeled_train_iter.next()
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x1, inputs_x2, targets_x = labeled_train_iter.next()

        try:
            inputs_u1, inputs_u2 = unlabeled_train_iter.next()
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u1, inputs_u2 = unlabeled_train_iter.next()

        batch_size = inputs_x1.size(0)
        targets_x = torch.zeros(batch_size, args.num_classes).scatter_(
            1, targets_x.view(-1, 1), 1
        )
        inputs_x1, inputs_x2, targets_x = (
            inputs_x1.cuda(),
            inputs_x2.cuda(),
            targets_x.cuda(),
        )
        inputs_u1, inputs_u2 = inputs_u1.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            outputs_u1, _ = model(inputs_u1)
            outputs_u2, _ = model(inputs_u2)
            prob_u = (
                torch.softmax(outputs_u1, dim=1) + torch.softmax(outputs_u2, dim=1)
            ) / 2

            # add temperature to sharpen
            prob_u_temp = prob_u ** (1 / args.mixmatch_t)

            targets_u = prob_u_temp / prob_u_temp.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        all_inputs = torch.cat([inputs_x1, inputs_x2, inputs_u1, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        rand_idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[rand_idx]
        target_a, target_b = all_targets, all_targets[rand_idx]

        mixmatch_l = np.random.beta(args.mixmatch_alpha, args.mixmatch_alpha)
        mixmatch_l = max(mixmatch_l, 1 - mixmatch_l)

        mixed_input = mixmatch_l * input_a + (1 - mixmatch_l) * input_b
        mixed_target = mixmatch_l * target_a + (1 - mixmatch_l) * target_b

        logits, _ = model(mixed_input)
        logits_x = logits[: batch_size * 2]
        logits_u = logits[batch_size * 2 :]

        Lx_mean = -torch.mean(
            F.log_softmax(logits_x, dim=1) * mixed_target[: batch_size * 2],
            0,
        )
        Lx = torch.sum(Lx_mean * class_weights)

        probs_u = torch.softmax(logits_u, dim=1)
        Lu = torch.mean((probs_u - mixed_target[batch_size * 2 :]) ** 2)
        loss = (
            Lx
            + linear_rampup(
                current=epoch + batch_idx / num_iter,
                lambda_u=args.mixmatch_lambda_u,
                warm_up=args.warmups,
                rampup_length=16,
            )
            * Lu
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_lx.update(Lx.item(), batch_size * 2)
        losses_lu.update(Lu.item(), len(logits) - batch_size * 2)
        losses.update(loss.item(), len(logits))

        # logger.info("{}, {}, {}".format(losses, losses_lx, losses_lu))
    return losses.avg, losses_lx.avg, losses_lu.avg
