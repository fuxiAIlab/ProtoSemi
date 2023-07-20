import torch
import torch.nn.functional as F
import numpy as np
import PIL.Image as Image
import logging
import torchvision.transforms as transforms

# import sys

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

logger = logging.getLogger(__name__)


class Semi_Unlabeled_Dataset(Dataset):
    # reference from PES
    def __init__(self, data, transform=None):
        self.train_data = np.array(data)
        self.length = self.train_data.shape[0]

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        img = self.train_data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            # 使用两个随机转换进行联合训练
            out1 = self.transform(img)
            out2 = self.transform(img)

        return out1, out2

    def __len__(self):
        return self.length

    def getData(self):
        return self.train_data


class Semi_Labeled_Dataset(Dataset):
    # reference from PES
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.train_data = np.array(data)
        self.train_labels = np.array(labels)
        self.length = len(self.train_labels)
        self.target_transform = target_transform

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            out1 = self.transform(img)
            out2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return out1, out2, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.train_data, self.train_labels


def split_confident(outputs, clean_targets, noisy_targets):
    # reference from PES
    probs, preds = torch.max(outputs.data, 1)
    confident_correct_num = 0
    unconfident_correct_num = 0
    confident_indexs = []
    unconfident_indexs = []

    for i in range(0, len(noisy_targets)):
        if preds[i] == noisy_targets[i]:
            confident_indexs.append(i)
            if clean_targets[i] == preds[i]:
                confident_correct_num += 1
        else:
            unconfident_indexs.append(i)
            if clean_targets[i] == preds[i]:
                unconfident_correct_num += 1
    logger.info(
        "Confident sample num: {}, predict correct num: {}, predict accuracy: {}%.".format(
            len(confident_indexs),
            confident_correct_num,
            round(confident_correct_num / len(confident_indexs) * 100, 2),
        )
    )
    logger.info(
        "Unconfident sample num: {}, predict correct num: {}, predict accuracy: {}%.".format(
            len(unconfident_indexs),
            unconfident_correct_num,
            round(unconfident_correct_num / len(unconfident_indexs) * 100, 2),
        )
    )
    return confident_indexs, unconfident_indexs


def predict_softmax(predict_loader, model):
    # reference from PES
    softmax_outs = []
    with torch.no_grad():
        for images1, images2 in predict_loader:
            if torch.cuda.is_available():
                images1 = Variable(images1).cuda()
                images2 = Variable(images2).cuda()
                logits1, _ = model(images1)
                logits2, _ = model(images2)
                outputs = (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2
                softmax_outs.append(outputs)

    return torch.cat(softmax_outs, dim=0).cpu()


def pes_split(model, train_data, transform_train, clean_targets, noisy_targets, args):
    # reference from PES
    predict_dataset = Semi_Unlabeled_Dataset(train_data, transform=transform_train)
    predict_loader = DataLoader(
        dataset=predict_dataset,
        batch_size=args.train_batch_size * 2,
        num_workers=args.num_workers,
        shuffle=False,
    )
    soft_outputs = predict_softmax(predict_loader=predict_loader, model=model)
    confident_indexs, unconfident_indexs = split_confident(
        soft_outputs, clean_targets, noisy_targets
    )
    confident_dataset = Semi_Labeled_Dataset(
        train_data[confident_indexs], noisy_targets[confident_indexs], transform_train
    )
    unconfident_dataset = Semi_Unlabeled_Dataset(
        train_data[unconfident_indexs], transform_train
    )
    # set the ratio of confident sample and unconfident sample in one batch
    unconfident_batch_size = (
        int(args.train_batch_size / 2)
        if len(unconfident_indexs) > len(confident_indexs)
        else int(
            len(unconfident_indexs)
            / (len(confident_indexs) + len(unconfident_indexs))
            * args.train_batch_size
        )
    )
    confident_batch_size = args.train_batch_size - unconfident_batch_size
    labeled_trainloader = DataLoader(
        dataset=confident_dataset,
        batch_size=confident_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    unlabeled_trainloader = DataLoader(
        dataset=unconfident_dataset,
        batch_size=unconfident_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    # loss function
    train_nums = np.zeros(args.num_classes, dtype=int)
    for item in noisy_targets[confident_indexs]:
        train_nums[item] += 1

    # zeros are not calculated by mean
    # avoid too large numbers that may result in out of range of loss
    with np.errstate(divide="ignore"):
        cw = np.mean(train_nums[train_nums != 0]) / train_nums
        cw[cw == np.inf] = 0
        cw[cw > 3] = 3
    class_weights = torch.FloatTensor(cw).cuda()
    return labeled_trainloader, unlabeled_trainloader, class_weights


def update_index(
    model,
    proto_embedding1,
    proto_embedding2,
    confident_dataset,
    confident_loader,
    confident_iter,
    confident_indexs,
    unconfident_loader,
    unconfident_dataset,
    unconfident_iter,
    unconfident_indexs,
    noisy_targets,
    clean_targets,
    args,
):
    # num_conf_iter = int(len(confident_dataset) / args.proto_batch_size)
    remove_indexs, add_indexs, add_labels = [], [], []

    (
        conf_label_correct_num,
        conf_label_delete_num,
        conf_label_hold_num,
        conf_label_random_correct_num,
        conf_label_random_hold_num,
        conf_label_random_delete_num,
        unconf_label_add_num,
        unconf_label_correct_num,
        unconf_label_hold_num,
    ) = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    (
        conf_label_correct_right_num,
        conf_label_delete_right_num,
        conf_label_hold_right_num,
        conf_label_random_correct_right_num,
        conf_label_random_hold_right_num,
        conf_label_random_delete_right_num,
        unconf_label_add_right_num,
        unconf_label_correct_right_num,
        unconf_label_hold_right_num,
    ) = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    # for batch_idx in range(num_conf_iter):
    #     try:
    #         inputs_x1, inputs_x2, targets_x = confident_iter.next()
    #     except StopIteration:
    #         confident_iter = iter(confident_loader)
    #         inputs_x1, inputs_x2, targets_x = confident_iter.next()
    #     inputs_x1, inputs_x2, targets_x = (
    #         inputs_x1.cuda(),
    #         inputs_x2.cuda(),
    #         targets_x.cuda(),
    #     )
    #     with torch.no_grad():
    #         _, embedding1 = model(inputs_x1)
    #         _, embedding2 = model(inputs_x2)

    #         batch_size = embedding1.size(0)

    #         cos1 = torch.cosine_similarity(
    #             embedding1.unsqueeze(1).repeat(1, args.num_classes, 1),
    #             proto_embedding1.unsqueeze(0).repeat(batch_size, 1, 1),
    #             dim=-1,
    #         )
    #         cos2 = torch.cosine_similarity(
    #             embedding2.unsqueeze(1).repeat(1, args.num_classes, 1),
    #             proto_embedding2.unsqueeze(0).repeat(batch_size, 1, 1),
    #             dim=-1,
    #         )
    #     print("cos1:", cos1)
    #     print("cos2:", cos2)
    #     for i in range(batch_size):
    #         max_pos1, max_pos2 = torch.argmax(cos1[i]), torch.argmax(cos2[i])
    #         print("max_pos1: {}, value: {}".format(max_pos1, cos1[i][max_pos1]))
    #         print("max_pos2: {}, value: {}".format(max_pos2, cos2[i][max_pos2]))

    #         conf_idx = confident_indexs[args.proto_batch_size * batch_idx + i]
    #         print("conf index:", conf_idx)
    #         print("before adjust label: ", noisy_targets[conf_idx])
    #         if (
    #             max_pos1 == max_pos2
    #             and cos1[i][max_pos1] > args.cos_up_bound
    #             and cos2[i][max_pos2] > args.cos_up_bound
    #         ):
    #             # label correction
    #             print("conf label correction")
    #             noisy_targets[conf_idx] = max_pos1
    #             conf_label_correct_num += 1
    #             if clean_targets[conf_idx] == max_pos1:
    #                 conf_label_correct_right_num += 1
    #         elif (
    #             cos1[i][targets_x[i]] < args.cos_low_bound
    #             and cos2[i][targets_x[i]] < args.cos_low_bound
    #         ):
    #             # label deletion
    #             print("conf label deletion")
    #             remove_indexs.append(conf_idx)
    #             conf_label_delete_num += 1
    #             if noisy_targets[conf_idx] != clean_targets[conf_idx]:
    #                 conf_label_delete_right_num += 1

    #         elif (
    #             cos1[i][targets_x[i]] > args.cos_up_bound
    #             and cos2[i][targets_x[i]] > args.cos_up_bound
    #         ):
    #             # label holding
    #             print("label holding")
    #             conf_label_hold_num += 1
    #             if noisy_targets[conf_idx] == clean_targets[conf_idx]:
    #                 conf_label_hold_right_num += 1
    #         else:
    #             # select the sample between the big circle and small circle
    #             # label holding 80%, label correction 10%, label deletion 10%
    #             flag = np.random.uniform(0, 1)
    #             print("flag:", flag)
    #             if flag < 0.8:
    #                 conf_label_random_hold_num += 1
    #                 if noisy_targets[conf_idx] != clean_targets[conf_idx]:
    #                     conf_label_random_hold_right_num += 1
    #                 print("conf label random hold")
    #             elif flag < 0.9:
    #                 if (
    #                     max_pos1 == max_pos2
    #                     and cos1[i][max_pos1] > args.cos_up_bound
    #                     and cos2[i][max_pos2] > args.cos_up_bound
    #                 ):
    #                     # label correction
    #                     print("conf label random correction")
    #                     noisy_targets[conf_idx] = max_pos1
    #                     conf_label_random_correct_num += 1
    #                     if max_pos1 == clean_targets[conf_idx]:
    #                         conf_label_random_correct_right_num += 1
    #                 else:
    #                     print("conf label random hold")
    #                     conf_label_random_hold_num += 1
    #                     if noisy_targets[conf_idx] == clean_targets[conf_idx]:
    #                         conf_label_random_hold_right_num += 1
    #             else:
    #                 print("conf label random delete")
    #                 remove_indexs.append(conf_idx)
    #                 conf_label_random_delete_num += 1
    #                 if noisy_targets[conf_idx] != clean_targets[conf_idx]:
    #                     conf_label_random_delete_right_num += 1
    #         print("after adjust label: ", noisy_targets[conf_idx])
    #         print("true label:", clean_targets[conf_idx])
    #         print("true value1: {}".format(cos1[i][clean_targets[conf_idx]]))
    #         print("true value2: {}".format(cos2[i][clean_targets[conf_idx]]))

    num_unconf_iter = int(len(unconfident_dataset) / args.proto_batch_size)
    for batch_idx in range(num_unconf_iter):
        try:
            inputs_x1, inputs_x2 = unconfident_iter.next()
        except StopIteration:
            unconfident_iter = iter(unconfident_loader)
            inputs_x1, inputs_x2 = unconfident_iter.next()
        inputs_x1, inputs_x2 = (
            inputs_x1.cuda(),
            inputs_x2.cuda(),
        )
        with torch.no_grad():
            _, embedding1 = model(inputs_x1)
            _, embedding2 = model(inputs_x2)

            batch_size = embedding1.size(0)

            cos1 = torch.cosine_similarity(
                embedding1.unsqueeze(1).repeat(1, args.num_classes, 1),
                proto_embedding1.unsqueeze(0).repeat(batch_size, 1, 1),
                dim=-1,
            )
            cos2 = torch.cosine_similarity(
                embedding2.unsqueeze(1).repeat(1, args.num_classes, 1),
                proto_embedding2.unsqueeze(0).repeat(batch_size, 1, 1),
                dim=-1,
            )

        for i in range(batch_size):
            # label holding 5%, label adding 95%
            unconf_idx = unconfident_indexs[args.proto_batch_size * batch_idx + i]
            # print("unconf index:", unconf_idx)
            # print("before adjust label: ", noisy_targets[unconf_idx])
            # flag = np.random.uniform(0, 1)
            # if flag < 0.95:
            # print("max_pos1: {}, value: {}".format(max_pos1, cos1[i][max_pos1]))
            # print("max_pos2: {}, value: {}".format(max_pos2, cos2[i][max_pos2]))
            max_pos1, max_pos2 = torch.argmax(cos1[i]), torch.argmax(cos2[i])
            if (
                max_pos1 == max_pos2
                and cos1[i][max_pos1] > args.cos_up_bound
                and cos2[i][max_pos2] > args.cos_up_bound
            ):
                # adding labels
                # print("unconf adding")
                add_indexs.append(unconf_idx)
                # add_labels.append(max_pos1)
                if noisy_targets[unconf_idx] != max_pos1:
                    noisy_targets[unconf_idx] = max_pos1
                    unconf_label_correct_num += 1
                    if max_pos1 == clean_targets[unconf_idx]:
                        unconf_label_correct_right_num += 1

                unconf_label_add_num += 1
                if max_pos1 == clean_targets[unconf_idx]:
                    unconf_label_add_right_num += 1

            # else:
            #     print("unconf holding")
            #     add_indexs.append(unconf_idx)
            #     add_labels.append(noisy_targets[unconf_idx])
            #     unconf_label_hold_num += 1
            #     if noisy_targets[unconf_idx] == clean_targets[unconf_idx]:
            #         unconf_label_hold_right_num += 1
            # print("after adjust label: ", noisy_targets[unconf_idx])
            # print("true label:", clean_targets[unconf_idx])
            # print("true value1: {}".format(cos1[i][clean_targets[unconf_idx]]))
            # print("true value2: {}".format(cos2[i][clean_targets[unconf_idx]]))

    # logger.info(
    #     "in confident dataset, correction num: {}, right num: {}, accuracy: {}%".format(
    #         conf_label_correct_num,
    #         conf_label_correct_right_num,
    #         round(
    #             conf_label_correct_right_num / (conf_label_correct_num + 1e-4) * 100, 2
    #         ),
    #     )
    # )
    # logger.info(
    #     "in confident dataset, deletion num: {}, right num: {}, accuracy: {}%".format(
    #         conf_label_delete_num,
    #         conf_label_delete_right_num,
    #         round(
    #             conf_label_delete_right_num / (conf_label_delete_num + 1e-4) * 100, 2
    #         ),
    #     )
    # )
    # logger.info(
    #     "in confident dataset, hold num: {}, right num: {}, accuracy: {}%".format(
    #         conf_label_hold_num,
    #         conf_label_hold_right_num,
    #         round(conf_label_hold_right_num / (conf_label_hold_num + 1e-4) * 100, 2),
    #     )
    # )
    # logger.info(
    #     "in confident dataset, random correction num: {}, right num: {}, accuracy: {}%".format(
    #         conf_label_random_correct_num,
    #         conf_label_random_correct_right_num,
    #         round(
    #             conf_label_random_correct_right_num
    #             / (conf_label_random_correct_num + 1e-4)
    #             * 100,
    #             2,
    #         ),
    #     )
    # )
    # logger.info(
    #     "in confident dataset, random deletion num: {}, right num: {}, accuracy: {}%".format(
    #         conf_label_random_delete_num,
    #         conf_label_random_delete_right_num,
    #         round(
    #             conf_label_random_correct_right_num
    #             / (conf_label_random_correct_num + 1e-4)
    #             * 100,
    #             2,
    #         ),
    #     )
    # )
    # logger.info(
    #     "in confident dataset, random hold num: {}, right num: {}, accuracy: {}%".format(
    #         conf_label_random_hold_num,
    #         conf_label_random_hold_right_num,
    #         round(
    #             conf_label_random_hold_right_num
    #             / (conf_label_random_hold_num + 1e-4)
    #             * 100,
    #             2,
    #         ),
    #     )
    # )
    logger.info(
        "in unconfident dataset, adding num: {}, right num: {}, accuracy: {}%".format(
            unconf_label_add_num,
            unconf_label_add_right_num,
            round(unconf_label_add_right_num / (unconf_label_add_num + 1e-4) * 100, 2),
        )
    )

    logger.info(
        "in unconfident dataset, correct num: {}, right num: {}, accuracy: {}%".format(
            unconf_label_correct_num,
            unconf_label_correct_right_num,
            round(
                unconf_label_correct_right_num
                / (unconf_label_correct_num + 1e-4)
                * 100,
                2,
            ),
        )
    )
    # logger.info(
    #     "in unconfident dataset, holding num: {}, right num: {}, accuracy: {}%".format(
    #         unconf_label_hold_num,
    #         unconf_label_hold_right_num,
    #         round(
    #             unconf_label_hold_right_num / (unconf_label_hold_num + 1e-4) * 100, 2
    #         ),
    #     )
    # )

    # update indexs and labels

    logger.info("the number of remove confident indexs: {}".format(len(remove_indexs)))
    logger.info("the number of add confident indexs:{}".format(len(add_indexs)))
    for i in range(len(add_indexs)):
        idx = add_indexs[i]
        confident_indexs.append(idx)
        unconfident_indexs.remove(idx)

    for i in range(len(remove_indexs)):
        idx = remove_indexs[i]
        confident_indexs.remove(idx)
        unconfident_indexs.append(idx)

    # sys.exit(0)

    return confident_indexs, unconfident_indexs, noisy_targets


def calculate_confident_accuracy(confident_indexs, noisy_targets, clean_targets):
    confident_num = len(confident_indexs)
    correct_num = 0
    for i in range(confident_num):
        if noisy_targets[confident_indexs[i]] == clean_targets[confident_indexs[i]]:
            correct_num += 1
    return round((correct_num / confident_num) * 100, 2)


def proto_adjust(
    model,
    train_data,
    confident_indexs,
    unconfident_indexs,
    clean_targets,
    noisy_targets,
    transform_train,
    args,
):
    # generate prototype embedding for each class
    confident_dataset = Semi_Labeled_Dataset(
        train_data[confident_indexs], noisy_targets[confident_indexs], transform_train
    )
    confident_loader = DataLoader(
        dataset=confident_dataset,
        batch_size=args.proto_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    confident_iter = iter(confident_loader)
    num_iter = int(len(confident_dataset) / args.proto_batch_size)
    proto_embedding1 = torch.zeros(args.num_classes, 512)
    proto_embedding2 = torch.zeros(args.num_classes, 512)

    # 512 is the size of the embedding before the last fully connected layer
    proto_nums = torch.zeros(args.num_classes, 512)
    proto_embedding1, proto_embedding2, proto_nums = (
        proto_embedding1.cuda(),
        proto_embedding2.cuda(),
        proto_nums.cuda(),
    )
    for batch_idx in range(num_iter):
        try:
            inputs_x1, inputs_x2, targets_x = confident_iter.next()
        except StopIteration:
            confident_iter = iter(confident_loader)
            inputs_x1, inputs_x2, targets_x = confident_iter.next()
        inputs_x1, inputs_x2, targets_x = (
            inputs_x1.cuda(),
            inputs_x2.cuda(),
            targets_x.cuda(),
        )
        with torch.no_grad():
            _, embedding1 = model(inputs_x1)
            _, embedding2 = model(inputs_x2)
            # print(embedding1.size())
        for i in range(targets_x.size(0)):
            proto_embedding1[targets_x[i]] += embedding1[i]
            proto_embedding2[targets_x[i]] += embedding2[i]
            proto_nums[targets_x[i]] += 1
    proto_embedding1 = torch.div(
        proto_embedding1,
        proto_nums,
    )
    proto_embedding2 = torch.div(
        proto_embedding2,
        proto_nums,
    )
    logger.info(
        "the number of prototypes in each classes: {}".format(proto_nums[:, :1])
    )

    unconfident_dataset = Semi_Unlabeled_Dataset(
        train_data[unconfident_indexs], transform_train
    )
    unconfident_loader = DataLoader(
        dataset=unconfident_dataset,
        batch_size=args.proto_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    unconfident_iter = iter(unconfident_loader)

    logger.info("Before proto-split...")
    logger.info("the number of confident samples: {}".format(len(confident_indexs)))
    logger.info("the number of unconfident samples: {}".format(len(unconfident_indexs)))
    confident_accuracy = calculate_confident_accuracy(
        confident_indexs, noisy_targets, clean_targets
    )
    logger.info("confident predict accuracy: {}%".format(confident_accuracy))
    confident_indexs, unconfident_indexs, noisy_targets = update_index(
        model,
        proto_embedding1,
        proto_embedding2,
        confident_dataset,
        confident_loader,
        confident_iter,
        confident_indexs,
        unconfident_loader,
        unconfident_dataset,
        unconfident_iter,
        unconfident_indexs,
        noisy_targets,
        clean_targets,
        args,
    )
    logger.info("After proto-split...")
    logger.info("the number of confident samples: {}".format(len(confident_indexs)))
    logger.info("the number of unconfident samples: {}".format(len(unconfident_indexs)))
    confident_accuracy = calculate_confident_accuracy(
        confident_indexs, noisy_targets, clean_targets
    )
    logger.info("confident predict accuracy: {}%".format(confident_accuracy))
    return confident_indexs, unconfident_indexs, noisy_targets


def proto_split(model, train_data, transform_train, clean_targets, noisy_targets, args):
    predict_dataset = Semi_Unlabeled_Dataset(train_data, transform=transform_train)
    predict_loader = DataLoader(
        dataset=predict_dataset,
        batch_size=args.train_batch_size * 2,
        num_workers=args.num_workers,
        shuffle=False,
    )
    soft_outputs = predict_softmax(predict_loader=predict_loader, model=model)
    confident_indexs, unconfident_indexs = split_confident(
        soft_outputs, clean_targets, noisy_targets
    )
    # use prototype embeeding to adjust confident_indexs, unconfident_index, and noisy_targets
    if args.epoch_now < args.warmups + args.proto_epochs:
        confident_indexs, unconfident_indexs, noisy_targets = proto_adjust(
            model,
            train_data,
            confident_indexs,
            unconfident_indexs,
            clean_targets,
            noisy_targets,
            transform_train,
            args,
        )

    confident_dataset = Semi_Labeled_Dataset(
        train_data[confident_indexs], noisy_targets[confident_indexs], transform_train
    )
    unconfident_dataset = Semi_Unlabeled_Dataset(
        train_data[unconfident_indexs], transform_train
    )
    # set the ratio of confident sample and unconfident sample in one batch
    unconfident_batch_size = (
        int(args.train_batch_size / 2)
        if len(unconfident_indexs) > len(confident_indexs)
        else int(
            len(unconfident_indexs)
            / (len(confident_indexs) + len(unconfident_indexs))
            * args.train_batch_size
        )
    )
    confident_batch_size = args.train_batch_size - unconfident_batch_size
    labeled_trainloader = DataLoader(
        dataset=confident_dataset,
        batch_size=confident_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    unlabeled_trainloader = DataLoader(
        dataset=unconfident_dataset,
        batch_size=unconfident_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    # loss function
    train_nums = np.zeros(args.num_classes, dtype=int)
    for item in noisy_targets[confident_indexs]:
        train_nums[item] += 1

    # zeros are not calculated by mean
    # avoid too large numbers that may result in out of range of loss
    with np.errstate(divide="ignore"):
        cw = np.mean(train_nums[train_nums != 0]) / train_nums
        cw[cw == np.inf] = 0
        cw[cw > 3] = 3
    class_weights = torch.FloatTensor(cw).cuda()
    return labeled_trainloader, unlabeled_trainloader, class_weights
