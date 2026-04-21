import os
import argparse
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
import datetime
from models import *
from earlystop import earlystop
import numpy as np
from utils import Logger
import attack_generator as attack
import itertools
from utils.convert_to_data_loader import dataloader_generation


args = None
out_dir = None
lambda_k = None
lambda_hat = None
intial_loss = []


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch AoRR Adversarial Training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train, 120 for WRN, 100 for resnet, 50 for lenet')
    parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
    parser.add_argument('--epsilon', type=float, default=0.0078,
                        help='perturbation bound， 0.0078, 0.031 for CIFAR, 0.1, 0.3 for MNIST')
    parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
    parser.add_argument('--step_size', type=float, default=0.0078 / 4, help='step size')
    parser.add_argument('--seed', type=int, default=7, metavar='S', help='random seed')
    parser.add_argument('--net', type=str, default="smallcnn",
                        help="decide which network to use,choose from smallcnn,smallcnn_for_mnist,resnet18,WRN,WRN_madry,lenet_mnist")
    parser.add_argument('--dataset', type=str, default="mnist_noise_asym_40",
                        help="choose from cifar10,svhn,mnist,mnist_noise_sym_20")
    parser.add_argument('--rand_init', type=bool, default=True,
                        help="whether to initialize adversarial sample with random noise")
    parser.add_argument('--depth', type=int, default=32, help='WRN depth')
    parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
    parser.add_argument('--out_dir', type=str, default='./AoRRAT_results', help='dir of output')
    parser.add_argument('--resume', type=str, default='', help='whether to resume training, default: None')
    parser.add_argument('--gpuid', type=str, default='2', help='GPU ID')
    parser.add_argument('--k', type=int, default=50000, help='k')
    parser.add_argument('--m', type=int, default=500, help='m')
    parser.add_argument('--aorr', type=bool, default=True, help="use aorr or not")
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Run expensive robust evaluation every N epochs, and always on the final epoch')
    return parser.parse_args()


def train(epoch, model, train_loader, optimizer, device):
    global lambda_k, lambda_hat, intial_loss, args

    starttime = datetime.datetime.now()
    loss_sum = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx % 50 == 0:
            print(f"epoch {epoch} batch {batch_idx}/{len(train_loader)}")

        data, target = data.to(device), target.to(device)

        output_adv = attack.pgd(
            model,
            data,
            target,
            epsilon=args.epsilon,
            step_size=args.step_size,
            num_steps=args.num_steps,
            loss_fn='cent',
            category="Madry",
            rand_init=True
        )

        model.train()
        optimizer.zero_grad()
        output = model(output_adv)

        if args.aorr:
            if epoch == 0:
                intial_loss.append(
                    nn.CrossEntropyLoss(reduction='none')(output, target).cpu().detach().numpy().tolist()
                )
                loss = nn.CrossEntropyLoss(reduction='mean')(output, target)
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()
            else:
                if epoch == 1 and batch_idx == 0:
                    lambda_k.data = torch.topk(
                        torch.from_numpy(
                            np.asarray(list(itertools.chain(*intial_loss)), dtype=np.float32)
                        ).to(device),
                        args.k,
                        sorted=True,
                        dim=0
                    )[0][-1].data.flatten().to(device)

                    lambda_hat.data = torch.topk(
                        torch.from_numpy(
                            np.asarray(list(itertools.chain(*intial_loss)), dtype=np.float32)
                        ).to(device),
                        args.m,
                        sorted=True,
                        dim=0
                    )[0][-1].data.flatten().to(device)

                n_train = len(train_loader.dataset)
                loss_term_1 = (args.k - args.m) * lambda_k / n_train
                loss_term_2 = (n_train - args.m) * lambda_hat / n_train

                cr_loss = nn.CrossEntropyLoss(reduction='none')(output, target)
                loss_term_3 = cr_loss - lambda_k
                loss_term_3[loss_term_3 < 0] = 0
                loss_term_3 = lambda_hat - loss_term_3
                loss_term_3[loss_term_3 < 0] = 0
                loss = loss_term_1 + loss_term_2 - loss_term_3
                loss = torch.mean(loss)
                loss_sum += loss.item()

                optimizer.zero_grad()
                lambda_k.retain_grad()
                lambda_hat.retain_grad()
                loss.backward()
                optimizer.step()

                lambda_k.data = lambda_k.data - args.lr * lambda_k.grad.data
                lambda_hat.data = lambda_hat.data + args.lr * lambda_hat.grad.data
                lambda_k.grad.data.zero_()
                lambda_hat.grad.data.zero_()
        else:
            loss = nn.CrossEntropyLoss(reduction='mean')(output, target)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds

    return time, loss_sum


def adjust_learning_rate(optimizer, epoch):
    global args

    lr = args.lr
    if epoch >= 30:
        lr = args.lr * 0.1
    if epoch >= 60:
        lr = args.lr * 0.01
    if epoch >= 110:
        lr = args.lr * 0.005

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, checkpoint=None, filename='checkpoint.pth.tar'):
    global out_dir

    if checkpoint is None:
        checkpoint = out_dir

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def run_orat_training(
    model,
    train_loader,
    test_loader,
    device,
    epochs,
    lr,
    momentum,
    weight_decay,
    epsilon,
    num_steps,
    step_size,
    k,
    m,
    aorr,
    out_dir,
    resume='',
    eval_every=10,
):
    global args, lambda_k, lambda_hat, intial_loss

    class TempArgs:
        pass

    args = TempArgs()
    args.epochs = epochs
    args.lr = lr
    args.momentum = momentum
    args.weight_decay = weight_decay
    args.epsilon = epsilon
    args.num_steps = num_steps
    args.step_size = step_size
    args.k = k
    args.m = m
    args.aorr = aorr
    args.out_dir = out_dir
    args.resume = resume
    args.eval_every = max(1, int(eval_every))

    lambda_k = Variable(torch.tensor([0.0], device=device), requires_grad=True)
    lambda_hat = Variable(torch.tensor([0.0], device=device), requires_grad=True)
    intial_loss = []

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    start_epoch = 0
    title = 'AoRRAT train'

    if resume:
        print('==> AoRR Adversarial Training Resuming from checkpoint ..')
        print(resume)
        assert os.path.isfile(resume)
        actual_out_dir = os.path.dirname(resume)
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger_test = Logger(os.path.join(actual_out_dir, 'log_results.txt'), title=title, resume=True)
    else:
        print('==> AoRR Adversarial Training')
        actual_out_dir = out_dir
        logger_test = Logger(os.path.join(actual_out_dir, 'log_results.txt'), title=title)
        logger_test.set_names(['Epoch', 'Natural Test Acc', 'FGSM Acc', 'PGD20 Acc', 'CW Acc'])

    test_nat_acc = 0.0
    fgsm_acc = float("nan")
    test_pgd20_acc = float("nan")
    cw_acc = float("nan")
    best_natural = 0.0
    best_fsgm = 0.0
    best_pgd20 = 0.0
    best_cw = 0.0

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch + 1)
        train_time, train_loss = train(epoch, model, train_loader, optimizer, device)

        loss, test_nat_acc = attack.eval_clean(model, test_loader)

        should_run_expensive_eval = ((epoch + 1) % args.eval_every == 0) or ((epoch + 1) == epochs)

        if should_run_expensive_eval:
            loss, fgsm_acc = attack.eval_robust(
                model, test_loader,
                perturb_steps=1,
                epsilon=epsilon,
                step_size=epsilon,
                loss_fn="cent",
                category="Madry",
                rand_init=True
            )
            loss, test_pgd20_acc = attack.eval_robust(
                model, test_loader,
                perturb_steps=20,
                epsilon=epsilon,
                step_size=epsilon / 4,
                loss_fn="cent",
                category="Madry",
                rand_init=True
            )
            loss, cw_acc = attack.eval_robust(
                model, test_loader,
                perturb_steps=30,
                epsilon=epsilon,
                step_size=epsilon / 4,
                loss_fn="cw",
                category="Madry",
                rand_init=True
            )

            if best_fsgm < fgsm_acc:
                best_fsgm = fgsm_acc
            if best_pgd20 < test_pgd20_acc:
                best_pgd20 = test_pgd20_acc
            if best_cw < cw_acc:
                best_cw = cw_acc
        else:
            fgsm_acc = float("nan")
            test_pgd20_acc = float("nan")
            cw_acc = float("nan")

        if best_natural < test_nat_acc:
            best_natural = test_nat_acc

        print(
            'Epoch: [%d | %d] | Train Time: %.2f s | train_loss: %.4f | Natural Test Acc %.4f | FGSM Test Acc %s | PGD20 Test Acc %s | CW Test Acc %s |\n'
            % (
                epoch + 1,
                epochs,
                train_time,
                train_loss,
                test_nat_acc,
                f"{fgsm_acc:.4f}" if not np.isnan(fgsm_acc) else "skipped",
                f"{test_pgd20_acc:.4f}" if not np.isnan(test_pgd20_acc) else "skipped",
                f"{cw_acc:.4f}" if not np.isnan(cw_acc) else "skipped",
            )
        )

        if (epoch + 1) == epochs:
            print(
                'Best: | Natural Best Acc %.4f | FGSM Best Acc %.4f | PGD20 Best Acc %.4f | CW Best Acc %.4f |\n'
                % (
                    best_natural,
                    best_fsgm,
                    best_pgd20,
                    best_cw
                )
            )

        logger_test.append([epoch + 1, test_nat_acc, fgsm_acc, test_pgd20_acc, cw_acc])

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'test_nat_acc': test_nat_acc,
            'test_pgd20_acc': test_pgd20_acc if not np.isnan(test_pgd20_acc) else -1.0,
            'optimizer': optimizer.state_dict(),
        }, checkpoint=actual_out_dir)

    return model


def main():
    global args, out_dir

    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    out_str = str(args)
    print(out_str)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('==> Load Test Data')
    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    if args.dataset == "svhn":
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    if args.dataset == "mnist":
        trainset = torchvision.datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, pin_memory=True)
        testset = torchvision.datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, pin_memory=True)

    if 'mnist_noise' in args.dataset:
        trainset, testset = dataloader_generation(
            data_path='./data/mnist_noise_data/{}.mat'.format(args.dataset)
        )
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    if 'cifar_10_noise' in args.dataset:
        trainset, testset = dataloader_generation(
            data_path='./data/cifar_10_noise_data/{}.mat'.format(args.dataset)
        )
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    print('==> Load Model')
    if args.net == "smallcnn":
        model = SmallCNN().to(device)
        net = "smallcnn"
    if args.net == "smallcnn_for_mnist":
        model = SmallCNN_for_mnist().to(device)
        net = "smallcnn_for_mnist"
    if args.net == "lenet_mnist":
        model = LeNet_mnist().to(device)
        net = "lenet_mnist"
    if args.net == "resnet18":
        model = ResNet18().to(device)
        net = "resnet18"
    if args.net == "resnet20":
        model = resnet20().to(device)
        net = "resnet20"
    if args.net == "WRN":
        model = Wide_ResNet(
            depth=args.depth,
            num_classes=10,
            widen_factor=args.width_factor,
            dropRate=args.drop_rate
        ).to(device)
        net = "WRN{}-{}-dropout{}".format(args.depth, args.width_factor, args.drop_rate)
    if args.net == 'WRN_madry':
        model = Wide_ResNet_Madry(
            depth=args.depth,
            num_classes=10,
            widen_factor=args.width_factor,
            dropRate=args.drop_rate
        ).to(device)
        net = "WRN_madry{}-{}-dropout{}".format(args.depth, args.width_factor, args.drop_rate)

    print(net)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = run_orat_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        epsilon=args.epsilon,
        num_steps=args.num_steps,
        step_size=args.step_size,
        k=args.k,
        m=args.m,
        aorr=args.aorr,
        out_dir=args.out_dir,
        resume=args.resume,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
