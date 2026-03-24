import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from AA.autoattack.autoattack import AutoAttack as AutoAttack_mod
from models import *


def _get_model_device(model):
    return next(model.parameters()).device


def cwloss(output, target, confidence=50):
    """
    output: logits [N, C]
    target: class indices [N]
    """
    target = target.long()
    num_classes = output.size(1)

    # true-class logit
    real = output.gather(1, target.view(-1, 1)).squeeze(1)

    # max logit among all other classes
    one_hot = F.one_hot(target, num_classes=num_classes).bool()
    other = output.masked_fill(one_hot, float('-inf')).max(1).values

    # same sign convention as your original code (negative clamp then sum)
    loss = -torch.clamp(real - other + confidence, min=0.0)
    return loss.sum()


def pgd(model, data, target, epsilon, step_size, num_steps, loss_fn, category, rand_init):
    model.eval()
    device = data.device

    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape, device=device).detach() if rand_init else data.detach()

    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(
            np.random.uniform(-epsilon, epsilon, data.shape)
        ).float().to(device) if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()

        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output, target)

        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


def eval_clean(model, test_loader):
    model.eval()
    device = _get_model_device(model)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    log = 'Natrual Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, rand_init):
    model.eval()
    device = _get_model_device(model)
    test_loss = 0
    correct = 0

    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x_adv = pgd(model, data, target, epsilon, step_size, perturb_steps, loss_fn, category, rand_init=rand_init)
            output = model(x_adv)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    log = 'Attack Setting ==> Loss_fn:{}, Perturb steps:{}, Epsilon:{}, Step dize:{} \n Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        loss_fn, perturb_steps, epsilon, step_size,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


####For AutoAttack
def eval_robust_aa(model, test_loader, epsilon, step_size):
    model.eval()
    device = _get_model_device(model)
    test_loss = 0
    correct = 0
    adversary = AutoAttack_mod(model, norm='Linf', eps=epsilon, version='standard', verbose=False)

    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x_adv = adversary.run_standard_evaluation(data, target, bs=data.shape[0])
            output = model(x_adv[0])
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    log = 'Attack Setting ==> Epsilon:{}, Step dize:{} \n Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epsilon, step_size,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


# Geometry-aware projected gradient descent (GA-PGD)
def GA_PGD(model, data, target, epsilon, step_size, num_steps, loss_fn, category, rand_init):
    model.eval()
    device = data.device
    Kappa = torch.zeros(len(data), device=device)

    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape, device=device).detach() if rand_init else data.detach()
        nat_output = model(data)

    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(
            np.random.uniform(-epsilon, epsilon, data.shape)
        ).float().to(device) if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        predict = output.max(1, keepdim=True)[1]

        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1

        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output, target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).to(device)
                loss_adv = criterion_kl(F.log_softmax(output, dim=1), F.softmax(nat_output, dim=1))

        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()

        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, Kappa
