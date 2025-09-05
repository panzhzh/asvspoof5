"""
Utility functions: optimizer, scheduler, seeding, and helpers.
"""

import random
import sys

import numpy as np
import torch


def str_to_bool(val):
    """Convert common string/boolean to bool."""
    if isinstance(val, bool):
        return val
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise ValueError('invalid truth value {}'.format(val))


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def keras_decay(step, decay=0.0001):
    return 1. / (1. + decay * step)


class SGDRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T0, T_mul, eta_min, last_epoch=-1):
        self.Ti = T0
        self.T_mul = T_mul
        self.eta_min = eta_min
        self.last_restart = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        T_cur = self.last_epoch - self.last_restart
        if T_cur >= self.Ti:
            self.last_restart = self.last_epoch
            self.Ti = self.Ti * self.T_mul
            T_cur = 0
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + np.cos(np.pi * T_cur / self.Ti)) / 2
            for base_lr in self.base_lrs
        ]


def _get_optimizer(model_parameters, optim_config):
    name = optim_config['optimizer'] if isinstance(optim_config, dict) else optim_config.optimizer
    if name == 'sgd':
        if isinstance(optim_config, dict):
            params = optim_config
        else:
            params = optim_config.__dict__
        optimizer = torch.optim.SGD(model_parameters,
                                    lr=params['base_lr'],
                                    momentum=params.get('momentum', 0.9),
                                    weight_decay=params['weight_decay'],
                                    nesterov=params.get('nesterov', False))
    elif name == 'adam':
        if isinstance(optim_config, dict):
            params = optim_config
        else:
            params = optim_config.__dict__
        optimizer = torch.optim.Adam(model_parameters,
                                     lr=params['base_lr'],
                                     betas=tuple(params['betas']) if isinstance(params['betas'], list) else params['betas'],
                                     weight_decay=params['weight_decay'],
                                     amsgrad=str_to_bool(params.get('amsgrad', False)))
    else:
        print('Un-known optimizer', name)
        sys.exit(1)
    return optimizer


def _get_scheduler(optimizer, optim_config, num_epochs=100, steps_per_epoch=1000):
    name = optim_config['scheduler'] if isinstance(optim_config, dict) else optim_config.scheduler
    if isinstance(optim_config, dict):
        params = optim_config
    else:
        params = optim_config.__dict__

    if name == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=params['milestones'],
            gamma=params['lr_decay'])
    elif name == 'sgdr':
        scheduler = SGDRScheduler(optimizer, params['T0'], params['Tmult'], params['lr_min'])
    elif name == 'cosine':
        total_steps = num_epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,
                params['lr_min'] / params['base_lr']))
    elif name == 'keras_decay':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: keras_decay(step))
    else:
        scheduler = None
    return scheduler


def create_optimizer(model_parameters, optim_config, num_epochs=100, steps_per_epoch=1000):
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config, num_epochs, steps_per_epoch)
    return optimizer, scheduler


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed, config=None):
    if config is None:
        raise ValueError("config should not be None")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Accept both dict and object-like config
        if isinstance(config, dict):
            deterministic = str_to_bool(config.get("cudnn_deterministic_toggle", False))
            benchmark = str_to_bool(config.get("cudnn_benchmark_toggle", True))
        else:
            deterministic = str_to_bool(getattr(config, "cudnn_deterministic_toggle", False))
            benchmark = str_to_bool(getattr(config, "cudnn_benchmark_toggle", True))
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark

