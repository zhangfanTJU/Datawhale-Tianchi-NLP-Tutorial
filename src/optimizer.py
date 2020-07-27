import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup


class Optimizer:
    def __init__(self, model_parameters, config, batch_num):
        self.all_params = []
        self.optims = []
        self.schedulers = []

        for name, parameters in model_parameters.items():
            if name.startswith("basic"):
                optim = torch.optim.Adam(parameters, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                         eps=config.epsilon)
                self.optims.append(optim)

                decay, decay_step = config.decay, config.decay_steps
                l = lambda step: decay ** (step // decay_step)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=l)
                self.schedulers.append(scheduler)
                self.all_params.extend(parameters)

            elif name.startswith("bert"):
                optim_bert = AdamW(parameters, config.bert_lr, eps=1e-8)
                self.optims.append(optim_bert)

                steps = int(np.ceil(batch_num / config.update_every)) * config.epochs
                scheduler_bert = get_linear_schedule_with_warmup(optim_bert, 0, steps)
                self.schedulers.append(scheduler_bert)

                for group in parameters:
                    for p in group['params']:
                        self.all_params.append(p)

            else:
                Exception("no nameed parameters.")

        self.num = len(self.optims)

    def step(self):
        for optim, scheduler in zip(self.optims, self.schedulers):
            optim.step()
            scheduler.step()
            optim.zero_grad()

    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def get_lr(self):
        lrs = tuple(map(lambda x: x.get_lr()[-1], self.schedulers))
        lr = ' %.5f' * self.num
        res = lr % lrs
        return res
