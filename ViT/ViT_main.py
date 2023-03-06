# Modified from https://github.com/omihub777/ViT-CIFAR

import sys
sys.path.append('../Optimizers')
sys.path.append('..')

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import warmup_scheduler
import numpy as np
from utils import get_model, get_dataset, get_experiment_name, get_criterion, get_model_square



class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams.update(vars(hparams))
        if constraint=='Across':
            self.model = get_model_square(hparams)   
        elif constraint=='OnlyWithin' or constraint==None:
            self.model = get_model(hparams)
        else:
            raise NotImplementedError()
        self.criterion = get_criterion(args)
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if constraint !=None:
            if constraint=='Across':
                orth_param_list=[]
                non_orth_param_list=[]
                for name, param in net.named_parameters():
                    if 'q.weight' in name or 'k.weight' in name:
                        orth_param_list.append(param)
                    else:
                        non_orth_param_list.append(param)
            elif constraint=='OnlyWithin':
                orth_param_list=[]
                non_orth_param_list=[]
                for name, param in net.named_parameters():
                    if 'q_list' in name or 'k_list' in name:
                        orth_param_list.append(param)
                    else:
                        non_orth_param_list.append(param)
            else:
                raise NotImplementedError()
            if 'SGD' in optim_method:
                op1=torch.optim.SGD(non_orth_param_list, lr=self.hparams.lr,  momentum=self.hparams.beta1, weight_decay=self.hparams.weight_decay)
                if optim_method=='StiefelSGD':
                    from StiefelOptimizers import StiefelSGD
                    op2=StiefelSGD(orth_param_list, lr=self.hparams.lr,  momentum=self.hparams.beta1, dampening=0)
                elif optim_method=='RegularizerStiefelSGD':
                    from StiefelRegularizer import RegularizerStiefelSGD
                    op2=RegularizerStiefelSGD(orth_param_list, lr=self.hparams.lr, momentum=self.hparams.beta1, stiefel_regularizer=self.hparams.regularizer_parameter)
                elif optim_method=='ProjectedStiefelSGD':
                    import ProjectedStiefelOptimizer.stiefel_optimizer
                    from ProjectedStiefelOptimizer.gutils import unit, qr_retraction
                    for value in orth_param_list:
                        q = qr_retraction(value.data.view(value.size(0), -1)) 
                        value.data.copy_(q.view(value.size()))
                    op2=ProjectedStiefelOptimizer.stiefel_optimizer.SGDG([{'params':orth_param_list,'lr':self.hparams.lr,'momentum':self.hparams.beta1, 'stiefel':True,'nesterov':True, 'ratio':self.hparams.lr_ratio}])
                elif optim_method=='MomentumlessStiefelSGD':
                    import MomentumlessStiefelSGD
                    op2=MomentumlessStiefelSGD.MomentumlessStiefelSGD(orth_param_list, lr=self.hparams.lr)
            elif 'Adam' in optim_method:
                op1=torch.optim.Adam(non_orth_param_list, lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
                if optim_method=='StiefelAdam':
                    from StiefelOptimizers import StiefelAdam
                    op2=StiefelAdam(orth_param_list, lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))
                elif optim_method=='RegularizerStiefelAdam':
                    from StiefelRegularizer import RegularizerStiefelAdam
                    op2=RegularizerStiefelAdam(orth_param_list, lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2),  stiefel_regularizer=self.hparams.regularizer_parameter)
                elif optim_method=='ProjectedStiefelAdam':
                    import ProjectedStiefelOptimizer.stiefel_optimizer
                    from ProjectedStiefelOptimizer.gutils import unit, qr_retraction
                    for value in orth_param_list:
                        q = qr_retraction(value.data.view(value.size(0), -1)) 
                        value.data.copy_(q.view(value.size()))
                    op2=ProjectedStiefelOptimizer.stiefel_optimizer.AdamG([{'params':orth_param_list,'lr':self.hparams.lr,'momentum':self.hparams.beta1,'beta2':self.hparams.beta2, 'stiefel':True,'nesterov':True, 'ratio':self.hparams.lr_ratio}])
            else:
                raise NotImplementedError()
            from StiefelOptimizers import CombinedOptimizer
            self.optimizer=CombinedOptimizer(op1, op2)
        else:
            if optim_method=='Adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
            elif optim_method=='AdamW':
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=4*self.hparams.weight_decay/self.hparams.lr)
            elif optim_method=='SGD':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=self.hparams.beta1, weight_decay=self.hparams.weight_decay)
            else:
                raise NotImplementedError()
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=self.hparams.warmup_epoch, after_scheduler=self.base_scheduler)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("loss", loss)
        self.log("acc", acc)
        
        
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss


if __name__ == "__main__":
    # The hyperparameters for model and training process are from https://github.com/omihub777/ViT-CIFAR
    parser = argparse.ArgumentParser()
    parser.add_argument("--optim-method", default=None, type=str, help="['SGD','Adam', 'AdamW','RegularizerStiefelSGD', 'RegularizerStiefelAdam', 'ProjectedStiefelSGD', 'ProjectedStiefelAdam', 'StiefelSGD', 'StiefelAdam', 'MomentumlessStiefelSGD']")
    parser.add_argument("--constraint", default=None, type=str, help="['Across', 'OnlyWithin', None]")
    parser.add_argument("--dataset", default="c10", type=str, help="[c10, c100]")
    parser.add_argument("--patch", default=8, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--eval-batch-size", default=1024, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--min-lr", default=None, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--off-benchmark", action="store_true")
    parser.add_argument("--max-epochs", default=200, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--weight-decay", default=5e-5, type=float)
    parser.add_argument("--warmup-epoch", default=5, type=int)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--autoaugment", action="store_true")
    parser.add_argument("--criterion", default="ce")
    parser.add_argument("--label-smoothing", action="store_true")
    parser.add_argument("--smoothing", default=0.1, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--head", default=12, type=int)
    parser.add_argument("--num-layers", default=7, type=int)
    parser.add_argument("--hidden", default=384, type=int)
    parser.add_argument("--mlp-hidden", default=384, type=int)
    parser.add_argument("--off-cls-token", action="store_true")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--lr-ratio", default=10.0, type=float, help='Only needed for ICLR 2020. The ratio for lr for Stiefel parameters and Euclidean parameters') # The default is the choice after carefully adjusted
    parser.add_argument("--regularizer-parameter", default=1e-6, type=float, help='Only needed for regularizer SGD/Adam') # The default is the choice after carefully adjusted
    parser.add_argument("--experiment-name", default=None, type=str)

    args = parser.parse_args()

    optim_method=args.optim_method
    constraint=args.constraint
    assert optim_method in ['SGD','Adam', 'AdamW','RegularizerStiefelSGD', 'RegularizerStiefelAdam', 'ProjectedStiefelSGD', 'ProjectedStiefelAdam', 'StiefelSGD', 'StiefelAdam', 'MomentumlessStiefelSGD']
    assert constraint in ['Across', 'OnlyWithin', None]
    if args.lr==None:
        if optim_method in ['SGD','RegularizerStiefelSGD', 'ProjectedStiefelSGD', 'MomentumlessStiefelSGD']:
            args.lr=1e-1 # already carefully adjusted
        elif optim_method== 'StiefelSGD':
            args.lr=1.5e-1 # already carefully adjusted
        else:
            args.lr=1e-3 # already carefully adjusted

    if args.min_lr==None:
        args.min_lr=args.lr*0.01

    if args.seed!=None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    args.benchmark = True if not args.off_benchmark else False
    args.gpus = torch.cuda.device_count()

    # if there is an error, try to change num_workers = 0
    args.num_workers = 4*args.gpus if args.gpus else 8

    if args.mlp_hidden != args.hidden*4:
        print(f"[INFO] In original paper, mlp_hidden(CURRENT:{args.mlp_hidden}) is set to: {args.hidden*4}(={args.hidden}*4)")

    train_ds, test_ds = get_dataset(args)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)
    if args.experiment_name== None:
        attribute_list=[str(attribute) for attribute in [args.optim_method, args.constraint,  args.dataset]]
        experiment_name = '_'.join(attribute_list)
    else:
        experiment_name=args.experiment_name
    print(experiment_name)


    print("[INFO] Log with CSV") 
    logger = pl.loggers.CSVLogger(
    save_dir="logs",
    name=experiment_name
    )
    refresh_rate = 1

    net = Net(args)
    trainer = pl.Trainer(precision=args.precision,fast_dev_run=args.dry_run, gpus=args.gpus, benchmark=args.benchmark, logger=logger, max_epochs=args.max_epochs)
    trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=test_dl)


