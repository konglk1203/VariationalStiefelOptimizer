import torch
import pytorch_lightning as pl
import warmup_scheduler
from utils import get_model, get_criterion


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams.update(vars(hparams))
        self.model = get_model(hparams)
        self.criterion = get_criterion(self.hparams)
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        constraint=self.hparams.constraint
        optim_method=self.hparams.optim_method
        if constraint !=None:
            orth_param_list=[]
            non_orth_param_list=[]
            for name, param in self.named_parameters():
                if 'q.weight' in name or 'k.weight' in name or (('q_list' in name or 'k_list' in name) and 'weight' in name):
                    orth_param_list.append(param)
                else:
                    non_orth_param_list.append(param)
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