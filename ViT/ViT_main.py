# Modified from https://github.com/omihub777/ViT-CIFAR

import sys
sys.path.append('../Optimizers')
sys.path.append('..')

import argparse
import torch
import pytorch_lightning as pl
import numpy as np
from utils import get_dataset
from pl_net import Net





if __name__ == "__main__":
    # The hyperparameters for model and training process are from https://github.com/omihub777/ViT-CIFAR
    parser = argparse.ArgumentParser()
    parser.add_argument("--optim-method", default='ProjectedStiefelAdam', type=str, help="['SGD','Adam', 'AdamW','RegularizerStiefelSGD', 'RegularizerStiefelAdam', 'ProjectedStiefelSGD', 'ProjectedStiefelAdam', 'StiefelSGD', 'StiefelAdam', 'MomentumlessStiefelSGD']")
    parser.add_argument("--constraint", default='OnlyWithin', type=str, help="['Across', 'OnlyWithin', None]")
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
    parser.add_argument('--no-autoaugment', dest='autoaugment', action='store_false')
    parser.set_defaults(autoaugment=True)
    parser.add_argument("--criterion", default="ce")
    parser.add_argument('--label-smoothing', action='store_true')
    parser.add_argument('--no-label-smoothing', dest='label-smoothing', action='store_false')
    parser.set_defaults(label_smoothing=True)
    parser.add_argument("--smoothing", default=0.1, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--head", default=12, type=int)
    parser.add_argument("--num-layers", default=7, type=int)
    parser.add_argument("--hidden", default=384, type=int)
    parser.add_argument("--mlp-hidden", default=384, type=int)
    parser.add_argument("--off-cls-token", action="store_true")
    parser.add_argument("--seed", default=None, type=int)
    # if there is an error, try to change num_workers = 0
    parser.add_argument("--num_workers", default=0, type=int)
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
    if torch.cuda.is_available():
        args.accelerator='gpu'
        args.devices = torch.cuda.device_count()
    else:
        args.accelerator='cpu'
        args.devices = 1

    

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
    trainer = pl.Trainer(precision=args.precision,fast_dev_run=args.dry_run, devices=args.devices, accelerator=args.accelerator, benchmark=args.benchmark, logger=logger, max_epochs=args.max_epochs)
    trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=test_dl)


