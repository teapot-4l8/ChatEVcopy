import lightning as L
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import parse
import torch
import random
from model_interface import MInterface
from data_interface import MyDataModule
import os


def load_callbacks(args):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='metric',
        mode='max',
        patience=10,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='metric',
        dirpath='../ckpt/' + args.data_name,
        filename='{epoch:02d}-{metric:.3f}',
        save_top_k=-1,
        mode='max',
        save_last=True,
        #train_time_interval=args.val_check_interval
        every_n_epochs=1
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='step'))
    return callbacks


def main(args):
    print(f"Welcome to ChatEV! Now, we are working on Charging Data: {args.data_name}.")
    L.seed_everything(args.seed)
    
    # Memory optimizations
    torch.set_float32_matmul_precision('medium')
    torch.cuda.empty_cache()
    
    # Set environment variable for memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    callbacks = load_callbacks(args)
    model = MInterface(**vars(args))
    if args.ckpt:
        ckpt_path = '../ckpt/' + args.data_name + '/' + args.ckpt_name + '.ckpt'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print("load checkpoints from {}".format(ckpt_path))
    
    if args.test_only:
        data_module = MyDataModule(args)
        trainer.test(model=model, datamodule=data_module)  # test
    else:
        
        # meta-learning (First-order Reptile): Learn shared knowledge from source zones and improve prediction performance in the target zone.
        if args.meta_learning:  
            random_zones = [random.randint(1, 200) for _ in range(args.inner_loop-1)]  # You can select the source zones based on specific rules
            random_zones.append(args.zone)
            for i in range(args.outer_loop):  # the meta-learning epochs
                for j in range(args.inner_loop):  # the meta-learning zones
                    print(f"We are now in the {i}/{args.outer_loop} outer epoch and the {j}/{args.inner_loop} zone.")
                    args.zone = random_zones[j]
                    data_module = MyDataModule(args)
                    trainer = pl.Trainer(
                        devices=[int(args.cuda)],
                        accelerator='cuda',
                        max_epochs=1,
                        logger=True,
                        callbacks=callbacks,
                        precision='16-mixed',  # Enable mixed precision
                        gradient_clip_val=1.0,  # Add gradient clipping
                        accumulate_grad_batches=2,  # Add gradient accumulation
                        enable_progress_bar=True,
                        deterministic=True,
                    )
                    trainer.fit(model=model, datamodule=data_module)  # train and valid
            trainer.test(model=model, datamodule=data_module)  # test  
            
        # normal learning          
        else:
            data_module = MyDataModule(args)
            trainer = pl.Trainer(
                devices=[int(args.cuda)],
                accelerator='cuda',
                max_epochs=args.max_epochs,
                logger=True,
                callbacks=callbacks,
                precision='16-mixed',  # Enable mixed precision
                gradient_clip_val=1.0,  # Add gradient clipping
                accumulate_grad_batches=2,  # Add gradient accumulation
                enable_progress_bar=True,
                deterministic=True,
            )
            trainer.fit(model=model, datamodule=data_module)  # train and valid
            trainer.test(model=model, datamodule=data_module)  # test


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parse.parse_args()
    main(args)
