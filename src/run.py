import argparse
from data_preprocess.data_builder import SummaryDataModule
from models.bart import BartOrigin
from models.t5 import T5Origin, T5MultiModal
from models.multi_modal_model import BartMultiModal
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy as DDPPlugin
import os
import glob


def find_all_checkpoints(log_name):
    """
    Find all checkpoints for a given model.
    Handles the nested path: lightning_logs/{name}/lightning_logs/version_X/checkpoints/
    """
    patterns = [
        f"./lightning_logs/{log_name}/lightning_logs/version_*/checkpoints/*.ckpt",
        f"./lightning_logs/{log_name}/version_*/checkpoints/*.ckpt",
    ]
    ckpt_files = []
    for pattern in patterns:
        ckpt_files.extend(glob.glob(pattern))
    return ckpt_files


def find_last_checkpoint(log_name):
    """Find the last.ckpt for resuming interrupted training."""
    ckpt_files = find_all_checkpoints(log_name)
    if not ckpt_files:
        return None
    
    last_ckpts = [f for f in ckpt_files if "last.ckpt" in f]
    if last_ckpts:
        return max(last_ckpts, key=os.path.getmtime)
    return None


def find_best_checkpoint(log_name):
    """Find the best (non-last) checkpoint for testing."""
    ckpt_files = find_all_checkpoints(log_name)
    if not ckpt_files:
        return None
    
    best_ckpts = [f for f in ckpt_files if "last" not in f]
    if best_ckpts:
        return max(best_ckpts, key=os.path.getmtime)
    return None


def is_training_complete(log_name):
    """Check if training completed (has best checkpoint with high epoch number)."""
    best = find_best_checkpoint(log_name)
    if best is None:
        return False
    # Extract epoch number from filename like "epoch=198-step=4179.ckpt"
    basename = os.path.basename(best)
    try:
        epoch = int(basename.split("epoch=")[1].split("-")[0])
        return epoch > 20  # If best checkpoint is past epoch 20, training likely completed
    except (IndexError, ValueError):
        return True  # Can't parse, but checkpoint exists


if __name__ == '__main__':
    # for training
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default='text_only_bart', type=str, help='We have for models to choose, text_only_bart, multi_modal_bart,  text_only_t5 and multi_modal_t5')
    parser.add_argument('-checkpoint', default='./lightning_logs/version_7/checkpoints/epoch=29-step=4499.ckpt', type=str, help='The checkpoint path')
    parser.add_argument('-train_src_path', default='./dataset/sum_train/tran.tok.txt', type=str, help='training input data path (dialogue)')
    parser.add_argument('-train_tgt_path', default='./dataset/sum_train/desc.tok.txt', type=str, help='training output data path (summary)')
    parser.add_argument('-val_src_path', default='./dataset/sum_cv/tran.tok.txt', type=str, help='validatioin input data path (dialogue)')
    parser.add_argument('-val_tgt_path', default='./dataset/sum_cv/desc.tok.txt', type=str, help='validatioin output data path (summary)')
    parser.add_argument('-test_src_path', default='./dataset/sum_devtest/tran.tok.txt', type=str, help='testing input data path (dialogue)')
    parser.add_argument('-test_tgt_path', default='./dataset/sum_devtest/desc.tok.txt', type=str, help='testing output data path (summary)')
    parser.add_argument('-image_feature_path', default='./dataset/video_action_features/', type=str, help='video features path')
    parser.add_argument('-val_save_file', default='./evaluation/temp_valid_file', type=str, help='the validation results for each epoch')
    parser.add_argument('-test_save_file', default='./evaluation/results/test_summaries.txt', type=str, help='the generated summary for testing data')
    parser.add_argument('-log_name', default='multi_modal_bart', type=str, help='lightning log path')
    parser.add_argument('-gpus', default='0,2,3,4', type=str, help='choose gpus to run the code, you can choose multipple gpus')
    parser.add_argument('-batch_size', type=int, default=4, help='batch size for each gpu')
    parser.add_argument('-max_input_len', type=int, default=512, help='the maximun length for input dialogue')
    parser.add_argument('-max_output_len', type=int, default=64, help='the maximun length for output summary')
    parser.add_argument('-max_img_len', type=int, default=256, help='the maximun length for video features')
    parser.add_argument('-n_beams', type=int, default=4, help='the number of beams using for generation')
    parser.add_argument('-no_repeat_ngram_size', type=int, default=3, help='the size of no repeat ngrams during generation')
    parser.add_argument('-learning_rate', default=3e-5, type=float, help='learning rate')
    parser.add_argument('-scheduler_lambda1', default=20, type=int, help='change the learning each lambda1 epoch')
    parser.add_argument('-scheduler_lambda2', default=0.95, type=float, help='the learning rate will times lambda2 for each change')
    parser.add_argument('-num_epochs', type=int, default=100, help='maximun number of training epoches')
    parser.add_argument('-grad_accumulate', type=int, default=10, help='gradient accumulation for this number iterations')
    parser.add_argument('-random_seed', type=int, default=0, help='global random seed')
    parser.add_argument('-do_train', type=str, default='True', help='set True to training, set False to not training')
    parser.add_argument('-do_test', type=str, default='True', help='set True to testing, set False to not testing')
    parser.add_argument('-limit_val_batches', default=1.0, type=float, help='do validation for each epoch')
    parser.add_argument('-val_check_interval', type=float, default=1.0, help='do validation for each epoch')
    parser.add_argument('-img_lr_factor', type=float, default=1, help='the learning rate for visual guidance part will times this number')

    # About cross-modal attention and fusion
    parser.add_argument('-use_img_trans', action='store_true', help='whether or not to use VTF')
    parser.add_argument('-use_forget_gate', action='store_true', help='whether or not to use forget gate')
    parser.add_argument('-fusion_layer', type=int, default=5, help='number of fusion layers')
    parser.add_argument('-cross_attn_type', type=int, default=0)
    parser.add_argument('-dim_common', type=int, default=256)
    parser.add_argument('-n_attn_heads', type=int, default=1)

    # Add to decoding
    parser.add_argument('-fusion_in_decoding', action='store_true')
    parser.add_argument('-vision_use_noise', action='store_true')

    # CLIP-related arguments
    parser.add_argument('-use_clip', action='store_true', 
                       help='Use CLIP features (512-dim) instead of ResNet features (2048-dim)')
    parser.add_argument('-visual_hidden_size', type=int, default=512,
                       help='Visual feature dimension: 512 for CLIP, 768 for BLIP, 2048 for ResNet')
    parser.add_argument('-clip_model', type=str, default='openai/clip-vit-base-patch32',
                       help='CLIP model name for end-to-end training')
    parser.add_argument('-freeze_clip', action='store_true',
                       help='Freeze CLIP weights during training')
    parser.add_argument('-video_dir', type=str, default='./dataset/videos',
                       help='Directory containing video files')
    parser.add_argument('-num_frames', type=int, default=50,
                       help='Number of frames to sample per video')

    # Auto-resume and skip controls
    parser.add_argument('-auto_resume', type=str, default='True',
                       help='Automatically resume from last checkpoint if interrupted')
    parser.add_argument('-skip_if_trained', type=str, default='True',
                       help='Skip training if best checkpoint already exists')

    args = parser.parse_args()

    # random seed
    seed_everything(args.random_seed)

    # ================================================================
    # CHECKPOINT MANAGEMENT
    # ================================================================
    
    # Check if training already completed — skip if so
    if args.do_train == 'True' and args.skip_if_trained == 'True':
        if is_training_complete(args.log_name):
            best = find_best_checkpoint(args.log_name)
            print("=" * 70)
            print(f"SKIPPING: {args.log_name} (already trained)")
            print(f"  Best checkpoint: {best}")
            print(f"  Use -skip_if_trained False to force retrain")
            print("=" * 70)
            if args.do_test != 'True':
                exit(0)
            else:
                args.do_train = 'False'
                args.checkpoint = best

    # Auto-resume from last checkpoint if interrupted
    resume_ckpt = None
    if args.do_train == 'True' and args.auto_resume == 'True':
        last_ckpt = find_last_checkpoint(args.log_name)
        if last_ckpt and (args.checkpoint == 'None' or args.checkpoint is None):
            resume_ckpt = last_ckpt
            print("=" * 70)
            print(f"RESUMING: {args.log_name}")
            print(f"  From: {resume_ckpt}")
            print("=" * 70)

    # set logger
    logger = pl_loggers.TensorBoardLogger(f'./lightning_logs/{args.log_name}')

    # save checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='validation_Rouge2_one_epoch',
        save_last=True,
        save_top_k=2,
        mode='max',
        filename='{epoch}-{validation_Rouge2_one_epoch:.4f}',
    )

    # early stopping
    early_stop_callback = EarlyStopping(
        monitor='validation_Rouge2_one_epoch',
        patience=15,
        mode='max',
        verbose=True,
    )

    # make trainer
    if args.checkpoint == 'None':
        args.checkpoint = None
    
    trainer = Trainer(
        deterministic=True,
        num_sanity_val_steps=10,
        logger=logger,
        devices=args.gpus,
        strategy=DDPPlugin(find_unused_parameters=False),
        gradient_clip_val=1.0,
        max_epochs=args.num_epochs,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=args.grad_accumulate,
        fast_dev_run=False,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    # make dataloader & model
    summary_data = SummaryDataModule(args)
    if args.model == 'text_only_bart':
        model = BartOrigin(args)
    elif args.model == 'multi_modal_bart':
        model = BartMultiModal(args)
    elif args.model == 'text_only_t5':
        model = T5Origin(args)
    elif args.model == 'multi_modal_t5':
        model = T5MultiModal(args)
    else:
        raise ValueError("Invalid model")

    # Train
    if args.do_train == 'True':
        ckpt = resume_ckpt if resume_ckpt else args.checkpoint
        print(f"Training {args.log_name} | ckpt: {ckpt}")
        trainer.fit(model, summary_data, ckpt_path=ckpt)
    
    # Test
    if args.do_test == 'True':
        test_ckpt = args.checkpoint
        if test_ckpt is None:
            test_ckpt = find_best_checkpoint(args.log_name)
        
        if test_ckpt:
            print(f"Testing {args.log_name} | ckpt: {test_ckpt}")
            model = BartMultiModal.load_from_checkpoint(test_ckpt, args=args)
            trainer.test(model=model, dataloaders=summary_data.test_loader)
        else:
            print(f"ERROR: No checkpoint found for testing {args.log_name}")