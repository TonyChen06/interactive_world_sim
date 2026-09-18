#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR=${1:?Pass the full PushT dataset directory}
AE_CHECKPOINT=${2:?Pass the released PushT or Stage 1 autoencoder checkpoint}
HISTORY_STEPS=${3:-0}  # 0: control; 1 or 3: generated history
case "$HISTORY_STEPS" in
  0) GENERATED_HISTORY=false ;;
  1|3) GENERATED_HISTORY=true ;;
  *) echo "History steps must be 0, 1, or 3" >&2; exit 2 ;;
esac

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -c 'import lightning.pytorch as pl; import runpy; pl.seed_everything(42, workers=True); runpy.run_path("main.py", run_name="__main__")' \
  +name="pusht_history_${HISTORY_STEPS}" algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=real_aloha_dataset \
  dataset.dataset_dir="$DATASET_DIR" \
  dataset.horizon=10 dataset.val_horizon=200 \
  dataset.obs_keys=[camera_1_color] \
  dataset.action_mode=bimanual_push \
  experiment.training.batch_size=4 \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.limit_batch=1.0 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=30000 \
  experiment.training.checkpointing.every_n_train_steps=10000 \
  experiment.training.data.num_workers=4 \
  experiment.validation.data.num_workers=4 \
  algorithm.latent_dim=512 algorithm.action_dim=4 \
  algorithm.noise_scheduler.loss_weighting=uniform \
  algorithm.sampling_strategy=terminal_only \
  algorithm.load_ae="$AE_CHECKPOINT" \
  algorithm.training_stage=2 algorithm.generated_history="$GENERATED_HISTORY" \
  algorithm.generated_history_steps="$HISTORY_STEPS" \
  wandb.mode=offline wandb.entity=local
