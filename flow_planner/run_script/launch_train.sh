export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=null
export HYDRA_FULL_ERROR=1
export PROJECT_ROOT= # path to the repo
export SAVE_DIR= # path to save training results
export TENSORBOARD_LOG_PATH= # path to save tensorboard logs
export TRAINING_DATA= # path to the training data npz
export TRAINING_JSON= # path to the training data list json
export TORCH_LOGS="dynamic,recompiles"

python -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone ../trainer.py --config-name flow_planner_standard
