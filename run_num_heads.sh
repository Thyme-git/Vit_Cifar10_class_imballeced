set -ex
export CUDA_VISIBLE_DEVICES=$1

NUM_HEADS=$2

# self-supervised learning pretraining
python -u main.py \
    --exp_dir "exp/2.3/num_heads_${NUM_HEADS}" \
    --num_epochs 500 \
    --model.classifier False \
    --model.num_heads $NUM_HEADS \


# supervised learning for classification
python -u main.py \
    --exp_dir "exp/2.3/num_heads_${NUM_HEADS}" \
    --num_epochs 1000 \
    --model.num_heads $NUM_HEADS \
