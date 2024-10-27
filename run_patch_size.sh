set -ex
export CUDA_VISIBLE_DEVICES=$1

PATCH_SIZE=$2

# self-supervised learning pretraining
python -u main.py \
    --exp_dir "exp/2.3/patch_size_${PATCH_SIZE}" \
    --num_epochs 500 \
    --model.classifier False \
    --model.dropout 0.0 \
    --model.patch_size $PATCH_SIZE \


# supervised learning for classification
python -u main.py \
    --exp_dir "exp/2.3/patch_size_${PATCH_SIZE}" \
    --num_epochs 1000 \
    --model.patch_size $PATCH_SIZE \