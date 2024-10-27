set -ex
export CUDA_VISIBLE_DEVICES=$1

EMBEDDING_SIZE=$2

# self-supervised learning pretraining
python -u main.py \
    --exp_dir "exp/2.3/ebd_size_${EMBEDDING_SIZE}" \
    --num_epochs 500 \
    --model.classifier False \
    --model.d_model $EMBEDDING_SIZE \


# supervised learning for classification
python -u main.py \
    --exp_dir "exp/2.3/ebd_size_${EMBEDDING_SIZE}" \
    --num_epochs 1000 \
    --model.d_model $EMBEDDING_SIZE \