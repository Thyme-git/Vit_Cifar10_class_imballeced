set -ex
export CUDA_VISIBLE_DEVICES=$1
optimized=$2


if [ -z "$optimized" ]; then
    echo "Using Vanilla Algorithm"
    python -u main.py \
        --exp_dir "exp/2.2/vanilla" \
        --num_epochs 500 \
        --data.augment False \
        --loss_augment False \

else
    echo "Using Optimized Algorithm"
    python -u main.py \
        --exp_dir "exp/2.2/optimized" \
        --num_epochs 500 \
        --model.classifier False \


    # supervised learning for classification
    python -u main.py \
        --exp_dir "exp/2.2/optimized" \
        --num_epochs 1000 \

fi

