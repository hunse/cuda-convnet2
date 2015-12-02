
DATA=(
    --data-path ~/data/cifar-100-py-colmajor/
    # --data-path ~/data/cifar-100-whitened/
    --data-provider cifar
    --inner-size 24)
# DATA+=(--color-noise 0.1)

# LAYERS=(--layer-def ./layers/layers-cifar100-avg.cfg)
LAYERS=(--layer-def ./layers/layers-cifar100-lif.cfg)
# LAYERS=(--layer-def ./layers/cifar-100_def.cfg)

LAYERS+=(--layer-params ./layers/layer-params-cifar10-11pct.cfg)
# LAYERS+=(--layer-params ./layers/cifar-100_params.cfg)

OPTS=(--save-path ./checkpoints
      --gpu 0
      --test-freq 25)

# EPOCHS=(100 140 150 160)  # 13% error
# EPOCHS=(200 280 300 320)
EPOCHS=(350 500 510 520)  # should be 11% error


if [[ $1 == "" ]];
then
    stamp=`date +"%y-%m-%d_%H.%M.%S"`
    savefile="./checkpoints/cifar100_ConvNet__$stamp"
else
    savefile="$1"
fi

if [ -e $savefile ];
then
    OPTS+=(--load-file "$savefile")
else
    OPTS+=(--save-file "$savefile")
fi

# train on sets 1-4
python convnet.py "${DATA[@]}" "${LAYERS[@]}" "${OPTS[@]}" \
    --train-range 1-4 --test-range 6 --epochs ${EPOCHS[0]}

# fold in fifth set for training
python convnet.py --load-file "$savefile" \
    --train-range 1-5 --test-range 6 --epochs ${EPOCHS[1]}

# divide learning rates by 10
python convnet.py --load-file "$savefile" \
    --layer-params ./layers/layer-params-cifar10-11pct-eps10.cfg \
    --train-range 1-5 --test-range 6 --epochs ${EPOCHS[2]}

# divide learning rates by 10 again
python convnet.py --load-file "$savefile" \
    --layer-params ./layers/layer-params-cifar10-11pct-eps100.cfg \
    --train-range 1-5 --test-range 6 --epochs ${EPOCHS[3]}

# test with shifts
python convnet.py --load-file "$savefile" \
    --multiview-test 1 --test-only 1 --logreg-name logprob --test-range 6
