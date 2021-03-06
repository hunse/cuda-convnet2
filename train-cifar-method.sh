
DATA=(--data-path ~/data/cifar-10-py-colmajor/
      --data-provider cifar
      --inner-size 24)
# DATA+=(--color-noise 0.1)

# LAYERS=(--layer-def ./layers/layers-cifar10-11pct.cfg)
# LAYERS=(--layer-def ./layers/layers-cifar10-max.cfg)
# LAYERS=(--layer-def ./layers/layers-cifar10-relu.cfg)
# LAYERS=(--layer-def ./layers/layers-cifar10-dropout.cfg)
# LAYERS=(--layer-def ./layers/layers-cifar10-lif.cfg)
# LAYERS=(--layer-def ./layers/layers-cifar10-lif-noise10.cfg)
# LAYERS=(--layer-def ./layers/layers-cifar10-lifnoise2.cfg)
# LAYERS=(--layer-def ./layers/layers-cifar10-lif-sparse.cfg)
# LAYERS=(--layer-def ./layers/layers-cifar10-lifalpha-5ms.cfg)
# LAYERS=(--layer-def ./layers/layers-cifar10-lifalpha-3ms.cfg)
# LAYERS=(--layer-def ./layers/layers-cifar10-lifalpharc.cfg)
LAYERS=(--layer-def ./layers/layers-cifar10-lifalpharc-5ms.cfg)

LAYERS+=(--layer-params ./layers/layer-params-cifar10-11pct.cfg)
# LAYERS+=(--layer-params ./layers/layer-params-cifar10-lifnoise2.cfg)
# LAYERS+=(--layer-params ./layers/layer-params-cifar10-softlifalpha.cfg)
# LAYERS+=(--layer-params ./hyperopt_output/cifar10_2017-03-09_16.14.17_params.cfg)

OPTS=(--save-path ./checkpoints
      --gpu 0
      --test-freq 25)

# EPOCHS=(100 140 150 160)  # 13% error
# EPOCHS=(200 280 300 320)
EPOCHS=(350 500 510 520)  # should be 11% error
# EPOCHS=(350 600 630 650)  # 12% error?


if [[ $1 == "" ]];
then
    stamp=`date +"%y-%m-%d_%H.%M.%S"`
    savefile="./checkpoints/cifar10_ConvNet__$stamp"
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
# ipython --pdb convnet.py -- "${DATA[@]}" "${LAYERS[@]}" "${OPTS[@]}" \
#     --train-range 1-4 --test-range 6 --epochs ${EPOCHS[0]}
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
