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

stamp=`date +"%y-%m-%d_%H.%M.%S"`
savefile="./checkpoints/cifar10_ConvNet__$stamp"
OPTS+=(--save-file "$savefile")

python convnet.py "${DATA[@]}" "${LAYERS[@]}" "${OPTS[@]}" \
    --train-range 1-5 --test-range 6 --epochs 1 \
    --init-only 1 --logreg-name logprob
