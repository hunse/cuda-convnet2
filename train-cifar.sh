DATA=(
    --data-path ~/data/cifar-10-py-colmajor/
    --data-provider cifar
    --inner-size 24)
    # --inner-size 0)

LAYERS=(--layer-def ./layers/cifar10-single.cfg)
LAYERS+=(--layer-params ./layers/cifar10-single-params.cfg)
# LAYERS=(--layer-def ./layers/cifar10-double.cfg)
# LAYERS+=(--layer-params ./layers/cifar10-double-params.cfg)

OPTS=(--save-path ./checkpoints
      --gpu 0
      --test-freq 30)

EPOCHS=50
# EPOCHS=100
# EPOCHS=200

if [[ $1 == "" ]];
then
    stamp=`date +"%y-%m-%d_%H.%M.%S"`
    savefile="./checkpoints/cifar10_$stamp"
else
    savefile="$1"
fi

if [ -e $savefile ];
then
    OPTS+=(--load-file "$savefile")
else
    OPTS+=(--save-file "$savefile")
fi

python convnet.py "${DATA[@]}" "${LAYERS[@]}" "${OPTS[@]}" \
    --train-range 1-5 --test-range 6 --epochs ${EPOCHS}
