
DATA=(--data-path ~/data/mnist-py-colmajor/
      --data-provider mnist
      --inner-size 0)
      # --inner-size 22)

LAYERS=(--layer-def ./layers/layers-mnist-lif.cfg)
LAYERS+=(--layer-params ./layers/layer-params-mnist-lif.cfg)
# LAYERS=(--layer-def ./layers/layers-mnist-1.cfg)
# LAYERS+=(--layer-params ./layers/layer-params-mnist-1.cfg)

OPTS=(--save-path ./checkpoints
      --gpu 0
      --test-freq 30)

EPOCHS=200

if [[ $1 == "" ]];
then
    stamp=`date +"%y-%m-%d_%H.%M.%S"`
    savefile="./checkpoints/mnist_$stamp"
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
    --train-range 1-6 --test-range 7 --epochs ${EPOCHS}
