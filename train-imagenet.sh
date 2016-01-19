
DATA=(--data-path ~/data/ilsvrc-2012-batches
      --data-provider image
      --inner-size 224)
DATA+=(--mini 128)
DATA+=(--color-noise 0.1)

LAYERS=(--layer-def ./layers/layers-imagenet-1gpu.cfg)
LAYERS+=(--layer-params ./layers/layer-params-imagenet-1gpu.cfg)

OPTS=(--save-path ./checkpoints
      --gpu 0
      --test-freq 201)

EPOCHS=90

if [[ $1 == "" ]];
then
    stamp=`date +"%y-%m-%d_%H.%M.%S"`
    savefile="./checkpoints/ilsvrc2012_ConvNet__$stamp"
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
    --train-range 0-417 --test-range 1000-1016 --epochs ${EPOCHS}
