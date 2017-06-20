
DATA=(--data-path ~/data/ilsvrc-2012-batches
      --data-provider image
      --inner-size 224)
DATA+=(--color-noise 0.1)

OPTS=(--save-path ./checkpoints)
#      --test-freq 201)

### one GPU
#LAYERS=(--layer-def ./layers/layers-imagenet-1gpu.cfg)
#DATA+=(--mini 128)
#OPTS+=(--gpu 3)

### four GPU data parallelism
#LAYERS=(--layer-def ./layers/layers-imagenet-4gpu-data.cfg)
#LAYERS=(--layer-def ./layers/layers-imagenet-4gpu-data-relu.cfg)
#LAYERS=(--layer-def ./layers/layers-imagenet-4gpu-data-lif.cfg)
LAYERS=(--layer-def ./layers/layers-imagenet-4gpu-data-lifalpharc.cfg)

DATA+=(--mini 512)
OPTS+=(--gpu 3,2,1,0)

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

### start with small learning rate to get good initial params
python convnet.py "${DATA[@]}" "${OPTS[@]}" "${LAYERS[@]}" \
    --layer-params ./layers/layer-params-imagenet-4gpu-data-smallrate.cfg \
    --train-range 0-25 --test-range 1000-1016 --test-freq 26 --epochs 1
#    --train-range 0-417 --test-range 1000-1016 --test-freq 201 --epochs 1

python convnet.py --load-file "$savefile" \
    --layer-params ./layers/layer-params-imagenet-4gpu-data.cfg \
    --train-range 0-417 --test-range 1000-1016 --test-freq 201 --epochs ${EPOCHS}
