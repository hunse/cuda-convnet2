
DATA=(--data-path ~/data/svhn/
      --data-provider svhn
      # --inner-size 0)
      # --inner-size 30)
      --inner-size 28)

# LAYERS=(--layer-def ./layers/layers-svhn-relu.cfg)
# LAYERS+=(--layer-params ./layers/layer-params-svhn-relu.cfg)
LAYERS=(--layer-def ./layers/layers-svhn-lif.cfg)
LAYERS+=(--layer-params ./layers/layer-params-svhn-lif.cfg)

OPTS=(--save-path ./checkpoints
      --gpu 0
      --test-freq 45)

EPOCHS=100

if [[ $1 == "" ]];
then
    stamp=`date +"%y-%m-%d_%H.%M.%S"`
    savefile="./checkpoints/svhn_$stamp"
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
    --train-range 1-9 --test-range 10-12 --epochs ${EPOCHS}
