
DATA=(--data-path ~/data/spaun-py-colmajor/
      --data-provider mnist
      --inner-size 0)
      # --inner-size 26)
      # --inner-size 22)

# LAYERS=(--layer-def ./layers/layers-mnist-relu.cfg)
# LAYERS+=(--layer-params ./layers/layer-params-mnist-relu.cfg)
LAYERS=(--layer-def ./layers/layers-spaun-lif.cfg)
LAYERS+=(--layer-params ./layers/layer-params-mnist-lif.cfg)

OPTS=(--save-path ./checkpoints
      --gpu 0
      --test-freq 30)

EPOCHS=200

if [[ $1 == "" ]];
then
    stamp=`date +"%y-%m-%d_%H.%M.%S"`
    savefile="./checkpoints/spaun_$stamp"
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
