
DATA=(--data-provider dummy-labeled-192)

# LAYERS=(--layer-def ./layers/layers-grad.cfg
#         --layer-params ./layers/layer-params-grad.cfg)

LAYERS=(--layer-def ./layers/layers-neuron-grad.cfg
        --layer-params ./layers/layer-params-neuron-grad.cfg)

OPTS=(--gpu 0)

ipython --pdb convnet.py -- "${DATA[@]}" "${LAYERS[@]}" "${OPTS[@]}" \
    --check-grads 1
