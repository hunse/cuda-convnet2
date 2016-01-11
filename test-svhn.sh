python convnet.py --load-file "$1" \
    --multiview-test 0 --test-only 1 --logreg-name logprob --test-range 10
python convnet.py --load-file "$1" \
    --multiview-test 1 --test-only 1 --logreg-name logprob --test-range 10
