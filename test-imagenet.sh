python convnet.py --load-file "$1" \
    --multiview-test 0 --test-only 1
python convnet.py --load-file "$1" \
    --multiview-test 1 --test-only 1
