# Copyright 2014 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################


# This script makes batches suitable for training from raw ILSVRC 2012 tar files.

import tarfile
import gzip
from StringIO import StringIO
from random import shuffle
import sys
from time import time
from pyext._MakeDataPyExt import resizeJPEG
import itertools
import os
import cPickle
import scipy.io
import math
import argparse as argp

# Set this to True to crop images to square. In this case each image will be
# resized such that its shortest edge is OUTPUT_IMAGE_SIZE pixels, and then the
# center OUTPUT_IMAGE_SIZE x OUTPUT_IMAGE_SIZE patch will be extracted.
#
# Set this to False to preserve image borders. In this case each image will be
# resized such that its shortest edge is OUTPUT_IMAGE_SIZE pixels. This was
# demonstrated to be superior by Andrew Howard in his very nice paper:
# http://arxiv.org/abs/1312.5402
CROP_TO_SQUARE          = True
OUTPUT_IMAGE_SIZE       = 256

# Number of threads to use for JPEG decompression and image resizing.
NUM_WORKER_THREADS      = 8

# Don't worry about these.
OUTPUT_BATCH_SIZE = 3072
OUTPUT_SUB_BATCH_SIZE = 1024

def pickle(filename, data):
    with open(filename, "w") as fo:
        cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)

def unpickle(filename):
    fo = open(filename, 'r')
    contents = cPickle.load(fo)
    fo.close()
    return contents

def partition_list(l, partition_size):
    divup = lambda a,b: (a + b - 1) / b
    return [l[i*partition_size:(i+1)*partition_size] for i in xrange(divup(len(l),partition_size))]

def open_tar(path, name):
    if not os.path.exists(path):
        print "ILSVRC 2012 %s not found at %s. Make sure to set ILSVRC_SRC_DIR correctly at the top of this file (%s)." % (name, path, sys.argv[0])
        sys.exit(1)
    return tarfile.open(path)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_devkit_meta(ILSVRC_DEVKIT_TAR):
    tf = open_tar(ILSVRC_DEVKIT_TAR, 'devkit tar')
    fmeta = tf.extractfile(tf.getmember('ILSVRC2012_devkit_t12/data/meta.mat'))
    meta_mat = scipy.io.loadmat(StringIO(fmeta.read()))
    labels_dic = dict((m[0][1][0], m[0][0][0][0]-1) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
    label_names_dic = dict((m[0][1][0], m[0][2][0]) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
    label_names = [tup[1] for tup in sorted([(v,label_names_dic[k]) for k,v in labels_dic.items()], key=lambda x:x[0])]

    fval_ground_truth = tf.extractfile(tf.getmember('ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'))
    validation_ground_truth = [[int(line.strip()) - 1] for line in fval_ground_truth.readlines()]
    tf.close()
    return labels_dic, label_names, validation_ground_truth

def write_batches(target_dir, name, start_batch_num, labels, jpeg_files):
    jpeg_files = partition_list(jpeg_files, OUTPUT_BATCH_SIZE)
    labels = partition_list(labels, OUTPUT_BATCH_SIZE)
    makedir(target_dir)
    print "Writing %s batches..." % name
    for i,(labels_batch, jpeg_file_batch) in enumerate(zip(labels, jpeg_files)):
        t = time()
        jpeg_strings = list(itertools.chain.from_iterable(resizeJPEG([jpeg.read() for jpeg in jpeg_file_batch], OUTPUT_IMAGE_SIZE, NUM_WORKER_THREADS, CROP_TO_SQUARE)))
        batch_path = os.path.join(target_dir, 'data_batch_%d' % (start_batch_num + i))
        makedir(batch_path)
        for j in xrange(0, len(labels_batch), OUTPUT_SUB_BATCH_SIZE):
            pickle(os.path.join(batch_path, 'data_batch_%d.%d' % (start_batch_num + i, j/OUTPUT_SUB_BATCH_SIZE)),
                   {'data': jpeg_strings[j:j+OUTPUT_SUB_BATCH_SIZE],
                    'labels': labels_batch[j:j+OUTPUT_SUB_BATCH_SIZE]})
        print "Wrote %s (%s batch %d of %d) (%.2f sec)" % (batch_path, name, i+1, len(jpeg_files), time() - t)
    return i + 1

def make_ilsvrc():
    parser = argp.ArgumentParser()
    parser.add_argument('--src-dir', help='Directory containing ILSVRC2012_img_train.tar, ILSVRC2012_img_val.tar, and ILSVRC2012_devkit_t12.tar.gz', required=True)
    parser.add_argument('--tgt-dir', help='Directory to output ILSVRC 2012 batches suitable for cuda-convnet to train on.', required=True)
    args = parser.parse_args()

    print "CROP_TO_SQUARE: %s" % CROP_TO_SQUARE
    print "OUTPUT_IMAGE_SIZE: %s" % OUTPUT_IMAGE_SIZE
    print "NUM_WORKER_THREADS: %s" % NUM_WORKER_THREADS

    ILSVRC_TRAIN_TAR = os.path.join(args.src_dir, 'ILSVRC2012_img_train.tar')
    ILSVRC_VALIDATION_TAR = os.path.join(args.src_dir, 'ILSVRC2012_img_val.tar')
    ILSVRC_DEVKIT_TAR = os.path.join(args.src_dir, 'ILSVRC2012_devkit_t12.tar.gz')

    assert OUTPUT_BATCH_SIZE % OUTPUT_SUB_BATCH_SIZE == 0
    labels_dic, label_names, validation_labels = parse_devkit_meta(ILSVRC_DEVKIT_TAR)

    with open_tar(ILSVRC_TRAIN_TAR, 'training tar') as tf:
        synsets = tf.getmembers()
        synset_tars = [tarfile.open(fileobj=tf.extractfile(s)) for s in synsets]
        print "Loaded synset tars."
        print "Building training set image list (this can take 10-20 minutes)..."
        sys.stdout.flush()

        train_jpeg_files = []
        for i,st in enumerate(synset_tars):
            if i % 100 == 0:
                print "%d%% ..." % int(round(100.0 * float(i) / len(synset_tars))),
                sys.stdout.flush()
            train_jpeg_files += [st.extractfile(m) for m in st.getmembers()]
            st.close()

        shuffle(train_jpeg_files)
        train_labels = [[labels_dic[jpeg.name[:9]]] for jpeg in train_jpeg_files]
        print "done"

        # Write training batches
        i = write_batches(args.tgt_dir, 'training', 0, train_labels, train_jpeg_files)

    # Write validation batches
    val_batch_start = int(math.ceil((i / 1000.0))) * 1000
    with open_tar(ILSVRC_VALIDATION_TAR, 'validation tar') as tf:
        validation_jpeg_files = sorted([tf.extractfile(m) for m in tf.getmembers()], key=lambda x:x.name)
        write_batches(args.tgt_dir, 'validation', val_batch_start, validation_labels, validation_jpeg_files)

    # Write meta file
    meta = unpickle('input_meta')
    meta_file = os.path.join(args.tgt_dir, 'batches.meta')
    meta.update({'batch_size': OUTPUT_BATCH_SIZE,
                 'num_vis': OUTPUT_IMAGE_SIZE**2 * 3,
                 'label_names': label_names})
    pickle(meta_file, meta)
    print "Wrote %s" % meta_file
    print "All done! ILSVRC 2012 batches are in %s" % args.tgt_dir


def make_cifar100(fine=True):
    import numpy as np

    parser = argp.ArgumentParser()
    parser.add_argument(
        '--src-dir', required=True, help= "Directory containing extracted "
        "python (http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)")
    parser.add_argument(
        '--tgt-dir', required=True, help="Directory for output batches")
    args = parser.parse_args()
    source_dir = args.src_dir
    target_dir = args.tgt_dir
    makedir(target_dir)

    train = np.load(os.path.join(source_dir, 'train'))
    test = np.load(os.path.join(source_dir, 'test'))
    meta = np.load(os.path.join(source_dir, 'meta'))

    ltype = 'fine' if fine else 'coarse'
    lfield = '%s_labels' % ltype

    batch_size = 10000
    image_size = 32
    channels = 3
    num_vis = image_size**2 * channels

    train_data = train['data'].T
    full_data = np.hstack([train_data, test['data'].T])
    full_labels = np.hstack([train[lfield], test[lfield]])
    label_names = meta['%s_label_names' % ltype]
    assert full_data.shape == (num_vis, full_labels.size)

    for i in range(full_data.shape[-1] / batch_size):
        batch_file = os.path.join(target_dir, 'data_batch_%d' % (i + 1))
        pickle(batch_file,
               {'data': full_data[:,i*batch_size:(i+1)*batch_size],
                'labels': full_labels[i*batch_size:(i+1)*batch_size]})
        print "Wrote %s" % batch_file

    # Write meta file
    meta = {'data_mean': train_data.mean(axis=-1, keepdims=True),
            'batch_size': batch_size,
            'img_size': image_size,
            'num_vis': num_vis,
            'label_names': label_names}
    assert meta['num_vis'] == full_data.shape[0]
    meta_file = os.path.join(args.tgt_dir, 'batches.meta')
    pickle(meta_file, meta)
    print "Wrote %s" % meta_file
    print "All done! CIFAR100 batches are in %s" % target_dir


def make_mnist():
    import numpy as np

    parser = argp.ArgumentParser()
    parser.add_argument('--src-dir', help='Directory containing mnist.pkl.gz', required=True)
    parser.add_argument('--tgt-dir', help='Directory for output batches', required=True)
    args = parser.parse_args()
    source_dir = args.src_dir
    target_dir = args.tgt_dir
    makedir(target_dir)

    # Available at http://deeplearning.net/data/mnist/mnist.pkl.gz
    MNIST_DATA = os.path.join(source_dir, "mnist.pkl.gz")

    batch_size = 10000
    image_size = 28
    channels = 1
    num_vis = image_size**2 * channels

    with gzip.open(MNIST_DATA, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)

    label_names = ["%d" % i for i in range(10)]

    full_data = np.hstack([train_set[0].T, valid_set[0].T, test_set[0].T])
    full_labels = np.hstack([train_set[1], valid_set[1], test_set[1]])
    assert full_data.shape == (num_vis, full_labels.size)

    for i in range(full_data.shape[-1] / batch_size):
        batch_file = os.path.join(target_dir, 'data_batch_%d' % (i + 1))
        pickle(batch_file,
               {'data': full_data[:,i*batch_size:(i+1)*batch_size],
                'labels': full_labels[i*batch_size:(i+1)*batch_size]})
        print "Wrote %s" % batch_file

    # Write meta file
    meta = {'data_mean': full_data.mean(axis=-1, keepdims=True),
            'batch_size': batch_size,
            'img_size': image_size,
            'num_vis': num_vis,
            'label_names': label_names}
    meta_file = os.path.join(args.tgt_dir, 'batches.meta')
    pickle(meta_file, meta)
    print "Wrote %s" % meta_file
    print "All done! MNIST batches are in %s" % target_dir


def make_spaun():
    import numpy as np
    rng = np.random.RandomState(9)

    parser = argp.ArgumentParser()
    parser.add_argument('--src-dir', help='Directory containing mnist.pkl.gz', required=True)
    parser.add_argument('--tgt-dir', help='Directory for output batches', required=True)
    args = parser.parse_args()
    source_dir = args.src_dir
    target_dir = args.tgt_dir
    makedir(target_dir)

    # Available at http://deeplearning.net/data/mnist/mnist.pkl.gz
    MNIST_DATA = os.path.join(source_dir, "mnist.pkl.gz")
    SPAUN_DATA = os.path.join(source_dir, "spaun_sym.pkl.gz")

    batch_size = 10000
    image_size = 28
    channels = 1
    num_vis = image_size**2 * channels

    with gzip.open(MNIST_DATA, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
    with gzip.open(SPAUN_DATA, 'rb') as f:
        spaun_set, _, _ = cPickle.load(f)  # 'valid' and 'test' == 'train'

    # Augment spaun data
    x, y = spaun_set[0][10:], spaun_set[1][10:]  # only non-digits

    def aug(data, ratio):
        images, labels = data
        n = images.shape[0] / 10  # approximate examples per label
        na = int(n * ratio)       # examples per augmented category

        xx = np.vstack([images, np.tile(x, (na, 1))])
        yy = np.hstack([labels, np.tile(y, na)])
        i = rng.permutation(len(xx))
        return xx[i], yy[i]

    ratio = 0.2
    train_set = aug(train_set, ratio)
    valid_set = aug(valid_set, ratio)
    test_set = aug(test_set, ratio)

    full_data = np.hstack([train_set[0].T, valid_set[0].T, test_set[0].T])
    full_labels = np.hstack([train_set[1], valid_set[1], test_set[1]])
    assert full_data.shape == (num_vis, full_labels.size)
    print(full_data.shape)

    for i in range(full_data.shape[-1] / batch_size):
        batch_file = os.path.join(target_dir, 'data_batch_%d' % (i + 1))
        pickle(batch_file,
               {'data': full_data[:,i*batch_size:(i+1)*batch_size],
                'labels': full_labels[i*batch_size:(i+1)*batch_size]})
        print "Wrote %s" % batch_file

    # Write meta file
    label_names = ['%d' % i for i in np.unique(full_labels)]
    print(label_names)
    meta = {'data_mean': full_data.mean(axis=-1, keepdims=True),
            'batch_size': batch_size,
            'img_size': image_size,
            'num_vis': num_vis,
            'label_names': label_names}
    meta_file = os.path.join(args.tgt_dir, 'batches.meta')
    pickle(meta_file, meta)
    print "Wrote %s" % meta_file
    print "All done! MNIST batches are in %s" % target_dir


def make_svhn():
    import numpy as np
    import scipy.io.matlab
    rng = np.random.RandomState(8)

    parser = argp.ArgumentParser()
    parser.add_argument('--src-dir', help='Directory containing train_32x32.mat and test_32x32.mat', required=True)
    parser.add_argument('--tgt-dir', help='Directory for output batches', required=True)
    args = parser.parse_args()
    source_dir = args.src_dir
    target_dir = args.tgt_dir
    makedir(target_dir)

    # available at http://ufldl.stanford.edu/housenumbers/
    SVHN_TRAIN_DATA = os.path.join(source_dir, "train_32x32.mat")
    SVHN_TEST_DATA = os.path.join(source_dir, "test_32x32.mat")
    # SVHN_EXTRA_DATA = os.path.join(source_dir, "extra_32x32.mat")

    # batch_size = 10000
    batch_size = 8000
    image_size = 32
    channels = 3
    num_vis = image_size**2 * channels

    def load(path):
        data = scipy.io.matlab.loadmat(path)
        X, y = data['X'], data['y'].ravel()
        X = np.transpose(X, (2, 0, 1, 3)).reshape(-1, X.shape[-1])
        y[y == 10] = 0  # 10 actually means 0
        assert X.shape[-1] == y.size
        print("Loaded %r (%d images)" % (path, y.size))

        # shuffle
        i = rng.permutation(y.size)
        X, y = X[:, i], y[i]

        # clip to multiple of batch size
        n = int(y.size / batch_size) * batch_size
        X, y = X[:, :n], y[:n]
        print("Clipped to %d images" % n)

        return X, y

    train_set = load(SVHN_TRAIN_DATA)
    test_set = load(SVHN_TEST_DATA)
    # extra_set = load(SVHN_EXTRA_DATA)
    label_names = ["%d" % i for i in range(10)]

    full_data = np.hstack([train_set[0], test_set[0]])
    full_labels = np.hstack([train_set[1], test_set[1]])
    assert full_data.shape == (num_vis, full_labels.size)

    for i in range(full_labels.size / batch_size):
        batch_file = os.path.join(target_dir, 'data_batch_%d' % (i + 1))
        pickle(batch_file,
               {'data': full_data[:,i*batch_size:(i+1)*batch_size],
                'labels': full_labels[i*batch_size:(i+1)*batch_size]})
        print "Wrote %s" % batch_file

    # Write meta file
    meta = {'data_mean': full_data.mean(axis=-1, keepdims=True),
            'batch_size': batch_size,
            'img_size': image_size,
            'num_vis': num_vis,
            'label_names': label_names}
    meta_file = os.path.join(args.tgt_dir, 'batches.meta')
    pickle(meta_file, meta)
    print "Wrote %s" % meta_file
    print "All done! SVHN batches are in %s" % target_dir


if __name__ == "__main__":
    # make_ilsvrc()
    # make_cifar100()
    # make_mnist()
    make_spaun()
    # make_svhn()
