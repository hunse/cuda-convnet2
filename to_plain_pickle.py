from nengo.utils.compat import pickle

from python_util.gpumodel import IGPUModel


def plain_pickle(loadfile, savefile):
    load_dic = IGPUModel.load_checkpoint(loadfile)

    options = {}
    for o in load_dic['op'].get_options_list():
        options[o.name] = o.value
    load_dic['op'] = options

    with open(savefile, 'wb') as f:
        pickle.dump(load_dic, f, protocol=-1)
    print("Wrote %r" % savefile)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert checkpoint to plain pickle file")
    parser.add_argument('loadfile', help="Checkpoint to load")
    parser.add_argument('savefile', help="Pickle file to save")
    args = parser.parse_args()

    plain_pickle(args.loadfile, args.savefile)
