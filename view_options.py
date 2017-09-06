""" View the options used to create a checkpoint

    python view_options.py --load-file <checkpoint>
"""
from convnet import ConvNet
from python_util.gpumodel import IGPUModel

op = ConvNet.get_options_parser()

op, load_dic = IGPUModel.parse_options(op)
model = ConvNet(op, load_dic)

model.op.print_values()
print "========================="
model.print_model_state()
