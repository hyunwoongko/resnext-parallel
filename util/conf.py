"""
@author : Hyunwoong
@when : 2019-10-15
@homepage : https://github.com/gusdnd852
"""

# configuration for basic setting
batch_size = 16
epochs = 1000
weight_decay = 5e-4
cardinality = 32

# configuration for SGD
init_lr = 1e-4 * batch_size
momentum = 0.9

# configuration for callback
monitor = 'val_loss'
factor = 0.8
patience = 5
verbose = 0
mode = 'auto'
min_lr = 1e-10

# configuration for se block
se_ratio = 4

# configuration for storing result
home_dir_windows = 'C:\\Users\\User\\Desktop'
home_dir_linux = '/home/xingshuli/Desktop/'
