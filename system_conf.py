import os
from keras import backend as K

import warnings 

def supress_warnings():
	warnings.filterwarnings("ignore")


def gpu_conf(gpu_id =0, 
			 load = 0.8):
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

	if K._backend=='tensorflow':
	    print 'Set GPU:'
	    import tensorflow as tf
	    gpu_load = load

	    tf.device('/gpu:' + str(gpu_id))
	    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_load)
	    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	    #sess=tf.Session()
	    K.set_session(sess)

	    print "This session will use GPU " + str(gpu_id) + " and  " + str(load) + "of the available GPU"

