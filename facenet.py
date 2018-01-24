import tensorflow as tf
import numpy as np
import os
from numpy import genfromtxt
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
# import fr_utils
from keras.layers.core import Lambda, Flatten, Dense

from termcolor import colored
from system_conf import gpu_conf
import h5py 


gpu_conf(gpu_id =0, 
			 load = 0.1)

data_format = 'channels_first'

K.set_image_data_format(data_format)

# K.set_image_data_format(data_format)
# K.set_image_dim_ordering('th')


def conv2d_bn(x,
              layer=None,
              cv1_out=None,
              cv1_filter=(1, 1),
              cv1_strides=(1, 1),
              cv2_out=None,
              cv2_filter=(3, 3),
              cv2_strides=(1, 1),
              padding=None):
    num = '' if cv2_out == None else '1'
    tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format=data_format, name=layer+'_conv'+num)(x)
    tensor = BatchNormalization(axis=1, epsilon=0.00001, name=layer+'_bn'+num)(tensor)
    tensor = Activation('relu')(tensor)
    if padding == None:
        return tensor
    tensor = ZeroPadding2D(padding=padding, data_format=data_format)(tensor)
    if cv2_out == None:
        return tensor
    tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, data_format=data_format, name=layer+'_conv'+'2')(tensor)
    tensor = BatchNormalization(axis=1, epsilon=0.00001, name=layer+'_bn'+'2')(tensor)
    tensor = Activation('relu')(tensor)
    return tensor


def inception_block_1a(X):
    """
    Implementation of an inception block
    """
    
    X_3x3 = Conv2D(96, (1, 1), data_format=data_format, name ='inception_3a_3x3_conv1')(X)
    X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name = 'inception_3a_3x3_bn1')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)
    X_3x3 = ZeroPadding2D(padding=(1, 1), data_format=data_format)(X_3x3)
    X_3x3 = Conv2D(128, (3, 3), data_format=data_format, name='inception_3a_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_3x3_bn2')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)
    
    X_5x5 = Conv2D(16, (1, 1), data_format=data_format, name='inception_3a_5x5_conv1')(X)
    X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_5x5_bn1')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)
    X_5x5 = ZeroPadding2D(padding=(2, 2), data_format=data_format)(X_5x5)
    X_5x5 = Conv2D(32, (5, 5), data_format=data_format, name='inception_3a_5x5_conv2')(X_5x5)
    X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_5x5_bn2')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)

    X_pool = MaxPooling2D(pool_size=3, strides=2, data_format=data_format)(X)
    X_pool = Conv2D(32, (1, 1), data_format=data_format, name='inception_3a_pool_conv')(X_pool)
    X_pool = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_pool_bn')(X_pool)
    X_pool = Activation('relu')(X_pool)
    X_pool = ZeroPadding2D(padding=((3, 4), (3, 4)), data_format=data_format)(X_pool)

    X_1x1 = Conv2D(64, (1, 1), data_format=data_format, name='inception_3a_1x1_conv')(X)
    X_1x1 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_1x1_bn')(X_1x1)
    X_1x1 = Activation('relu')(X_1x1)
        
    # CONCAT
    inception = concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis=1)

    return inception

def inception_block_1b(X):
    X_3x3 = Conv2D(96, (1, 1), data_format=data_format, name='inception_3b_3x3_conv1')(X)
    X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_3x3_bn1')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)
    X_3x3 = ZeroPadding2D(padding=(1, 1), data_format=data_format)(X_3x3)
    X_3x3 = Conv2D(128, (3, 3), data_format=data_format, name='inception_3b_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_3x3_bn2')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)

    X_5x5 = Conv2D(32, (1, 1), data_format=data_format, name='inception_3b_5x5_conv1')(X)
    X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_5x5_bn1')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)
    X_5x5 = ZeroPadding2D(padding=(2, 2), data_format=data_format)(X_5x5)
    X_5x5 = Conv2D(64, (5, 5), data_format=data_format, name='inception_3b_5x5_conv2')(X_5x5)
    X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_5x5_bn2')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)

    X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format=data_format)(X)
    X_pool = Conv2D(64, (1, 1), data_format=data_format, name='inception_3b_pool_conv')(X_pool)
    X_pool = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_pool_bn')(X_pool)
    X_pool = Activation('relu')(X_pool)
    X_pool = ZeroPadding2D(padding=(4, 4), data_format=data_format)(X_pool)

    X_1x1 = Conv2D(64, (1, 1), data_format=data_format, name='inception_3b_1x1_conv')(X)
    X_1x1 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_1x1_bn')(X_1x1)
    X_1x1 = Activation('relu')(X_1x1)

    inception = concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis=1)

    return inception

def inception_block_1c(X):
    X_3x3 = conv2d_bn(X,
                           layer='inception_3c_3x3',
                           cv1_out=128,
                           cv1_filter=(1, 1),
                           cv2_out=256,
                           cv2_filter=(3, 3),
                           cv2_strides=(2, 2),
                           padding=(1, 1))

    X_5x5 = conv2d_bn(X,
                           layer='inception_3c_5x5',
                           cv1_out=32,
                           cv1_filter=(1, 1),
                           cv2_out=64,
                           cv2_filter=(5, 5),
                           cv2_strides=(2, 2),
                           padding=(2, 2))

    X_pool = MaxPooling2D(pool_size=3, strides=2, data_format=data_format)(X)
    X_pool = ZeroPadding2D(padding=((0, 1), (0, 1)), data_format=data_format)(X_pool)

    inception = concatenate([X_3x3, X_5x5, X_pool], axis=1)

    return inception

def inception_block_2a(X):
    X_3x3 = conv2d_bn(X,
                           layer='inception_4a_3x3',
                           cv1_out=96,
                           cv1_filter=(1, 1),
                           cv2_out=192,
                           cv2_filter=(3, 3),
                           cv2_strides=(1, 1),
                           padding=(1, 1))
    X_5x5 = conv2d_bn(X,
                           layer='inception_4a_5x5',
                           cv1_out=32,
                           cv1_filter=(1, 1),
                           cv2_out=64,
                           cv2_filter=(5, 5),
                           cv2_strides=(1, 1),
                           padding=(2, 2))

    X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format=data_format)(X)
    X_pool = conv2d_bn(X_pool,
                           layer='inception_4a_pool',
                           cv1_out=128,
                           cv1_filter=(1, 1),
                           padding=(2, 2))
    X_1x1 = conv2d_bn(X,
                           layer='inception_4a_1x1',
                           cv1_out=256,
                           cv1_filter=(1, 1))
    inception = concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis=1)

    return inception

def inception_block_2b(X):
    #inception4e
    X_3x3 = conv2d_bn(X,
                           layer='inception_4e_3x3',
                           cv1_out=160,
                           cv1_filter=(1, 1),
                           cv2_out=256,
                           cv2_filter=(3, 3),
                           cv2_strides=(2, 2),
                           padding=(1, 1))
    X_5x5 = conv2d_bn(X,
                           layer='inception_4e_5x5',
                           cv1_out=64,
                           cv1_filter=(1, 1),
                           cv2_out=128,
                           cv2_filter=(5, 5),
                           cv2_strides=(2, 2),
                           padding=(2, 2))
    
    X_pool = MaxPooling2D(pool_size=3, strides=2, data_format=data_format)(X)
    X_pool = ZeroPadding2D(padding=((0, 1), (0, 1)), data_format=data_format)(X_pool)

    inception = concatenate([X_3x3, X_5x5, X_pool], axis=1)

    return inception

def inception_block_3a(X):
    X_3x3 = conv2d_bn(X,
                           layer='inception_5a_3x3',
                           cv1_out=96,
                           cv1_filter=(1, 1),
                           cv2_out=384,
                           cv2_filter=(3, 3),
                           cv2_strides=(1, 1),
                           padding=(1, 1))
    X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format=data_format)(X)
    X_pool = conv2d_bn(X_pool,
                           layer='inception_5a_pool',
                           cv1_out=96,
                           cv1_filter=(1, 1),
                           padding=(1, 1))
    X_1x1 = conv2d_bn(X,
                           layer='inception_5a_1x1',
                           cv1_out=256,
                           cv1_filter=(1, 1))

    inception = concatenate([X_3x3, X_pool, X_1x1], axis=1)

    return inception

def inception_block_3b(X):
    X_3x3 = conv2d_bn(X,
                           layer='inception_5b_3x3',
                           cv1_out=96,
                           cv1_filter=(1, 1),
                           cv2_out=384,
                           cv2_filter=(3, 3),
                           cv2_strides=(1, 1),
                           padding=(1, 1))
    X_pool = MaxPooling2D(pool_size=3, strides=2, data_format=data_format)(X)
    X_pool = conv2d_bn(X_pool,
                           layer='inception_5b_pool',
                           cv1_out=96,
                           cv1_filter=(1, 1))
    X_pool = ZeroPadding2D(padding=(1, 1), data_format=data_format)(X_pool)

    X_1x1 = conv2d_bn(X,
                           layer='inception_5b_1x1',
                           cv1_out=256,
                           cv1_filter=(1, 1))
    inception = concatenate([X_3x3, X_pool, X_1x1], axis=1)

    return inception

def faceRecoModel(input_shape):
    """
    Implementation of the Inception model used for FaceNet
    
    Arguments:
    input_shape -- shape of the images of the dataset
    Returns:
    model -- a Model() instance in Keras
    """
        
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # First Block
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(X)
    X = BatchNormalization(axis = 1, name = 'bn1')(X)
    X = Activation('relu')(X)
    
    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D((3, 3), strides = 2)(X)
    
    # Second Block
    X = Conv2D(64, (1, 1), strides = (1, 1), name = 'conv2')(X)
    X = BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)

    # Second Block
    X = Conv2D(192, (3, 3), strides = (1, 1), name = 'conv3')(X)
    X = BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn3')(X)
    X = Activation('relu')(X)
    
    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D(pool_size = 3, strides = 2)(X)
    
    # Inception 1: a/b/c
    X = inception_block_1a(X)
    X = inception_block_1b(X)
    X = inception_block_1c(X)
    
    # Inception 2: a/b
    X = inception_block_2a(X)
    X = inception_block_2b(X)
    
    # Inception 3: a/b
    X = inception_block_3a(X)
    X = inception_block_3b(X)
    
    # Top layer
    X = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format=data_format)(X)
    X = Flatten()(X)
    X = Dense(128, name='dense_layer')(X)
    
    # L2 normalization
    X = Lambda(lambda  x: K.l2_normalize(x,axis=1))(X)

    # Create model instance
    model = Model(inputs = X_input, outputs = X, name='FaceRecoModel')
        
    return model


model = faceRecoModel(input_shape = (3, 96,96))

def get_layer_names(model, layer_names_file="facenet_layer_names.npy"):
	layer_names = [] 
	for layers in model.layers:
		layer_names.append(layers.name)

	layer_names_mod = [] 

	for name in layer_names:

		if 'bn' in name:
			continue 
		if 'conv' in name:
			layer_names_mod.append(name)

		if 'inception' in name:
			layer_names_mod.append(name)
			
		if 'bn' in name:
			layer_names_mod.append(name)
			
		

	facenet_layers = {}
	cnt =0 
	for name in layer_names_mod:
		cnt +=1 
		facenet_layers[name] = cnt

	np.save(layer_names_file, facenet_layers)

	print ("Layer Names written to file : ", layer_names_file)
	return layer_names_file



def get_layer_names_all(model, layer_names_file="facenet_layer_names_all.npy"):
	layer_names = [] 
	for layers in model.layers:
		layer_names.append(layers.name)

	layer_names_mod = [] 

	for name in layer_names:

		if 'bn' in name:
			layer_names_mod.append(name) 
		if 'conv' in name:
			layer_names_mod.append(name)

		if 'inception' in name:
			layer_names_mod.append(name)
			
		

	facenet_layers = {}
	cnt =0 
	for name in layer_names_mod:
		cnt +=1 
		facenet_layers[name] = cnt

	np.save(layer_names_file, facenet_layers)

	print ("All layer Names written to file : ", layer_names_file)
	return layer_names_file



#saved to facenet_layers.npy

# ["kernel_size"][0]

def get_layer_shapes(model, layer_names_file, layer_shapes_file = "facenet_layer_shapes.npy"):

	layer_names_dict = np.load(layer_names_file).item()

	layer_shape_dict = {}
	for layer_name in layer_names_dict:

		# print layer_name
		
		kernel_size = model.get_layer(layer_name).get_config()["kernel_size"][0]
		# print kernel_size
		input_channels = model.get_layer(layer_name).input_shape[1]
		output_channels = model.get_layer(layer_name).output_shape[1]
		layer_shape = (output_channels, input_channels, kernel_size, kernel_size)

		# print layer_shape
		layer_shape_dict[layer_name] = layer_shape

	np.save(layer_shapes_file, layer_shape_dict)
	print ("Layer shapes written to file :", layer_shapes_file)

	return layer_shapes_file


layer_names_all_file = get_layer_names_all(model)
layer_names_file = get_layer_names(model)
layer_shapes_file = get_layer_shapes(model, layer_names_file)


def create_weights_dict(model):

	WEIGHTS = np.load(layer_names_all_file).item()
	conv_shape = np.load(layer_shapes_file).item()

	weights_path = "./weights"
	fileNames = filter(lambda f: not f.startswith('.'), os.listdir(weights_path))

	paths = {}
	weights_dict = {}

	for filename in fileNames:
		paths[filename.replace('.csv','')]= weights_path + "/" + filename

	for name in WEIGHTS:

		if 'conv' in name:

			print colored('conv :' + name, "cyan")
			conv_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
			conv_w = np.reshape(conv_w, conv_shape[name])
			conv_w = np.transpose(conv_w, (2, 3, 1, 0))
			print colored(conv_shape[name], "cyan")
			conv_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
			weights_dict[name] = [conv_w, conv_b]

		elif 'bn' in name:

			print colored('bn :' + name, "yellow")
			bn_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
			bn_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
			bn_m = genfromtxt(paths[name + '_m'], delimiter=',', dtype=None)
			bn_v = genfromtxt(paths[name + '_v'], delimiter=',', dtype=None)
			weights_dict[name] = [bn_w, bn_b, bn_m, bn_v]


		elif 'dense' in name:

			print colored('dense :' + name, "red")
			dense_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
			dense_w = np.reshape(128, 736)
			dense_w = np.transpose(dense_w, (1, 0))
			print colored(conv_shape[name], "red")
			dense_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
			weights_dict[name] = [dense_w, dense_b]


	return weights_dict


weight_dict = create_weights_dict(model)

np.save("facenet_weight_ptm.npy", weight_dict)

def save_weights_ptm(model, weight_dict, weights_file= "facenet_ptm.h5"):
	# K.set_image_data_format('channels_first')
	# K.set_image_dim_ordering('tf')
	for layer in model.layers:
	# print layer.name
		# print weight_dict[layer]
		keys = [key for key in weight_dict]
		if layer.name in keys:

			print layer.name
			
			model.get_layer(layer.name).set_weights(weight_dict[layer.name])
			# print model.get_layer(layer.name).input_shape
			# print model.get_layer(layer.name).output_shape
			# print colored(np.array(weight_dict[layer.name]).shape, "red")
			# weights = layer.get_weights()
			# print colored(weights[0].shape, "green")
			# print colored(np.array(weight_dict[layer.name][0]).shape, "red")
			# layer.set_weights(weight_dict[layer.name])

	model.save_weights(weights_file)

save_weights_ptm(model, weight_dict)


def load_dataset():

	# (u'list_classes', <HDF5 dataset "list_classes": shape (2,), type "<i8">)
	# (u'train_set_x', <HDF5 dataset "train_set_x": shape (600, 64, 64, 3), type "|u1">)
	# (u'train_set_y', <HDF5 dataset "train_set_y": shape (600,), type "<i8">)


	train_dataset = h5py.File('dataset/train_happy.h5', "r")
	train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # shape (600,64,64,3)
	train_set_y_orig = np.array(train_dataset["train_set_y"][:])

	test_dataset = h5py.File('dataset/test_happy.h5', "r")
	test_set_x_orig = np.array(test_dataset["test_set_x"][:])
	test_set_y_orig = np.array(test_dataset["test_set_y"][:])

	classes = np.array(test_dataset["list_classes"][:])


	train_set_y_orig = train_set_y_orig.reshape(1,train_set_y_orig.shape[0]) # add the batch
	test_set_y_orig = test_set_y_orig.reshape(1, test_set_y_orig.shape[0]) # add the batch size 

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


train_x, train_y, test_x, test_y = load_dataset()


# this function defines the triplet loss as defined in the facenet paper

def triplet_loss(y_true, y_pred, alpha = 0.2):

  anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

  # computing the distance between anchor and positive 
  pos_distance = tf.square(anchor-positive)
  # computer the distance between anchor and negative 
  neg_distance = tf.square(anchor-negative)

  # subtract two distances and add alpha 
  basic_loss = tf.reduce_sum(pos_distance - neg_distance) + alpha

  # take the maximum of basic loss and 0.0 and sum over training examples   
  # loss = tf.reduce_sum(tf.maximum(basic_loss, 0.))

  loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

  return loss 


# testing triplelet loss 

with tf.Session() as test:
  tf.set_random_seed(1)

  y_true = (None, None, None)
  y_pred = (tf.random_normal([3,128], mean = 6, stddev = 0.1, seed = 1), 
            tf.random_normal([3,128], mean = 1, stddev = 1, seed = 1), 
            tf.random_normal([3,128], mean = 3, stddev = 4, seed = 1)) 
  
  loss = triplet_loss(y_true, y_pred)

  print ("loss : " + str(loss.eval()))



# load_weight

model.load_weights("./facenet_ptm.h5", by_name = True)
print colored("pre-traind weights loaded", "green")
model.save("facenet_model_weight_defn.h5")
print colored("face recognition model and definition saved ")

from keras.models import load_model

print ("loading pre-trained face recognition model")
model = load_model("facenet_model_weight_defn.h5")

model.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])


import cv2
def img_to_encoding(img_path, model):

	img = cv2.imread(img_path, 1) # BGR
	img = img[:,:,::-1] # BGR->RGB 
	img = np.around(np.transpose(img, (2,0,1)) / 255.0, decimals=12) # hxwxch -> chxhxw -> normalization 
	x_train = np.array([img])

	embedding = model.predict_on_batch(x_train)

	return embedding 


database = {}
database["danielle"] = img_to_encoding("./images/danielle.png", model)
database["younes"] = img_to_encoding("./images/younes.jpg", model)
database["tian"] = img_to_encoding("./images/tian.jpg", model)
database["andrew"] = img_to_encoding("./images/andrew.jpg", model)
database["kian"] = img_to_encoding("./images/kian.jpg", model)
database["dan"] = img_to_encoding("./images/dan.jpg", model)
database["sebastiano"] = img_to_encoding("./images/sebastiano.jpg", model)
database["bertrand"] = img_to_encoding("./images/bertrand.jpg", model)
database["kevin"] = img_to_encoding("./images/kevin.jpg", model)
database["felix"] = img_to_encoding("./images/felix.jpg", model)
database["benoit"] = img_to_encoding("./images/benoit.jpg", model)
database["arnaud"] = img_to_encoding("./images/arnaud.jpg", model)


np.set_printoptions(precision=2) 

def confusion_matrix_compute(database):

	confusion_matrix = np.zeros((len(database), len(database)))
	col = 0
	row = 0 
	keys = [key for key in database]
	for key in keys:
		col =0 
		for key1 in keys:
			conf_value = np.linalg.norm(database[key] - database[key1])  


			confusion_matrix[row][col] = conf_value
			col +=1 
		row +=1 
	return confusion_matrix

confusion_matrix = confusion_matrix_compute(database)

def verify(img_path, identity, database, model):

	embedding = img_to_encoding(img_path, model)

	dist = np.linalg.norm(embedding - database[identity])

	if dist < 0.7:
		print("It's " + str(identity) + ", welcome home")

	else:
		print("Identity verification failed, not  " + str(identity) + ", gate closed")
	
	return dist


dist = verify("./images/arnaud.jpg", "arnaud", database, model)

print dist


def distance_calc(img1, img2, model):

  embedding_img1 = img_to_encoding(img1, model)
  embedding_img2 = img_to_encoding(img2, model)
  distance = np.linalg.norm(embedding_img1 - embedding_img2)

  return distance 

def identify_person(img, database, model):

  embedding = embedding = img_to_encoding(img, model)

  min_dist = 2.0 
  min_key = None
  for key in database:
    dist = np.linalg.norm(embedding - database[key])

    if dist < min_dist:
      min_dist = dist 
      min_key = key


  if(min_dist > 0.7 or min_key is None ):
    print(colored("Not matching with any one in this database " + ":" + str(min_dist) , "red"))
  else:
    print(colored("Match found with : " + str(key) + ":" + str(min_dist)), "green")


identify_person("images/camera_0.jpg", database, model)




