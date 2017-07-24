from __future__ import print_function

import sys
import os
import time
import string
import random
import collections
import cPickle as pickle
import gzip
import ast
import argparse

import numpy as np
import theano
import theano.tensor as T
import lasagne

from utils import *
from model import *

#from StyleTransfer.evaluate import 
from StyleTransfer.utils import *

def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--inputloc', type=str, default='gogh-fig-pics/test_input/',
						help="the path to the image, or to a folder containing the images, to be stylized")
	parser.add_argument('-o', '--outputloc', type=str, default='gogh-fig-pics/test_output/',
						help="the location where the stylized image/images is/are to be stored")
	parser.add_argument('-m', '--modelloc', type=str, default='data/model/trained_cin_se-4_i2c_c33_ea_misc32/e2.npz',
						help="the location of the trained model file")
	parser.add_argument("-n", "--numstyles", type=int, default=32,
						help="the number of styles in the trained model file, i.e., the number of style images used to train the loaded model")
	parser.add_argument("-b", "--batchsize", type=int, default=4,
						help="the batchsize to be used during stylization")
	parser.add_argument("-d", "--dim", type=str, default=None,
						help="the (rough) output dimension; None (default) leaves the images the same size as the input; providing a tuple makes the pastiche roughly this size; (975, 1300) seems to work well")
	parser.add_argument('--output_basename', type=str, default='transformed',
                        help='output base name')
	parser.add_argument("--border", type=int, default=32,
			help="border size to add to the image before transformation (default 32)")
	parser.add_argument('--resize', type=int, default=None,
                        help='resize the image to this size before transformation while keeping the aspect ratio'
                             '(default no resize)')  
	parser.add_argument("--styles", type=str, default=None,
                        help="styles location for descriptive output naming (default name for each style is its label id in range {0, ..., NUM_STYLES-1} )")
	
	parser.add_argument('-c1', '--channels1', type=int, default=64,
                                                help="the suffix to be added to the folders used to store debug images and trained model params")
        parser.add_argument('-c2', '--channels2', type=int, default=128,
                                                help="the suffix to be added to the folders used to store debug images and trained model params")
	parser.add_argument('-g', '--gray', action='store_true',
                                                help="force output to grayscal")
	args = parser.parse_args()

	if args.dim:
		args.dim = ast.literal_eval(args.dim)

	return args

def _get_style_names(styles_dir, num_styles):
	if styles_dir is not None:
		return  [style.split('.')[0] for style in os.listdir(styles_dir)]
	return list(map(str, range(num_styles)))	

def stylize(args):
	INPUT_LOCATION = args.inputloc
	OUTPUT_LOCATION = args.outputloc
	MODEL_FILE = args.modelloc
	NUM_STYLES = args.numstyles
	MAX_BATCH_SIZE = args.batchsize
	DIMENSION = args.dim
	
	style_names = _get_style_names(args.styles, NUM_STYLES)

	#_arg_parser('--images_path=' + INPUT_LOCATION + \
	#	    ' --out_dir=' + OUTPUT_LOCATION + \
	#	    ' --resize=512' )
	
	image_var = T.tensor4('inputs')
	chosen_style_var = T.ivector('chosen_style')
	
	
	print('Loading Images...')
	#images = get_images(INPUT_LOCATION, dim=DIMENSION, center=False, correct_vertical=True)
	
	images_paths = image_list(INPUT_LOCATION)
	images_names = [os.path.basename(image_path) for image_path in images_paths]
	images = pre_process(images_paths, args.border, args.resize)
	images = [image.transpose(0, 3, 1, 2)/255. for image in images]
	
	print('Loading Networks...')
	net = Network(image_var, NUM_STYLES, args.channels1, args.channels2, shape=(None,3,None,None), gray=args.gray)
	load_params(net.network['transform_net'], MODEL_FILE)

	print('Compiling Functions...')
	# initialize transformer network function
	transform_pastiche_out = lasagne.layers.get_output(net.network['transform_net'], style=chosen_style_var)
	pastiche_transform_fn = theano.function([image_var, chosen_style_var], transform_pastiche_out)
	print('Transforming images...')

	# TODO: There might be a more efficient way to do this if there's only one style image,
	# but it would require all the content images to be of the same size to get batched together
	transformed_images = []
	#for num, image in enumerate(images):
	for image, image_name in zip(images, images_names):
		start_time = time.time()
		transformed_images = []
		image = image.astype('float32')
		#print(image.shape)
		#image = np.expand_dims(image, axis=0)
		image=np.tile(image, (MAX_BATCH_SIZE,1,1,1))
		for i in range(NUM_STYLES//MAX_BATCH_SIZE):
			out_ims = list(pastiche_transform_fn(image, range(MAX_BATCH_SIZE*i, MAX_BATCH_SIZE*(i+1))))
			#print(out_ims[0].shape)
			out_ims = [post_process(transformed.transpose(1, 2, 0), args.border) for transformed in out_ims]
			#print(np.min(out_ims[0]), np.max(out_ims[1]))
			#out_ims = [post_process(transformed.transpose(1, 2, 0), args.border) for transformed in out_ims]
			transformed_images += list(zip(out_ims, style_names[MAX_BATCH_SIZE*i:MAX_BATCH_SIZE*(i+1)], [image_name]*len(out_ims))) 
		if MAX_BATCH_SIZE*(i+1) < NUM_STYLES:
			labels = list(range(MAX_BATCH_SIZE*(i+1), NUM_STYLES))
			relevant_length = len(labels)
			labels = labels + [0]*(MAX_BATCH_SIZE-relevant_length)
			out_ims = list(pastiche_transform_fn(image, labels))
                        out_ims = [post_process(transformed.transpose(1, 2, 0), args.border) for transformed in out_ims[:relevant_length]]
                        transformed_images += list(zip(out_ims, style_names[MAX_BATCH_SIZE*(i+1):NUM_STYLES], [image_name]*relevant_length))	
		
		print("  Done with image {}. Time: {:.3f}s".format(image_name, time.time() - start_time))
		#save_ims(OUTPUT_LOCATION, out_ims, 'im' + str(num) + '_debug2')
		#print('  Saved.')
		save_images(transformed_images, OUTPUT_LOCATION, args.output_basename)
	print('Done.')


if __name__ == '__main__':
	args = parse_args()
	stylize(args)
