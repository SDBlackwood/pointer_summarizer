"""

CODE IS AMMENDED FROM: 
https://bitbucket.org/tanya14109/cqasumm/src/master/

"""
import tensorflow as tf
from tensorflow.core.example import example_pb2
import struct
import glob
import os

data_dir = "/home/postscript/Workspace/DS/Workspace/Project/Code/stacksumm/data/processed/stackexchange"
finished_files_dir = "/home/postscript/Workspace/DS/Workspace/Project/Code/stacksumm/data/processed/stackexchange/finished_files"
chunks_dir = "/home/postscript/Workspace/DS/Workspace/Project/Code/stacksumm/data/processed/stackexchange/finished_files/chunked"

CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

from document import CQA_DOCUMENT

def buildPointerGeneratorDataset(samples):
	ctr=0
	samples_len = len(samples)

	train_split = 0.7
	test_split = 0.2
	eval_split = 0.1

	train_num =  int( samples_len * train_split ) -1
	test_num =  int( samples_len * test_split ) -1
	eval_num =  int( samples_len * eval_split ) -1

	out_file_train = finished_files_dir + "/train.bin"
	out_file_test = finished_files_dir + "/test.bin"
	out_file_val = finished_files_dir + "/val.bin"

	write(out_file_train, samples[0:train_num])
	write(out_file_test, samples[train_num:test_num+train_num])
	write(out_file_val, samples[test_num+train_num:test_num+train_num+eval_num])

def write(out_file, samples):
	with open(out_file, 'wb') as writer:
		for cqa in samples:	
			docs = cqa.setAllAnswersConcat();
			bestAnswer = cqa.getReference();

			# Write to tf.Example
			tf_example = example_pb2.Example()
			tf_example.features.feature['article'].bytes_list.value.extend([docs.encode()])
			tf_example.features.feature['abstract'].bytes_list.value.extend([bestAnswer.encode()])
			tf_example_str = tf_example.SerializeToString()
			str_len = len(tf_example_str)
			writer.write(struct.pack('q', str_len))
			writer.write(struct.pack('%ds' % str_len, tf_example_str))