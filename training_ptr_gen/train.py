from __future__ import unicode_literals, print_function, division

import sys
sys.path.append("/home/postscript/Workspace/DS/Workspace/Project/Code/stacksumm2/training_ptr_gen")

import os
import time
import argparse

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adagrad

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch

# Structured Attentions
from structured_attention import StructuredAttention

use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        

        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.create_file_writer(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + list(self.model.reduce_state.parameters())

        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):

        self.optimizer.zero_grad()
        loss = self.model(batch)
        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()
        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        while iter < n_iters:
    
            batch = self.batcher.next_batch()
            # Batches are (8,400) - 8 Rows of 400
            loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

                
            if iter % int(config.log_period) == 0:
                elapsed = time.time() - start
                estimated_time = ((config.max_iterations / iter) * elapsed) /60 / 60
                text = 'steps %d, seconds for %d batch: %.2f, loss: %f.  Estimated training time: %d hrs' % (iter, iter, time.time() - start, loss, estimated_time)  
                print(text)
                self.summary_writer.flush()
    
            if iter % 5000 == 0:
                self.save_model(running_avg_loss, iter)

        self.save_model(running_avg_loss, iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    parser.add_argument("-lsa",
                        dest="is_lsa", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    parser.add_argument("-esa",
                        dest="is_esa", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()
    
    train_processor = Train()
    train_processor.trainIters(config.max_iterations, args.model_file_path)
