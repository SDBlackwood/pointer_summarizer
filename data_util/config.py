import os

# Load the .env file
from dotenv import load_dotenv
load_dotenv()

root_dir = os.path.expanduser("~")

#train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
train_data_path = os.path.join(root_dir, os.getenv('train_data_path'))
eval_data_path = os.path.join(root_dir, os.getenv('eval_data_path'))
decode_data_path = os.path.join(root_dir, os.getenv('decode_data_path'))
vocab_path = os.path.join(root_dir, os.getenv('vocab_path'))
log_root = os.path.join(root_dir, os.getenv('log_root'))

# Hyperparameters
hidden_dim = 128
emb_dim= 64
batch_size= 8
max_enc_steps=100
max_dec_steps=50
beam_size=8
min_dec_steps=35
vocab_size=25000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
is_lsa = False
is_esa = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 2000
early_stopping = 0

use_gpu=True

lr_coverage=0.15

## Structured Config
seed=1
py_version="1.9"
data_file='data/yelp-2013/yelp-2013-all.pkl'
reload_path='./saved_models/best_model.pth'
dim_str=50 # size of word embeddings
dim_sem=50 #size of word embeddings
nlayers=1 #number of layers')
dropout=0.2 # dropout applied to layers (0 = no dropout)
clip=5 # gradient clip
log_period=10 # log interval
bidirectional=True