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
beam_size=4
min_dec_steps=35
vocab_size=2500

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = True
is_lsa = True
is_esa = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 2000
early_stopping = 0

use_gpu=True

lr_coverage=0.15
use_lstm=True

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

# parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
# parser.add_argument('--seed', type=int, default=1,help='random seed')
# parser.add_argument('--batch_size', type=int, default=8,help='batchsize')
# parser.add_argument('--data_parallel', action='store_true', default=False, help='flag to use nn.DataParallel')
# parser.add_argument('--lr', type=float, default=0.05,help='learning rate')
# parser.add_argument('--pytorch_version', type=str, default='nightly',help='location of the data corpus')
# parser.add_argument('--data_file', type=str, default='data/yelp-2013/yelp-2013-all.pkl',help='location of the data corpus')
# parser.add_argument('--save_path', type=str, default='./saved_models/debug',help='location of the best model and generated files to save')
# parser.add_argument('--word_emsize', type=int, default=300,help='size of word embeddings')

# parser.add_argument('--dim_str', type=int, default=50,help='size of word embeddings')
# parser.add_argument('--dim_sem', type=int, default=50,help='size of word embeddings')
# parser.add_argument('--dim_output', type=int, default=5,help='size of word embeddings')
# parser.add_argument('--n_embed', type=int, default=49030,help='size of word embeddings')
# parser.add_argument('--d_embed', type=int, default=200,help='size of word embeddings')

# parser.add_argument('--nlayers', type=int, default=1,help='number of layers')
# parser.add_argument('--dropout', type=float, default=0.2,help='dropout applied to layers (0 = no dropout)')
# parser.add_argument('--clip', type=float, default=5,help='gradient clip')
# parser.add_argument('--log_period', type=float, default=100,help='log interval')
# parser.add_argument('--epochs', type=int, default=50,help='epochs')
# parser.add_argument('--cnn', action='store_true', default=False, help='flag to use cnn encoder')