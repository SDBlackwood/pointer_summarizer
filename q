[1mdiff --git a/data_util/config.py b/data_util/config.py[m
[1mindex 36f5581..a6abc02 100644[m
[1m--- a/data_util/config.py[m
[1m+++ b/data_util/config.py[m
[36m@@ -36,7 +36,7 @@[m [mis_esa = True[m
 cov_loss_wt = 1.0[m
 [m
 eps = 1e-12[m
[31m-max_iterations = 20000[m
[32m+[m[32mmax_iterations = 10000[m
 early_stopping = 0[m
 [m
 use_gpu=True[m
[36m@@ -50,7 +50,7 @@[m [mpy_version="1.9"[m
 data_file='data/yelp-2013/yelp-2013-all.pkl'[m
 reload_path='./saved_models/best_model.pth'[m
 dim_str=50 # size of word embeddings[m
[31m-dim_sem=50 #size of word embeddings[m
[32m+[m[32mdim_sem=30 #size of word embeddings[m
 nlayers=1 #number of layers')[m
 dropout=0.2 # dropout applied to layers (0 = no dropout)[m
 clip=5 # gradient clip[m
[1mdiff --git a/training_ptr_gen/model.py b/training_ptr_gen/model.py[m
[1mindex e513933..b54e27d 100644[m
[1m--- a/training_ptr_gen/model.py[m
[1m+++ b/training_ptr_gen/model.py[m
[36m@@ -155,7 +155,7 @@[m [mclass Attention(nn.Module):[m
         self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)[m
         # Structural Attenion[m
         if config.is_lsa:[m
[31m-            self.W_r = nn.Linear(attention_dimension + 256, config.hidden_dim * 2, bias=False)[m
[32m+[m[32m            self.W_r = nn.Linear(attention_dimension + config.hidden_dim, config.hidden_dim * 2, bias=False)[m
 [m
     def forward([m
         self,[m
[1mdiff --git a/training_ptr_gen/structured_attention.py b/training_ptr_gen/structured_attention.py[m
[1mindex 3ef309a..d288c5f 100644[m
[1m--- a/training_ptr_gen/structured_attention.py[m
[1m+++ b/training_ptr_gen/structured_attention.py[m
[36m@@ -7,7 +7,6 @@[m [mExample[m
 self.sentence_structure_att = StructuredAttention(device, self.sem_dim_size, self.sent_hidden_size, bidirectional, py_version)[m
 self.document_structure_att = StructuredAttention(device, self.sem_dim_size, self.doc_hidden_size, bidirectional, py_version)[m
 [m
[31m-[m
 Returns:[m
     [type]: [description][m
 """[m
