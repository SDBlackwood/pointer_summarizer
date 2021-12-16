from __future__ import unicode_literals, print_function, division
import transformer_encoder
from structured_attention import StructuredAttention
from BiLSTM import BiLSTMEncoder
from numpy import random
from data_util import config
from data_util.utils import calc_mem, format_mem

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys
sys.path.append(".")

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag,
                                 config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)


def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)


def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

        self.drop = nn.Dropout(0.3)

        # (512, 512)
        self.W_h = nn.Linear(config.hidden_dim * 2,config.hidden_dim * 2, bias=False)

        self.sem_dim_size = 2*config.dim_sem
        self.token_hidden_size = 2*config.hidden_dim
        self.anwer_hidden_size = 2*config.hidden_dim

        """
        TODO - Decide whether we have 2 encoders or whether the answer
        are max pooled
        """
        self.token_level_encoder = BiLSTMEncoder(
            torch.device("cuda" if use_cuda else "cpu"), 
            self.token_hidden_size, 
            config.emb_dim, 
            1, 
            dropout=0.3,
            bidirectional=True
        )
        self.answer_level_encoder = BiLSTMEncoder(
            torch.device("cuda" if use_cuda else "cpu"), 
            self.anwer_hidden_size, 
            self.token_hidden_size, 
            1, 
            dropout=0.3,
            bidirectional=True
        )

        # Adding Structural Attention
        self.structure_attention = StructuredAttention(
            torch.device("cuda" if use_cuda else "cpu"),
            self.sem_dim_size,
            config.hidden_dim * 2,
            config.bidirectional,
            config.py_version
        )

    # seq_lens should be in descending order
    def forward(self, input, seq_lens, enc_padding_mask):

        embedded = self.embedding(input)

        # Sentence Level BiLSTM
        encoder_outputs, hidden = self.token_level_encoder.forward_packed(embedded,seq_lens)

        # B * t_k x 2*hidden_dim
        encoder_feature = encoder_outputs.view(-1, 2*config.hidden_dim)
        encoder_feature = self.W_h(encoder_feature)

        # Structural Encoder Attenion Network - returns structured infused `ri` for `i` timestep
        # We need the Wr.ri here W
        ri, a_jk = self.structure_attention(encoder_outputs)

        return encoder_outputs, encoder_feature, hidden, ri, a_jk


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden  # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        # h, c dim = 1 x b x hidden_dim
        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))

def calc_mem(layer):
    size_bytes = ((linear.in_features * linear.out_features) * (4 / (1024^3)))*2
    if size_bytes == 0:
           return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    print("%s %s" % (s, size_name[i]))
    return "%s %s" % (s, size_name[i])

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.W_s = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)
        # Structural Attenion
        self.W_r = nn.Linear(config.dim_sem * 2, config.hidden_dim * 2, bias=False)

    def forward(
        self,
        s_t_hat,
        encoder_outputs,
        encoder_feature,
        enc_padding_mask,
        coverage,
        ri
    ):
        # 8, 400, 512
        b, t_k, n = list(encoder_outputs.size())

        # s_t_hat [8,512], self.W_s (512,512) -> dec_fea [8,512]
        dec_fea = self.W_s(s_t_hat)
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # [8,400,512]
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # [3200,512]

        att_features = encoder_feature + dec_fea_expanded  # [3200,512] + [3200,512]

        # Coverage
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        # Calculate Attention
        e = torch.tanh(att_features)  # [3200,512]

        scores = self.v(e)  # [3200,1]
        scores = scores.view(-1, t_k)  # [8,400]
        attn_dist_ = torch.softmax(scores, dim=1)*enc_padding_mask  # [8,400]

        normalization_factor = attn_dist_.sum(1, keepdim=True)  # [8,1]
        attn_dist = attn_dist_ / normalization_factor  # [ 8,400]
        attn_dist = attn_dist.unsqueeze(1)  # #[8,1,400]

        # Context Vector
        # [8,1,400] * [8,400,512] = [8,1,400]
        c_t = torch.bmm(attn_dist, encoder_outputs)
        c_t = c_t.view(-1, config.hidden_dim * 2)  # [8,512]

        attn_dist = attn_dist.view(-1, t_k)  # [8,400]

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        # Structural attention
        # ri [8,400,50] Wr [50,512]
        Wrri = self.W_r(ri)  # [8,400,512]
        Wrri = Wrri.contiguous().view(-1, n)  # [3200,512]

        Wsst = self.W_s(s_t_hat)  # [8,400,512]
        Wsst = Wsst.unsqueeze(1)
        Wsst = Wsst.expand(b, t_k, n).contiguous()  # [3200,50]
        Wsst = Wsst.view(-1, n)  # [3200,50]

        att_struct_features = Wrri + Wsst

        # Wr(ri)            [8, 400, 512]
        # self.W_s(s_t_hat) [8, 512]
        e_stuct_t_i = self.v(
            torch.tanh(att_struct_features)
        )
        e_stuct_t_i = e_stuct_t_i.view(-1, t_k)

        a_t_struct = torch.softmax(e_stuct_t_i, dim=1)*enc_padding_mask
        a_t_struct = a_t_struct.unsqueeze(1)
        # Calculate the structural context vector
        c_t_stuct = torch.bmm(a_t_struct, encoder_outputs)
        c_t_stuct = c_t_stuct.view(-1, config.hidden_dim * 2)

        del e_stuct_t_i, att_struct_features, Wsst, Wrri, normalization_factor

        return c_t, attn_dist, coverage, c_t_stuct, a_t_struct


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(
            config.emb_dim,
            config.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        # p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)

    """
    y_t_1 - Inital Decoder Index Vector
    c_t_1 - Inital Context Vector
    s_t_1 - Inital Decoder State
    """
    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step, ri):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim), c_decoder.view(-1, config.hidden_dim)), 1)

            # `c_t` - context vector
            c_t, _, coverage_next, c_struct_t, a_struct_t = self.attention_network(
                s_t_hat,
                encoder_outputs,
                encoder_feature,
                enc_padding_mask,
                coverage
            )
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))

        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim

        # Intertoken Attention
        c_t, attn_dist, coverage_next, c_struct_t, a_struct_t = self.attention_network(
            s_t_hat,
            encoder_outputs,
            encoder_feature,
            enc_padding_mask,
            coverage,
            ri
        )

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            # B x (2*2*hidden_dim + emb_dim)
            p_gen_input = torch.cat((c_struct_t, s_t_hat, x), 1)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        # B x hidden_dim * 3
        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_struct_t), 1)
        output = self.out1(output)  # B x hidden_dim
        output = self.out2(output)  # B x vocab_size

        vocab_dist = torch.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist
        
        del vocab_dist, vocab_dist_, attn_dist_, output, s_t_hat, h_decoder, c_decoder, lstm_out

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


class Model(nn.Module):
    def __init__(self, model_file_path=None, is_eval=False):
        super(Model, self).__init__()

        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()

        # shared the embedding between encoder and decoder
        #decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            state = torch.load(
                model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(
                state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])
