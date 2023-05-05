import math
import torch
import torch.nn as nn
from functools import reduce
from utils import blosum_matrix
import torch.nn.functional as F
from torch.distributions import Normal

def add(x, y):
    return x+y

class OnehotEmbedding(nn.Module):
    def __init__(self, word_number):
        super().__init__()
        self.word_number = word_number
    
    def one_hot_encoding(self, labels):
        one_hot = torch.zeros(labels.size(0), labels.size(1), self.word_number).cuda()
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        return one_hot
        
    def forward(self, x):
        embeddings = F.one_hot(x, self.word_number).float()[:,:,1:]
        # print(embeddings.shape, embeddings[0, 0].shape, embeddings[0, -1].shape, embeddings[0, -1])
        return embeddings

class Attention(nn.Module):
    def __init__(self, hidden, alignment_network='dot'):
        super().__init__()
        self.style = alignment_network.lower()
        if self.style == 'general':
            self.transform = nn.Linear(hidden, hidden)
        elif self.style == 'bilinear':
            self.weight = nn.Parameter(torch.Tensor(hidden, hidden))
            self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        
    def forward(self, query, key):
        if self.style == 'dot':
            return torch.bmm(query, 
                             key.transpose(1, 2))
        elif self.style == 'general':
            return torch.bmm(query, 
                             self.transform(key).transpose(1, 2))
        elif self.style == 'bilinear':
            # return self.transform(query, key).squeeze(-1)
            return torch.bmm(query.matmul(self.weight), 
                             key.transpose(1, 2))
        elif self.style == 'decomposable':
            return torch.bmm(self.transform(query),
                             self.transform(key).transpose(1, 2))

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        channel = 8
        self.input_size = 20
        self.max_seq_length = 512
        total_layers = int(math.log2(self.max_seq_length))
        self.convs = nn.Sequential(
            nn.Conv1d(1, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
        )
        for i in range(total_layers - 1):
            self.convs.add_module('conv{}'.format(i), nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False))
            self.convs.add_module('pool{}'.format(i), nn.AvgPool1d(2))
        
        self.flat_size = self.input_size * channel
        self.linear1 = nn.Linear(self.flat_size, self.flat_size)
        self.linear2 = nn.Linear(self.flat_size, self.flat_size)
        
        self.radius = nn.Parameter(torch.Tensor([1e-2]), requires_grad=True)

        self.alpha = nn.Parameter(torch.Tensor([1e-2]), requires_grad=True)
        self.a     = nn.Parameter(torch.Tensor([1e-2]), requires_grad=True)
        self.b     = nn.Parameter(torch.Tensor([1e-2]), requires_grad=True)

        self.power = nn.Parameter(torch.Tensor([2.]), requires_grad=True)

        self.std = nn.Parameter(torch.Tensor([1e-2]), requires_grad=False)
        self.lpm = nn.Parameter(torch.tensor(blosum_matrix, dtype=torch.float32), requires_grad=True)
        self.pm = torch.tensor(blosum_matrix, dtype=torch.float32).clamp(min = 0)
  
    def column(self, x):
        index = x.nonzero()
        new_x = torch.zeros_like(x).cuda()
    
        for i in range(20):
            sub_index  = torch.nonzero(index[:, 1] == i).squeeze()
            tem_value  = torch.index_select(index, 0, sub_index)
            tem_index  = torch.split(tem_value, split_size_or_sections = 1, dim=1)
            tem_index1 = [tem_index[0].squeeze(), tem_index[1].squeeze(), tem_index[2].squeeze()]
            for j in range(1, 19 - i):
                my_index = [tem_index1[0], (tem_index1[1] + j).clamp(max = 19, min = 0), tem_index1[2]]
                new_x = new_x.index_put(my_index, self.lpm[i][i + j].clamp(max = 1, min = 0.001) * self.pm[i][i + j])
            for j in range(1, i + 1):
                my_index = [tem_index1[0], (tem_index1[1] - j).clamp(max = 19, min = 0), tem_index1[2]]
                new_x = new_x.index_put(my_index, self.lpm[i - j][i].clamp(max = 1, min = 0.001) * self.pm[i - j][i])
        
        return new_x

    def row(self, x):
        index = x.nonzero()

        new_x = torch.zeros_like(x).cuda()
        index = torch.split(index, split_size_or_sections = 1, dim=1)
        new_index = [index[0].squeeze(), index[1].squeeze(), index[2].squeeze()]
        u_index = [new_index[0], new_index[1], (new_index[2] + 1).clamp(max = self.max_seq_length - 1, min = 0)]
        d_index = [new_index[0], new_index[1], (new_index[2] - 1).clamp(max = self.max_seq_length - 1, min = 0)]

        gauss      = Normal(torch.tensor([1.0]).cuda(), self.std)
        base_value = gauss.log_prob(torch.Tensor([1.0]).cuda())
        base_value = base_value.exp()
        value      = gauss.log_prob(torch.Tensor([1.0 + 1]).cuda()).exp() / base_value
        new_x = new_x.index_put(u_index, value)
        new_x = new_x.index_put(d_index, value)
        
        u_index = [new_index[0], new_index[1], (new_index[2] + 2).clamp(max = self.max_seq_length - 1, min = 0)]
        d_index = [new_index[0], new_index[1], (new_index[2] - 2).clamp(max = self.max_seq_length - 1, min = 0)]
        value   = gauss.log_prob(torch.Tensor([1.0 + 2]).cuda()).exp() / base_value
        new_x = new_x.index_put(u_index, value)
        new_x = new_x.index_put(d_index, value)

        u_index = [new_index[0], new_index[1], (new_index[2] + 3).clamp(max = self.max_seq_length - 1, min = 0)]
        d_index = [new_index[0], new_index[1], (new_index[2] - 3).clamp(max = self.max_seq_length - 1, min = 0)]
        value   = gauss.log_prob(torch.Tensor([1.0 + 3]).cuda()).exp() / base_value
        new_x = new_x.index_put(u_index, value)
        new_x = new_x.index_put(d_index, value).cuda()
        
        return new_x
  
    def forward(self, x, masks):
        seq_num = len(x)
        x = x.permute(0, 2, 1)

        column_x = self.column(x)
        row_x    = self.row(x)
        x = x + column_x
        x = x + row_x
        
        x = x.contiguous().view(-1, 1, self.max_seq_length)
        x = self.convs(x).squeeze(-1)
        x = x.view(seq_num, self.input_size, -1)
        # x = self.convs(x)
        # x = x.view(seq_num, self.flat_size)
        
        return x
        
  
class RNN(nn.Module):
    def __init__(self, input_size=-1, hidden_size=-1, num_layers=1, output_size=None, rnn_type='lstm', bidirectional=True, dropout=0.0):
        super(RNN, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if bidirectional:
            hidden_size //= 2
            
        self.output_size = output_size if output_size is not None else hidden_size

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError("Invalid rnn_type. Choose either 'lstm' or 'gru'.")

    def forward(self, x, masks):
        """
        Args:
            x: (batch_size, seq_len, input_size)
            masks: (batch_size, seq_len)
        Returns:
            output: (batch_size, seq_len, output_size)
            hidden: (batch_size, num_layers * num_directions, hidden_size)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, masks.sum(1).int().cpu(), batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(x)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=seq_len)
        return output
        
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, alignment_network='dot'):
        super().__init__()
        self.alignment_network = Attention(hidden_size, alignment_network)
        
    def forward(self, inputs, masks):
        alignment_scores = self.alignment_network(inputs, inputs)
        alignment_scores.masked_fill_(~masks.unsqueeze(1), float('-inf'))
        alignment_scores = F.softmax(alignment_scores, dim=-1)
        SelfAttentionOutput = torch.bmm(alignment_scores, inputs)
        return SelfAttentionOutput
    
class PairAttention(nn.Module):
    def __init__(self, hidden_size, alignment_network='dot'):
        super().__init__()
        self.alignment_network = Attention(hidden_size, alignment_network)
        
    def forward(self, input_with_meta,
                      context_with_meta,
                      raw_input_with_meta,
                      raw_context_with_meta,
                      input_masks,
                      context_masks):
        alignment_scores = self.alignment_network(raw_input_with_meta, raw_context_with_meta)
        alignment_scores.masked_fill_(~context_masks.unsqueeze(1), float('-inf'))
        normalized_scores = F.softmax(alignment_scores, dim=-1)
        values_aligned = torch.bmm(normalized_scores, context_with_meta)
        
        return values_aligned
        
class GlobalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.alignment_network = nn.Linear(hidden_size, 1)
        
    def forward(self, inputs, masks=None):
        alignment_scores = self.alignment_network(inputs)
        if masks is not None:
            alignment_scores = alignment_scores.squeeze(-1).masked_fill_(~masks, float('-inf'))
        else:
            alignment_scores = alignment_scores.squeeze(-1)
        normalized_scores = F.softmax(alignment_scores, dim=1)
        output = torch.bmm(normalized_scores.unsqueeze(1), inputs).squeeze(1)
        
        return output
    
class GatingMechanism(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        stdv = (stdv1 + stdv2) / 2.
        self.weight1.data.uniform_(-stdv1, stdv1)
        self.weight2.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, inputs, hiddens):
        G = torch.sigmoid(inputs.matmul(self.weight1) + hiddens.matmul(self.weight2) + self.bias)
        output = G * inputs + (1 - G) * hiddens
        return output
    
class Merge(nn.Module):
    _style_map = {
        'sum': lambda *args: reduce(add, args),
        'concat': lambda *args: torch.cat(args, args[0].dim() - 1),
        'diff': lambda x, y: x - y,
        'abs-diff': lambda x, y: torch.abs(x - y),
        'concat-diff': lambda x, y: torch.cat((x, y, x - y), x.dim() - 1),
        'concat-abs-diff': lambda x, y: torch.cat((x, y, torch.abs(x - y)), x.dim() - 1),
        'mul': lambda x, y: torch.mul(x, y),
        'concat-mul-diff': lambda x, y: torch.cat((x, y, torch.mul(x, y), torch.abs(x - y)), x.dim() - 1)
    }
    def __init__(self, merge_type):
        super().__init__()    
        self.merge_type = merge_type
        self.op = Merge._style_map[merge_type]
        
    def forward(self, *args):
        return self.op(*args)

class Bypass(nn.Module):
    _supported_styles = ['residual', 'highway']

    @classmethod
    def supports_style(cls, style):
        return style.lower() in cls._supported_styles
    
    def __init__(self, style, residual_scale=True, highway_bias=-2, input_size=None):
        super().__init__()
        assert self.supports_style(style)
        self.style = style.lower()
        self.residual_scale = residual_scale
        self.highway_bias = highway_bias
        self.highway_gate = nn.Linear(input_size[1], input_size[0])
    
    def forward(self, transformed, raw):
        tsize = transformed.shape[-1]
        rsize = raw.shape[-1]
        adjusted_raw = raw
        if tsize < rsize:
            assert rsize / tsize <= 50
            if rsize % tsize != 0:
                padded = F.pad(raw, (0, tsize - rsize % tsize))
            else:
                padded = raw
            adjusted_raw = padded.view(*raw.shape[:-1], -1, tsize).sum(-2) * math.sqrt(
                tsize / rsize)
        elif tsize > rsize:
            multiples = math.ceil(tsize / rsize)
            adjusted_raw = raw.repeat(*([1] * (raw.dim() - 1)), multiples).narrow(
                -1, 0, tsize)
        
        if self.style == 'residual':
            res = transformed + raw
            if self.residual_scale:
                res *= math.sqrt(0.5)
            return res
        elif self.style == 'highway':
            gate = torch.sigmoid(self.highway_gate(raw) + self.highway_bias)
            return gate * transformed + (1 - gate) * adjusted_raw

class Transform(nn.Module):
    _supported_nonlinearities = [
        'sigmoid', 'tanh', 'relu', 'elu', 'selu', 'glu', 'leaky_relu'
    ]

    @classmethod
    def supports_nonlinearity(cls, nonlin):
        return nonlin.lower() in cls._supported_nonlinearities
    
    def __init__(self, transform_type, input_size, hidden_size, output_size=None):
        super().__init__()
        parts = transform_type.split('-')

        if 'layer' in parts:
            layers = int(parts[parts.index('layer') - 1])

        self.transforms = nn.ModuleList()
        self.bypass_networks = nn.ModuleList()

        for part in parts:
            if Bypass.supports_style(part):
                bypass_network = part
            if Transform.supports_nonlinearity(part):
                non_linearity = part

        transform_in_size = input_size
        transform_out_size = hidden_size
        for layer in range(layers):
            if layer == layers - 1:
                transform_out_size = output_size
            self.transforms.append(nn.Linear(transform_in_size, transform_out_size))
            self.bypass_networks.append(Bypass(bypass_network, input_size=(transform_out_size, transform_in_size)))
            transform_in_size = transform_out_size

    def forward(self, inputs):
        outputs = inputs
        for transform, bypass_network in zip(self.transforms, self.bypass_networks):
            new_outputs = transform(outputs)
            new_outputs = bypass_network(new_outputs, outputs) # new_outputs: 32 outputs: 256
            outputs = new_outputs
        return outputs

class Fusion(nn.Module):
    def __init__(self, merge="concat-mul-diff", transfrom="2-layer-highway", hidden_size=None):
        super().__init__()
        self.merge = Merge(merge)
        self.transform_network = Transform(transfrom, input_size=hidden_size * 4, hidden_size=hidden_size * 4, output_size=hidden_size)
    
    def forward(self, inputs, context):
        merged = self.merge(inputs, context)
        transformed = self.transform_network(merged)
        return transformed
    
class Model(nn.Module):
    def __init__(self, hidden_size, 
                 embedding_type="embedding",
                 encoder_type="cnn"):
        super().__init__()
        
        if embedding_type == "onehot":
            self.embedding = OnehotEmbedding(21)
            hidden_size = 8
        elif embedding_type == "embedding":
            self.embedding = nn.Embedding(21, 20, padding_idx=0)
            hidden_size = 8
            
        if encoder_type == "cnn":
            self.encoder = CNN()
            self.rnn_encoder = RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)
        elif encoder_type == "rnn":
            self.encoder = RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)
        
        self.self_attention = SelfAttention(hidden_size=hidden_size, alignment_network="dot")
        self.pair_attention = PairAttention(hidden_size=hidden_size, alignment_network='bilinear')
        self.word_fusion = Fusion(merge="concat-mul-diff", transfrom="2-layer-highway", hidden_size=hidden_size)
        self.gate_mechanism = GatingMechanism(hidden_size=hidden_size)
        self.global_attention = GlobalAttention(hidden_size=hidden_size)
        self.output_layer = nn.Linear(160, 160)
    
    def forward(self, left_inputs, right_inputs, left_masks, right_masks):
        batch_size = left_inputs.shape[0]
        
        left_sequences = self.embedding(left_inputs)
        right_sequences = self.embedding(right_inputs)
        
        left_contextualized = self.encoder(left_sequences, left_masks)
        right_contextualized = self.encoder(right_sequences, right_masks)
        
        left_sequences = left_contextualized
        right_sequences = right_contextualized
        
        left_masks = torch.ones_like(left_contextualized[:, :, 0]).bool()
        right_masks = torch.ones_like(right_contextualized[:, :, 0]).bool()

        left_contextualized = self.self_attention(left_contextualized, left_masks)
        right_contextualized = self.self_attention(right_contextualized, right_masks)

        left_fused = self.pair_attention(left_contextualized, right_contextualized, 
                                         left_sequences, right_sequences,
                                         left_masks, right_masks)
        right_fused = self.pair_attention(right_contextualized, left_contextualized,
                                          right_sequences, left_sequences,
                                          right_masks, left_masks)

        left_fused = self.word_fusion(left_contextualized, left_fused)
        right_fused = self.word_fusion(right_contextualized, right_fused)

        left_gated = self.gate_mechanism(left_sequences, left_fused)
        right_gated = self.gate_mechanism(right_sequences, right_fused)

        left_final = left_gated.view(batch_size, -1)
        right_final = right_gated.view(batch_size, -1)

        # left_final = self.global_attention(left_gated, left_masks)
        # right_final = self.global_attention(right_gated, right_masks)

        left_final = self.output_layer(left_final)
        right_final = self.output_layer(right_final)
        
        # return left_sequences, right_sequences
        return left_final, right_final
        # return left_contextualized, right_contextualized