import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]

        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


# class BiLSTM_Attention(nn.Module):
#     def __init__(self):
#         super(BiLSTM_Attention, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
#         self.out = nn.Linear(n_hidden * 2, num_classes)

# lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
def attention_net(lstm_output, final_state):
    batch_size = len(lstm_output)
    hidden = final_state.view(batch_size, -1, 1)
    # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
    attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
    soft_attn_weights = F.softmax(attn_weights, 1)

    # context : [batch_size, n_hidden * num_directions(=2)]
    context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
    return context, soft_attn_weights

    # def forward(self, X):
    #     '''
    #     X: [batch_size, seq_len]
    #     '''
    #     input = self.embedding(X)  # input : [batch_size, seq_len, embedding_dim]
    #     input = input.transpose(0, 1)  # input : [seq_len, batch_size, embedding_dim]
    #
    #     # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
    #     output, (final_hidden_state, final_cell_state) = self.lstm(input)
    #     output = output.transpose(0, 1)  # output : [batch_size, seq_len, n_hidden]
    #     attn_output, attention = self.attention_net(output, final_hidden_state)
    #     return self.out(attn_output), attention  # model : [batch_size, num_classes], attention : [batch_size, n_step]
