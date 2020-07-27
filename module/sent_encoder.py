from module.layers import *


class SentEncoder(nn.Module):
    def __init__(self, config, sent_rep_size):
        super(SentEncoder, self).__init__()
        self.dropout_mlp = config.dropout_mlp

        self.sent_lstm = LSTM(
            input_size=sent_rep_size,
            hidden_size=config.sent_hidden_size,
            num_layers=config.sent_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_input,
            dropout_out=config.dropout_hidden,
        )

    def forward(self, sent_reps, sent_masks):
        # sent_reps:  b x doc_len x sent_rep_size
        # sent_masks: b x doc_len

        sent_hiddens, _ = self.sent_lstm(sent_reps, sent_masks)  # doc_len x b x hidden*2
        sent_hiddens.transpose_(1, 0)  # b x doc_len x hidden*2

        if self.training:
            sent_hiddens = drop_sequence_sharedmask(sent_hiddens, self.dropout_mlp)

        return sent_hiddens
