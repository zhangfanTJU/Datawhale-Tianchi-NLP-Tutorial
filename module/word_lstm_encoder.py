import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

from module.layers import *


class WordLSTMEncoder(nn.Module):
    def __init__(self, config, vocab):
        super(WordLSTMEncoder, self).__init__()
        self.dropout_embed = config.dropout_embed
        self.dropout_mlp = config.dropout_mlp

        word_embed = np.zeros((vocab.word_size, config.word_dims), dtype=np.float32)
        self.word_embed = nn.Embedding(vocab.word_size, config.word_dims, padding_idx=0)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_embed))

        extword_embed = vocab.load_pretrained_embs(config.word2vec_path)
        extword_size, word_dims = extword_embed.shape
        logging.info("Load extword embed: words %d, dims %d." % (extword_size, word_dims))

        self.extword_embed = nn.Embedding(extword_size, word_dims, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        self.extword_embed.weight.requires_grad = False

        input_size = config.word_dims

        self.word_lstm = LSTM(
            input_size=input_size,
            hidden_size=config.word_hidden_size,
            num_layers=config.word_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_input,
            dropout_out=config.dropout_hidden,
        )

    def forward(self, word_ids, extword_ids, batch_masks):
        # word_ids: sen_num x sent_len
        # extword_ids: sen_num x sent_len
        # batch_masks   sen_num x sent_len

        word_embed = self.word_embed(word_ids)  # sen_num x sent_len x 100
        extword_embed = self.extword_embed(extword_ids)
        batch_embed = word_embed + extword_embed

        if self.training:
            batch_embed = drop_input_independent(batch_embed, self.dropout_embed)

        hiddens, _ = self.word_lstm(batch_embed, batch_masks)  # sent_len x sen_num x hidden*2
        hiddens.transpose_(1, 0)  # sen_num x sent_len x hidden*2

        if self.training:
            hiddens = drop_sequence_sharedmask(hiddens, self.dropout_mlp)

        return hiddens
