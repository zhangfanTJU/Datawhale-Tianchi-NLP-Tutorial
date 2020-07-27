import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

from module import *
from module.layers import *


class Model(nn.Module):
    def __init__(self, config, vocab):
        super(Model, self).__init__()
        model2size = {'lstm': config.word_hidden_size * 2, 'bert': 256, 'cnn': 300}
        self.sent_rep_size = model2size[config.word_encoder]
        self.doc_rep_size = config.sent_hidden_size * 2 if config.sent_encoder == 'lstm' else self.sent_rep_size
        self.all_parameters = {}
        parameters = []
        bert_parameters = None
        if config.word_encoder == 'bert':
            self.word_encoder = WordBertEncoder(config)
            bert_parameters = self.word_encoder.get_bert_parameters()
        elif config.word_encoder == 'cnn':
            self.word_encoder = WordCNNEncoder(config, vocab)
            parameters.extend(list(filter(lambda p: p.requires_grad, self.word_encoder.parameters())))
        elif config.word_encoder == 'lstm':
            self.word_encoder = WordLSTMEncoder(config, vocab)
            self.word_attention = Attention(self.sent_rep_size, config.dropout_mlp)
            parameters.extend(list(filter(lambda p: p.requires_grad, self.word_encoder.parameters())))
            parameters.extend(list(filter(lambda p: p.requires_grad, self.word_attention.parameters())))

        self._sent_encoder = config.sent_encoder
        if config.sent_encoder == 'lstm':
            self.sent_encoder = SentEncoder(config, self.sent_rep_size)
            self.sent_attention = Attention(self.doc_rep_size, config.dropout_mlp)
            parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_encoder.parameters())))
            parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))
        elif config.sent_encoder == 'atten':
            self.sent_attention = Attention(self.doc_rep_size, config.dropout_mlp)
            parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))
        else:
            self.sent_encoder = None

        self.out = NoLinear(self.doc_rep_size, vocab.label_size, bias=True)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))

        if config.use_cuda:
            self.to(config.device)

        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters

        if bert_parameters is not None:
            self.all_parameters["bert_parameters"] = bert_parameters

        logging.info('Build model with {} word encoder, {} sent encoder.'.format(config.word_encoder,
                                                                                 config.sent_encoder))
        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        logging.info('Model param num: %.2f M.' % (para_num / 1e6))

    def forward(self, batch_inputs):
        # batch_inputs(batch_inputs1, batch_inputs2): b x doc_len x sent_len
        # batch_masks : b x doc_len x sent_len
        batch_inputs1, batch_inputs2, batch_masks = batch_inputs
        batch_size, max_doc_len, max_sent_len = batch_inputs1.shape[0], batch_inputs1.shape[1], batch_inputs1.shape[2]
        batch_inputs1 = batch_inputs1.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_inputs2 = batch_inputs2.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_masks = batch_masks.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len

        if isinstance(self.word_encoder, WordLSTMEncoder):
            batch_hiddens = self.word_encoder(batch_inputs1, batch_inputs2,
                                              batch_masks)  # sen_num x sent_len x sent_rep_size
            sent_reps, atten_scores = self.word_attention(batch_hiddens, batch_masks)  # sen_num x sent_rep_size
        else:
            sent_reps = self.word_encoder(batch_inputs1, batch_inputs2)  # sen_num x sent_rep_size

        sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_rep_size)  # b x doc_len x sent_rep_size
        batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)  # b x doc_len x max_sent_len
        sent_masks = batch_masks.bool().any(2).float()  # b x doc_len

        if self._sent_encoder == 'lstm':
            sent_hiddens = self.sent_encoder(sent_reps, sent_masks)  # b x doc_len x doc_rep_size
            doc_reps, atten_scores = self.sent_attention(sent_hiddens, sent_masks)  # b x doc_rep_size
        elif self._sent_encoder == 'atten':
            doc_reps, atten_scores = self.sent_attention(sent_reps, sent_masks)  # b x doc_rep_size
        else:
            avg_sent_masks = sent_masks / torch.sum(sent_masks, 1, True)  # b x doc_len
            doc_reps = torch.bmm(avg_sent_masks.unsqueeze(1), sent_reps).squeeze(1)  # b x doc_rep_size

        batch_outputs = self.out(doc_reps)  # b x num_labels

        return batch_outputs
