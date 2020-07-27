import gc
import time
import subprocess
from sklearn.metrics import classification_report

from src.loader import *
from src.optimizer import Optimizer
from src.utils import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


class Trainer():
    def __init__(self, model, config, vocab, fitlog):
        self.model = model
        self.config = config
        self.fitlog = fitlog
        self.report = True

        self.train_data = get_examples(self.config.train_file, model.word_encoder, vocab)
        self.batch_num = int(np.ceil(len(self.train_data) / float(self.config.train_batch_size)))

        self.dev_data = get_examples(self.config.dev_file, model.word_encoder, vocab)

        # criterion
        if config.gamma > 0:
            weight = torch.FloatTensor(vocab.label_weights)
            weight = len(self.train_data) / weight
            weight = weight / torch.sum(weight)

            if config.use_cuda:
                weight = weight.to(config.device)

            self.criterion = nn.CrossEntropyLoss(weight)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # label name
        self.target_names = vocab.target_names

        # optimizer
        self.optimizer = Optimizer(model.all_parameters, config, self.batch_num)

        # count
        self.step = 0
        self.early_stop = -1
        self.best_train_f1, self.best_dev_f1 = 0, 0
        self.last_epoch = config.epochs + 1

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            gc.collect()
            train_f1 = self._train(epoch)
            self.logging_gpu_memory()

            gc.collect()
            dev_f1 = self._eval(epoch, "dev")
            self.logging_gpu_memory()

            if self.best_dev_f1 <= dev_f1:
                logging.info(
                    "Exceed history dev = %.2f, current dev = %.2f" % (self.best_dev_f1, dev_f1))
                if epoch > self.config.save_after:
                    torch.save(self.model.state_dict(), self.config.save_model)

                self.best_train_f1 = train_f1
                self.best_dev_f1 = dev_f1
                self.early_stop = 0
            else:
                self.early_stop += 1
                if self.early_stop == self.config.early_stops:
                    self.fitlog.add_best_metric(
                        {"Train": self.best_train_f1, "Dev": self.best_dev_f1,
                         "Epoch": epoch - self.config.early_stops})
                    logging.info(
                        "Eearly stop in epoch %d, best train: %.2f, dev: %.2f" % (
                            epoch - self.config.early_stops, self.best_train_f1, self.best_dev_f1))
                    self.last_epoch = epoch
                    break

    def test(self):
        self.model.load_state_dict(torch.load(self.config.save_model))
        test_batch_size = self.config.test_batch_size
        test_batch_size = 2 * test_batch_size if test_batch_size == 1 else test_batch_size // 2
        test_f1 = self._eval(self.last_epoch + 1, "test", test_batch_size=test_batch_size)

    def _train(self, epoch):
        self.optimizer.zero_grad()
        self.model.train()

        start_time = time.time()
        epoch_start_time = time.time()
        overall_losses = 0
        losses = 0
        batch_idx = 1
        y_pred = []
        y_true = []
        for batch_data in data_iter(self.train_data, self.config.train_batch_size, shuffle=True):
            torch.cuda.empty_cache()
            batch_inputs, batch_labels = self.batch2tensor(batch_data)
            batch_outputs = self.model(batch_inputs)
            loss = self.criterion(batch_outputs, batch_labels)
            loss = loss / self.config.update_every
            loss.backward()

            loss_value = loss.detach().cpu().item()
            losses += loss_value
            overall_losses += loss_value

            y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
            y_true.extend(batch_labels.cpu().numpy().tolist())

            if batch_idx % self.config.update_every == 0 or batch_idx == self.batch_num:
                nn.utils.clip_grad_norm_(self.optimizer.all_params, max_norm=self.config.clip)
                for optimizer, scheduler in zip(self.optimizer.optims, self.optimizer.schedulers):
                    optimizer.step()
                    scheduler.step()
                self.optimizer.zero_grad()

                self.step += 1

            if batch_idx % self.config.log_interval == 0:
                elapsed = time.time() - start_time

                lrs = self.optimizer.get_lr()
                logging.info(
                    '| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr{} | loss {:.4f} | s/batch {:.2f}'.format(
                        epoch, self.step, batch_idx, self.batch_num, lrs,
                        losses / self.config.log_interval,
                        elapsed / self.config.log_interval))

                start_time = time.time()
                corrects, totals, losses = 0, 0, 0

            batch_idx += 1

        overall_losses /= self.batch_num
        during_time = time.time() - epoch_start_time

        # reformat
        overall_losses = reformat(overall_losses, 4)
        score, f1 = get_score(y_true, y_pred)

        self.fitlog.add_loss(overall_losses, name="loss", step=epoch)
        self.fitlog.add_metric(f1, name="train", step=epoch)

        logging.info(
            '| epoch {:3d} | score {} | f1 {} | loss {:.4f} | time {:.2f}'.format(epoch, score, f1,
                                                                                          overall_losses,
                                                                                          during_time))
        if set(y_true) == set(y_pred) and self.report:
            report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
            logging.info('\n' + report)

        return f1

    def _eval(self, epoch, data_nane, test_batch_size=None):
        self.model.eval()

        start_time = time.time()

        if data_nane == "dev":
            data = self.dev_data
        elif data_nane == "test":
            data = self.test_data
        else:
            Exception("No name data.")

        if test_batch_size is None:
            test_batch_size = self.config.test_batch_size

        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_data in data_iter(data, test_batch_size, shuffle=False):
                torch.cuda.empty_cache()
                batch_inputs, batch_labels = self.batch2tensor(batch_data)
                batch_outputs = self.model(batch_inputs)
                y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())

            score, f1 = get_score(y_true, y_pred)

            during_time = time.time() - start_time
            self.fitlog.add_metric(f1, name=data_nane, step=epoch)
            logging.info(
                '| epoch {:3d} | {} | score {} | f1 {} | time {:.2f}'.format(epoch, data_nane, score, f1,
                                                                                during_time))
            if set(y_true) == set(y_pred) and self.report:
                report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
                logging.info('\n' + report)

            file = open(self.config.save_test + '.' + str(epoch), 'w')
            file.write(str(y_pred))
            file.write('\n')
            file.write(str(y_true))
            file.close()

        return f1

    def batch2tensor(self, batch_data):
        '''
            [[label, doc_len, [[sent_len, [sent_id0, ...], [sent_id1, ...]], ...]]
        '''
        batch_size = len(batch_data)
        doc_labels = []
        doc_lens = []
        doc_sent_lens = []
        doc_max_sent_len = []
        for doc_data in batch_data:
            doc_labels.append(doc_data[0])
            doc_lens.append(doc_data[1])
            sent_lens = [sent_data[0] for sent_data in doc_data[2]]
            doc_sent_lens.append(sent_lens)
            max_sent_len = max(sent_lens)
            doc_max_sent_len.append(max_sent_len)

        max_doc_len = max(doc_lens)
        max_sent_len = max(doc_max_sent_len)

        batch_inputs1 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_inputs2 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)

        masks_dtype = torch.uint8 if self.config.word_encoder == "bert" else torch.float32
        batch_masks = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=masks_dtype)
        batch_labels = torch.LongTensor(doc_labels)

        for b in range(batch_size):
            for sent_idx in range(doc_lens[b]):
                sent_data = batch_data[b][2][sent_idx]
                for word_idx in range(sent_data[0]):
                    batch_inputs1[b, sent_idx, word_idx] = sent_data[1][word_idx]
                    batch_inputs2[b, sent_idx, word_idx] = sent_data[2][word_idx]
                    batch_masks[b, sent_idx, word_idx] = 1

        if self.config.use_cuda:
            batch_inputs1 = batch_inputs1.to(self.config.device)
            batch_inputs2 = batch_inputs2.to(self.config.device)
            batch_masks = batch_masks.to(self.config.device)
            batch_labels = batch_labels.to(self.config.device)

        return (batch_inputs1, batch_inputs2, batch_masks), batch_labels

    def logging_gpu_memory(self):
        """
        Get the current GPU memory usage.
        Based on https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
        Returns
        -------
        ``Dict[int, int]``
            Keys are device ids as integers.
            Values are memory usage as integers in MB.
            Returns an empty ``dict`` if GPUs are not available.
        """
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,nounits,noheader"], encoding="utf-8",
            )
            info = [x.split(',') for x in result.strip().split("\n")]
            dic = {gpu: [int(mem[0]), int(mem[1])] for gpu, mem in enumerate(info)}
            gpu_id = self.config.gpu_id
            lst = dic[gpu_id]
            logging.info('| gpu id: {} | use {:5d}M / {:5d}M'.format(self.config.gpu_id, lst[0], lst[1]))

        except FileNotFoundError:
            # `nvidia-smi` doesn't exist, assume that means no GPU.
            return {}
        except:  # noqa
            # Catch *all* exceptions, because this memory check is a nice-to-have
            # and we'd never want a training run to fail because of it.
            logging.info("unable to check gpu_memory_mb(), continuing")
            return {}
