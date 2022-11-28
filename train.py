import config
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict


class PytorchTrainer:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.args.learning_rate, momentum=0.9)

    def train(self):
        # n_epochs = 20
        # learning_rate = 1e-3
        # weight_decay = 5e-4
        # use_scheduler = True

        metrics_dict = {
            'train': {
                'acc': [],
                'loss': [],
            },
            'val': {
                'acc': [],
                'loss': [],
            },
            'test': {
                'acc': [],
                'loss': [],
            }

        }

        train_num = len(train_data)
        valid_num = len(val_data)

        best_loss, best_acc, best_epoch = None, None, None

        for epoch in range(config.args.epochs):
            losses_train = []
            losses_valid = []
            train_true_num = 0
            valid_true_num = 0

            processbar = tqdm(total=(train_size // config.args.batch_size + 1))
            processbar.set_description("Epoch %02d" % (epoch + 1))
            self.model.train()  # 訓練時には勾配を計算するtrainモードにする
            for x, t in dataloader_train:
                self.optimizer.zero_grad()
                pred = self.model(x.to(self.device))

                loss = self.loss_criterion(pred.to('cpu'), t)
                loss.backward()
                self.optimizer.step()
                losses_train.append(loss.tolist())
                acc = torch.where(t - pred.argmax(1).to("cpu") == 0, torch.ones_like(t), torch.zeros_like(t))

                train_true_num += acc.sum().item()
                processbar.set_postfix(OrderedDict(loss=loss.tolist(), acc=(acc.sum().item() / acc.size()[0])))
                processbar.update(1)

            self.model.eval()  # 評価時には勾配を計算しないevalモードにする
            for x, t in dataloader_valid:
                pred = self.model(x.to(self.device))
                loss = self.loss_criterion(pred.to('cpu'), t)
                losses_valid.append(loss.tolist())
                acc = torch.where(t - pred.argmax(1).to("cpu") == 0, torch.ones_like(t), torch.zeros_like(t))

                valid_true_num += acc.sum().item()

            current_lr = config.args.learning_rate

            train_loss = np.mean(losses_train)
            train_acc = train_true_num / train_num
            val_loss = np.mean(losses_valid)
            val_acc = valid_true_num / valid_num

            metrics_dict['train']['acc'].append(train_acc)
            metrics_dict['train']['loss'].append(train_loss)
            metrics_dict['val']['acc'].append(val_acc)
            metrics_dict['val']['loss'].append(val_loss)

            print(
                'EPOCH: {}, LR: {} Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
                    epoch + 1,
                    current_lr,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                ))

            if (best_loss is None and best_acc is None) or (best_loss > val_loss and best_acc < val_acc):
                torch.save(self.model.state_dict(), "best_performing_model")
                print(f"saved model at epoch {epoch + 1}")
                best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            # else:
            #     self.model.load_state_dict(torch.load("best_performing_model"))
            #     print(f"loaded model from epoch {best_epoch+1}")
        pass

    def save(self):
        pass