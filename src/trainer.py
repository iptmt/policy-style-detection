import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tool import LossClock
from data_util import align_texts
from vocab import PLH_ID, PAD_ID


class MaskTrainer:
    def __init__(self, masker, clf, device, rollouts, gamma, opt_masker, opt_clf):
        self.masker = masker.to(device)
        self.clf = clf.to(device)

        self.dev = device

        self.rollout_num = rollouts
        self.gamma = gamma

        self.bce = nn.BCELoss(reduction="none")

        self.optimize_masker = opt_masker

        self.optimize_clf_0 = opt_clf

        self.clock_clf = LossClock(["Loss_cls"], 500)
        self.clock = LossClock(["Loss", "R"], 200)

    def train(self, dl):
        self.masker.train()
        self.clf.eval()
        for _, (x, y, pad_mask) in enumerate(dl):
            # transfer into device
            x, y, pad_mask = x.to(self.dev), y.to(self.dev), pad_mask.to(self.dev)
            # optimize masker
            probs, rewards = self.masker(x, y, pad_mask, self.rollout_num, self.clf)
            loss_r, r_mean = self.cal_loss_by_rewards(probs, rewards)
            self.optimize_masker.zero_grad()
            loss_r.backward()
            self.optimize_masker.step()

            self.clock.update(
                {"Loss": loss_r.item(), "R": r_mean.item()}
            )

    def evaluate(self, dl):
        self.masker.eval()
        self.clf.eval()
        rewards = []
        for _, (x, y, pad_mask) in enumerate(dl):
            x, y, pad_mask = x.to(self.dev), y.to(self.dev), pad_mask.to(self.dev)
            with torch.no_grad():
                outputs = self.masker.sample_sequence(x, y, pad_mask, 1, self.clf)
                for sent_set in outputs:
                    for _, r in sent_set:
                        rewards.append(r.item())
        return sum(rewards)/len(rewards)

    def inference(self, dl, file_name, vocab):
        self.masker.eval()
        self.clf.eval()
        f_obj = open(file_name, 'w+', encoding="utf-8")
        for _, (x, y, pad_mask) in enumerate(dl):
            x, y, pad_mask = x.to(self.dev), y.to(self.dev), pad_mask.to(self.dev)
            self.masker.eval()
            self.clf.eval()
            with torch.no_grad():
                outputs = self.masker.sample_sequence(x, y, pad_mask, 8, self.clf)
                for id_bat, sent_set in enumerate(outputs):
                    sent_set = list(sent_set)
                    sent_set.sort(key=lambda pair: pair[1].item(), reverse=True)
                    o = vocab.tensor_to_template(x[id_bat, :], pad_mask[id_bat, :])
                    m = vocab.tensor_to_template(sent_set[0][0], pad_mask[id_bat, :])
                    f_obj.write(o + "\t" + m + "\t" + str(y[id_bat].item()) + "\n")
        f_obj.close()

    def train_clf(self, dl):
        self.clf.train()
        for x, x_, y in dl:
            x, x_, y = x.to(self.dev), x_.to(self.dev), y.to(self.dev)
            pred = self.clf(x_)

            # weighted loss
            weights = (x_ != PAD_ID).sum(1).float() / (x != PAD_ID).sum(1).float()

            loss_cls_N = self.bce(pred, y.float()) * weights
            loss_cls = loss_cls_N.mean()

            self.optimize_clf_0.zero_grad()
            loss_cls.backward()
            self.optimize_clf_0.step()

            self.clock_clf.update({"Loss_cls": loss_cls.item()})

    def eval_clf(self, dl):
        self.clf.eval()
        hits, total = 0, 0
        for x, y, _ in dl:
            x, y = x.to(self.dev), y.to(self.dev)
            with torch.no_grad():
                pred = self.clf(x)
            pred_lb = (pred > 0.5).long()
            hits += torch.eq(pred_lb, y).sum().item()
            total += x.size(0)
        return hits / total


    def cal_loss_by_rewards(self, probs, rewards):
        r_mean = torch.stack(rewards, dim=1).mean()
        returns, R = [], 1.0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns, dim=1)
        returns = (returns - returns.mean(dim=1).unsqueeze(1))
        probs = torch.stack(probs, dim=1)
        return -(probs * returns).sum(dim=1).mean(), r_mean


class MLMTrainer:
    def __init__(self, mlm, device, optimizer):
        self.mlm = mlm.to(device)
        self.dev = device
        self.optimizer = optimizer

        self.ce = nn.CrossEntropyLoss()
        self.clock = LossClock(["Loss_mlm"], 200)

    def train(self, dl):
        self.mlm.train()
        for x, x_, y in dl:
            x, x_, y = x.to(self.dev), x_.to(self.dev), y.to(self.dev)
            logits = self.mlm(x_, y)

            loss = self.ce(logits.reshape(-1, logits.size(-1)), x.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.clock.update({"Loss_mlm": loss.item()})

    def evaluate(self, dl):
        losses = []
        self.mlm.eval()
        for x, x_, y in dl:
            x, x_, y = x.to(self.dev), x_.to(self.dev), y.to(self.dev)
            with torch.no_grad():
                logits = self.mlm(x_, y)
            loss = self.ce(logits.reshape(-1, logits.size(-1)), x.reshape(-1))
            losses.append(loss.item())
        return sum(losses) / len(losses)

    def rank_words(self, dl, file_name, vocab):
        file = open(file_name, "w+", encoding="utf-8")
        loss_cal = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        self.mlm.eval()
        for x, temp, ph_mask, y in dl:
            time.sleep(0.05)
            x, temp, ph_mask, y = x.to(self.dev), temp.to(self.dev), ph_mask.to(self.dev), y.to(self.dev)
            orders = ph_mask.new_zeros(ph_mask.shape)
            step = 1
            while True:
                if ph_mask.sum() == 0:
                    break
                with torch.no_grad():
                    logits = self.mlm(temp, y)
                    loss = loss_cal(logits.reshape(-1, logits.size(-1)), (x * ph_mask).reshape(-1)).reshape(ph_mask.shape)
                loss += (loss == 0).float() * 1e4
                _, indexes = loss.min(dim=1)
                step_mask = ph_mask.new_zeros(ph_mask.shape)
                step_mask[torch.arange(step_mask.size(0)),indexes] = 1
                step_mask = step_mask * ph_mask
                # step_mask = ((loss - best_values.unsqueeze(1)) == 0).long() * ph_mask
                # update temp, ph_mask
                temp = step_mask * x + (1 - step_mask) * temp
                ph_mask = ph_mask - step_mask
                # update order matrix
                orders += step * step_mask
                step += 1
            for sent, order, label in zip(x.unbind(0), orders.unbind(0), y.unbind(0)):
                sent = vocab.tensor_to_sent(sent)
                file.write(
                    " ".join(sent) + "\t" +\
                    " ".join([str(rk) for rk in order.cpu().numpy()[:len(sent)]]) + "\t" +\
                    str(label.item()) + "\n"
                )
        file.close()


class InsertLMTrainer:
    def __init__(self, ilm, device, optimizer):
        self.ilm = ilm.to(device)
        self.dev = device
        self.optimizer = optimizer

        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.mse = nn.MSELoss(reduction="none")
        self.clock = LossClock(["Loss_vocab", "Loss_position"], 200)

    def train(self, dl):
        self.ilm.train()
        for inp, out, pos, label in dl:
            inp, out, pos, label = inp.to(self.dev), out.to(self.dev),\
                 pos.to(self.dev), label.to(self.dev)
            logits_v, logits_p = self.ilm(inp, label)
            loss_v = self.ce(logits_v.reshape(-1, logits_v.size(-1)), out.reshape(-1))
            loss_p = self.mse(logits_p, pos).sum(-1).mean()

            self.optimizer.zero_grad()
            (loss_v + loss_p).backward()
            self.optimizer.step()

            self.clock.update({"Loss_vocab": loss_v.item(), "Loss_position": loss_p.item()})
            time.sleep(0.03) # prevent the overheat of GPU

    def evaluate(self, dl):
        self.ilm.eval()
        losses = []
        for inp, out, pos, label in dl:
            inp, out, pos, label = inp.to(self.dev), out.to(self.dev),\
                 pos.to(self.dev), label.to(self.dev)
            with torch.no_grad():
                logits_v, logits_p = self.ilm(inp, label)
            loss_v = self.ce(logits_v.reshape(-1, logits_v.size(-1)), out.reshape(-1))
            loss_p = self.mse(logits_p, pos).sum(-1).mean()
            losses.append(loss_v.item() + loss_p.item())
        return sum(losses) / len(losses)

    def transfer(self, dl, file_name, vocab):
        self.ilm.eval()
        file = open(file_name, "w+", encoding="utf-8")
        def split_and_merge(tokens, positions, mask):
            new_input = []
            dev = tokens.device
            tokens = tokens.cpu().numpy().tolist()
            for ts, ps, m in zip(tokens, positions.unbind(0), mask.unbind(0)):
                ps, m = ps.item(), m.item()
                if m == 0 or ts[ps] == PAD_ID:
                    new_input.append(ts)
                    continue
                ts = ts[:ps] + [PLH_ID] + ts[ps:]
                ts = list(filter(lambda t: t != PAD_ID, ts))
                new_input.append(ts)
            new_input = align_texts(new_input, pad_id=PAD_ID)
            return torch.tensor(new_input, dtype=torch.long, device=dev)

        for x, temp, label in dl:
            x, temp, label = x.to(self.dev), temp.to(self.dev), label.to(self.dev)
            out, mask = None, 1
            for _ in range(10):
                with torch.no_grad():
                    logits_v, logits_p = self.ilm(temp, 1 - label)
                out = logits_v.argmax(-1) # B, L
                positions = logits_p.argmax(-1) # B,
                mask = (positions != 0).long() * mask
                temp = split_and_merge(out, positions, mask)

            for s, s_tsf, lb in zip(x.unbind(0), out.unbind(0), label.unbind(0)):
                s, s_tsf = vocab.tensor_to_sent(s), vocab.tensor_to_sent(s_tsf)
                file.write(
                    " ".join(s) + "\t" +\
                    " ".join(s_tsf) + "\t" +\
                    str((1 - lb).item()) + "\n"
                )
        file.close()
