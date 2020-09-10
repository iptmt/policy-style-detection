import torch
import torch.nn as nn
import torch.nn.functional as F

from tool import LossClock, embed_device
from data_util import align_texts
from vocab import PLH_ID, PAD_ID, EOS_ID


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
            x, y, pad_mask = embed_device([x, y, pad_mask], self.dev)
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
            x, y, pad_mask = embed_device([x, y, pad_mask], self.dev)
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
            x, y, pad_mask = embed_device([x, y, pad_mask], self.dev)
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
            x, x_, y = embed_device([x, x_, y], self.dev)
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


class InsertLMTrainer:
    def __init__(self, ilm, device, optimizer):
        self.ilm = ilm.to(device)
        self.dev = device
        self.optimizer = optimizer

        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.clock = LossClock(["Lv", "Lp"], 100)
    
    def train(self, dl):
        self.ilm.train()
        for iv, ov, lv, ip, op, lp in dl:
            iv, ov, lv, ip, op, lp = embed_device([iv, ov, lv, ip, op, lp], self.dev)
            logits_v, logits_p = self.ilm(iv, lv, ip, lp)
            if logits_v is not None:
                loss_v = self.ce(logits_v.reshape(-1, logits_v.size(-1)), ov.reshape(-1))
            else:
                loss_v = 0
            loss_p = self.ce(logits_p.reshape(-1, 2), op.reshape(-1))

            self.optimizer.zero_grad()
            (loss_v + loss_p).backward()
            self.optimizer.step()

            self.clock.update({"Lv": loss_v.item(), "Lp": loss_p.item()})
    
    def evaluate(self, dl):
        self.ilm.eval()
        losses = []
        for iv, ov, lv, ip, op, lp in dl:
            iv, ov, lv, ip, op, lp = embed_device([iv, ov, lv, ip, op, lp], self.dev)
            with torch.no_grad():
                logits_v, logits_p = self.ilm(iv, lv, ip, lp)
            if logits_v is not None:
                loss_v = self.ce(logits_v.reshape(-1, logits_v.size(-1)), ov.reshape(-1))
            else:
                loss_v = 0
            loss_p = self.ce(logits_p.reshape(-1, 2), op.reshape(-1))
            losses.append(loss_v.item() + loss_p.item())
        return sum(losses) / len(losses)
    
    def inference(self, dl, file_name, vocab):
        self.ilm.eval()
        file = open(file_name, "w+", encoding="utf-8")

        def insert_slots(tokens, positions, borders):
            new_input = []
            dev = tokens.device
            tokens, positions = tokens.cpu().numpy().tolist(), positions.cpu().numpy().tolist()
            for ts, ps, b in zip(tokens, positions, borders.unbind(0)):
                new_sentence = []
                for idx, (t, p) in enumerate(zip(ts, ps)):
                    new_sentence.append(t)
                    if idx >= b:
                        break
                    if p == 1:
                        new_sentence.append(PLH_ID)
                new_input.append(new_sentence)
            new_input = align_texts(new_input, pad_id=PAD_ID)
            return torch.tensor(new_input, dtype=torch.long, device=dev)

        for x, temp, label in dl:
            x, temp, label = embed_device([x, temp, label], self.dev)
            
            # out, mask = None, 1
            for _ in range(10):
                # generate slots
                with torch.no_grad():
                    _, logits_p = self.ilm(None, None, temp, 1 - label)
                borders = (temp == EOS_ID).long().max(dim=-1)[1]
                positions = logits_p.argmax(-1) # B, 2
                temp_m = insert_slots(temp, positions, borders)

                # fill in slots
                with torch.no_grad():
                    logits_v, _ = self.ilm(temp_m, 1 - label, None, None)
                temp = logits_v.argmax(-1)
            
            for s, s_tsf, lb in zip(x.unbind(0), temp.unbind(0), (1 - label).unbind(0)):
                s, s_tsf = vocab.tensor_to_sent(s), vocab.tensor_to_sent(s_tsf)
                file.write(
                    " ".join(s) + "\t" +\
                    " ".join(s_tsf) + "\t" +\
                    str(lb.item()) + "\n"
                )
        file.close()