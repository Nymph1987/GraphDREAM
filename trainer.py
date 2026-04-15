import numpy as np, random
import torch
from sklearn.metrics import f1_score, accuracy_score, r2_score
import torch.nn.functional as F
from module import build_match_sen_shift_label
from utils import AutomaticWeightedLoss

seed = 2024


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SupConLoss(torch.nn.Module):

    def __init__(self,
                 temperature=0.07,
                 base_temperature=0.07,
                 eps=1e-8,
                 hard_negative_weight=2.0,
                 hard_pairs=((0, 4), (2, 5))):
        super(SupConLoss, self).__init__()
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)
        self.eps = float(eps)
        self.hard_negative_weight = float(hard_negative_weight)
        self.hard_pairs = tuple(hard_pairs)

    def forward(self, features, labels):
        if features.dim() == 2:
            features = features.unsqueeze(1)
        if features.dim() != 3:
            raise ValueError('features must be [N, V, D] or [N, D]')

        batch_size = features.size(0)
        view_count = features.size(1)
        device = features.device
        features = F.normalize(features, p=2, dim=-1)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = view_count

        labels = labels.contiguous().view(-1, 1)
        if labels.size(0) != batch_size:
            raise ValueError('labels size mismatch')
        mask = torch.eq(labels, labels.T).float().to(device)
        labels_flat = labels.view(-1)
        hard_mask = torch.zeros((batch_size, batch_size),
                                dtype=torch.bool,
                                device=device)
        for a, b in self.hard_pairs:
            hard_mask = hard_mask | ((labels_flat.unsqueeze(1) == a) &
                                     (labels_flat.unsqueeze(0) == b))
            hard_mask = hard_mask | ((labels_flat.unsqueeze(1) == b) &
                                     (labels_flat.unsqueeze(0) == a))

        temp = max(self.temperature, self.eps)
        logits = torch.matmul(anchor_feature, contrast_feature.T) / temp
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        mask = mask.repeat(anchor_count, view_count)
        hard_mask = hard_mask.repeat(anchor_count, view_count)
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0.0,
        )
        mask = mask * logits_mask

        neg_weight = torch.ones_like(mask)
        neg_only = (mask < 0.5) & (logits_mask > 0)
        neg_weight[hard_mask & neg_only] = self.hard_negative_weight
        exp_logits = torch.exp(logits) * logits_mask * neg_weight
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps))

        pos_count = mask.sum(dim=1)
        valid = pos_count > 0
        if valid.sum() == 0:
            return logits.new_zeros(())

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_count.clamp_min(
            1.0)
        loss = -(temp / max(self.base_temperature, self.eps)
                 ) * mean_log_prob_pos[valid]
        return loss.mean()


def aux_weight_scheduler(epoch,
                         base_gamma,
                         base_lambda,
                         peak_epochs=5,
                         peak_scale=1.5):
    if peak_epochs <= 0:
        return float(base_gamma), float(base_lambda)
    if epoch + 1 <= peak_epochs:
        ratio = (epoch + 1) / float(peak_epochs)
        scale = peak_scale - (peak_scale - 1.0) * ratio
    else:
        scale = 1.0
    return float(base_gamma) * scale, float(base_lambda) * scale


def class_balanced_weights(counts, num_classes, beta=0.9999, eps=1e-8):
    c = counts.float().clamp_min(1.0)
    effective_num = 1.0 - torch.pow(torch.tensor(beta, device=c.device), c)
    w = (1.0 - beta) / effective_num.clamp_min(eps)
    w = w / w.sum().clamp_min(eps) * num_classes
    return w.float()


def contrastive_margin_loss_pair(feats,
                                 labels,
                                 class_a=0,
                                 class_b=4,
                                 margin=0.25):
    if feats is None or feats.numel() == 0:
        return feats.new_zeros(()) if feats is not None else torch.tensor(0.0)
    feats = F.normalize(feats, p=2, dim=-1)
    mask_a = labels == class_a
    mask_b = labels == class_b
    if (mask_a.sum() == 0) or (mask_b.sum() == 0):
        return feats.new_zeros(())
    fa = feats[mask_a]
    fb = feats[mask_b]
    sim_ab = torch.matmul(fa, fb.t())
    loss_neg = F.relu(sim_ab - margin).mean()
    loss_pos_a = (1.0 - torch.matmul(fa, fa.t())).mean()
    loss_pos_b = (1.0 - torch.matmul(fb, fb.t())).mean()
    return loss_neg + 0.5 * (loss_pos_a + loss_pos_b)


def train_or_eval_model(
    model,
    loss_function_emo,
    loss_function_sen,
    loss_function_shift,
    dataloader,
    epoch,
    cuda,
    modals,
    optimizer=None,
    train=False,
    dataset='IEMOCAP',
    loss_type='',
    lambd=[1.0, 1.0, 1.0],
    epochs=100,
    classify='',
    shift_win=5,
    contrastive_weight=0.05,
    decouple_weight=0.02,
    warmup_epochs=8,
    supcon_temperature=0.07,
    label_smoothing=0.1,
    class_balance_beta=0.9999,
    hard_negative_weight=2.0,
    pair_aux_weight_04=0.20,
    pair_aux_weight_25=0.20,
    proto_aux_weight=0.10,
    hier_aux_weight=0.12,
    hier_kl_weight=0.05,
    temporal_aux_weight=0.15,
    modality_adv_weight=0.05,
    pair04_sample_boost=1.5,
    pair04_margin=0.25,
    pair04_margin_weight=0.10,
    collect_features=False,
):
    losses, preds_emo, labels_emo = [], [], []
    adv_loss_values = []
    preds_sft, labels_sft = [], []
    preds_sen, labels_sen = [], []
    vids = []
    initial_feats, extracted_feats = [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()
    core_model = model.module if hasattr(model, 'module') else model

    supcon_criterion = SupConLoss(temperature=supcon_temperature,
                                  base_temperature=supcon_temperature,
                                  hard_negative_weight=hard_negative_weight)
    p = float(epoch + 1) / float(max(1, epochs))
    alpha = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
    if float(modality_adv_weight) <= 0.0:
        alpha = 0.0
    if hasattr(core_model, 'set_grl_alpha'):
        core_model.set_grl_alpha(alpha)
    if not hasattr(train_or_eval_model, '_emo_counts'):
        train_or_eval_model._emo_counts = None
    if not hasattr(train_or_eval_model, '_emo_weights'):
        train_or_eval_model._emo_weights = None
    for iter, data in enumerate(dataloader):

        if train:
            optimizer.zero_grad(set_to_none=True)

        textf0, textf1, textf2, textf3, visuf, acouf, qmask, umask, label_emotion, label_sentiment = data[:-1]
        dia_lengths = umask.sum(dim=0).long().tolist()
        if cuda:
            textf0 = textf0.cuda(non_blocking=True)
            textf1 = textf1.cuda(non_blocking=True)
            textf2 = textf2.cuda(non_blocking=True)
            textf3 = textf3.cuda(non_blocking=True)
            visuf = visuf.cuda(non_blocking=True)
            acouf = acouf.cuda(non_blocking=True)
            umask = umask.cuda(non_blocking=True)
            label_emotion = label_emotion.cuda(non_blocking=True)
            label_sentiment = label_sentiment.cuda(non_blocking=True)

        label_emotions, label_sentiments = [], []
        for j in range(len(dia_lengths)):
            label_emotions.append(label_emotion[:dia_lengths[j], j])
            label_sentiments.append(label_sentiment[:dia_lengths[j], j])
        label_emo = torch.cat(label_emotions)
        label_sen = torch.cat(label_sentiments)
        if classify == 'regression':
            label_emo = label_emo.float()

        if train:
            logit_emo, logit_sen, logit_sft, extracted_feature, feat_contrast, loss_decouple, aux_outputs = model(
                textf0, textf1, textf2, textf3, visuf, acouf, umask, qmask,
                dia_lengths)
        else:
            with torch.no_grad():
                logit_emo, logit_sen, logit_sft, extracted_feature, feat_contrast, loss_decouple, aux_outputs = model(
                    textf0, textf1, textf2, textf3, visuf, acouf, umask,
                    qmask, dia_lengths)
        assert logit_emo.size(0) == label_emo.size(0)
        if classify == 'regression':
            pred_reg = logit_emo.view(-1)
            target_reg = label_emo.view(-1)
            loss = F.smooth_l1_loss(pred_reg, target_reg)
            preds_emo.append(pred_reg.detach())
            labels_emo.append(target_reg.detach())
            vids += data[-1]
            losses.append(loss.detach())
            adv_loss_values.append(logit_emo.new_zeros(()))

            if train:
                loss.backward()
                optimizer.step()

            if collect_features:
                extracted_feats.append(extracted_feature.detach())
            continue

        if classify == 'unsupervised':
            adv_logits = aux_outputs.get('adv_logits', None)
            adv_labels = aux_outputs.get('adv_labels', None)
            if adv_logits is not None and adv_labels is not None and len(
                    adv_logits) == len(adv_labels) and len(adv_logits) > 0:
                adv_losses = []
                for logit_adv, label_adv in zip(adv_logits, adv_labels):
                    if logit_adv is None or label_adv is None:
                        continue
                    if logit_adv.size(0) == 0:
                        continue
                    adv_losses.append(F.cross_entropy(logit_adv, label_adv))
                if len(adv_losses) > 0:
                    adv_losses_tensor = torch.stack(adv_losses)
                    loss_adv = adv_losses_tensor.mean()
                else:
                    loss_adv = logit_emo.new_zeros(())
            else:
                loss_adv = logit_emo.new_zeros(())

            decouple_weight_eff = float(decouple_weight)
            if decouple_weight_eff <= 0.0:
                decouple_weight_eff = 1.0
            adv_weight_eff = float(modality_adv_weight) * alpha
            if epoch + 1 > 30:
                adv_weight_eff = adv_weight_eff * 1.1
            loss = decouple_weight_eff * loss_decouple + adv_weight_eff * loss_adv
            vids += data[-1]
            losses.append(loss.detach())
            adv_loss_values.append(loss_adv.detach())

            if train:
                loss.backward()
                optimizer.step()

            if collect_features:
                extracted_feats.append(extracted_feature.detach())
            continue

        num_classes_emo = int(logit_emo.size(-1))
        if train_or_eval_model._emo_counts is None or train_or_eval_model._emo_counts.numel(
        ) != num_classes_emo:
            train_or_eval_model._emo_counts = torch.ones(num_classes_emo,
                                                         device=label_emo.device)
            train_or_eval_model._emo_weights = torch.ones(num_classes_emo,
                                                          device=label_emo.device)
        if train:
            batch_counts = torch.bincount(label_emo,
                                          minlength=num_classes_emo).float()
            train_or_eval_model._emo_counts = train_or_eval_model._emo_counts.to(
                batch_counts.device) + batch_counts
            train_or_eval_model._emo_weights = class_balanced_weights(
                train_or_eval_model._emo_counts,
                num_classes_emo,
                beta=class_balance_beta).to(label_emo.device)
        emo_weights = train_or_eval_model._emo_weights.to(label_emo.device)
        if train and hasattr(core_model, 'update_emotion_prototypes'):
            core_model.update_emotion_prototypes(extracted_feature, label_emo)

        prob_emo = F.log_softmax(logit_emo, -1)
        loss_emo = F.cross_entropy(logit_emo,
                                   label_emo,
                                   weight=emo_weights,
                                   label_smoothing=label_smoothing)
        prob_sen = F.log_softmax(logit_sen, -1)
        loss_sen = F.cross_entropy(logit_sen,
                                   label_sen,
                                   label_smoothing=label_smoothing)
        prob_sft = F.log_softmax(logit_sft, -1)
        label_sft = build_match_sen_shift_label(shift_win, dia_lengths,
                                                label_sen)
        loss_sft = F.cross_entropy(logit_sft,
                                   label_sft,
                                   label_smoothing=label_smoothing)
        loss_supcon = supcon_criterion(feat_contrast, label_emo)

        pair_logit_04 = aux_outputs['pair_logit_04']
        pair_logit_25 = aux_outputs['pair_logit_25']
        pair_feat_04 = aux_outputs.get('pair_feat_04', None)
        proto_logits = aux_outputs['proto_logits']
        coarse_from_emo = aux_outputs.get('coarse_from_emo', None)
        temp_logit_emo = aux_outputs.get('temp_logit_emo', None)
        adv_logits = aux_outputs.get('adv_logits', None)
        adv_labels = aux_outputs.get('adv_labels', None)
        mask_04 = (label_emo == 0) | (label_emo == 4)
        mask_25 = (label_emo == 2) | (label_emo == 5)
        valid_pair04 = pair_logit_04 is not None and mask_04.any() and (
            label_emo[mask_04] == 4).any() and (label_emo[mask_04] == 0).any()
        valid_pair25 = pair_logit_25 is not None and mask_25.any() and (
            label_emo[mask_25] == 5).any() and (label_emo[mask_25] == 2).any()
        if valid_pair04:
            target_04 = (label_emo[mask_04] == 4).long()
            loss_pair_04_raw = F.cross_entropy(pair_logit_04[mask_04],
                                               target_04,
                                               label_smoothing=label_smoothing,
                                               reduction='none')
            loss_pair_04 = (pair04_sample_boost * loss_pair_04_raw).mean()
        else:
            loss_pair_04 = logit_emo.new_zeros(())
        if valid_pair25:
            target_25 = (label_emo[mask_25] == 5).long()
            loss_pair_25 = F.cross_entropy(pair_logit_25[mask_25],
                                           target_25,
                                           label_smoothing=label_smoothing)
        else:
            loss_pair_25 = logit_emo.new_zeros(())
        if valid_pair04 and pair_feat_04 is not None:
            loss_pair04_margin = contrastive_margin_loss_pair(
                pair_feat_04[mask_04],
                label_emo[mask_04],
                class_a=0,
                class_b=4,
                margin=pair04_margin)
        else:
            loss_pair04_margin = logit_emo.new_zeros(())
        if proto_logits is not None:
            loss_proto = F.cross_entropy(proto_logits,
                                         label_emo,
                                         weight=emo_weights,
                                         label_smoothing=label_smoothing)
        else:
            loss_proto = logit_emo.new_zeros(())
        if coarse_from_emo is not None:
            loss_hier_ce = F.cross_entropy(coarse_from_emo,
                                           label_sen,
                                           label_smoothing=label_smoothing)
            dist_emo = F.softmax(coarse_from_emo, dim=-1).detach()
            dist_sen = F.softmax(logit_sen, dim=-1)
            loss_hier_kl = F.kl_div(F.log_softmax(logit_sen, dim=-1),
                                    dist_emo,
                                    reduction='batchmean') + F.kl_div(
                                        F.log_softmax(coarse_from_emo, dim=-1),
                                        dist_sen.detach(),
                                        reduction='batchmean')
        else:
            loss_hier_ce = logit_emo.new_zeros(())
            loss_hier_kl = logit_emo.new_zeros(())
        if temp_logit_emo is not None:
            loss_temp = F.cross_entropy(temp_logit_emo,
                                        label_emo,
                                        weight=emo_weights,
                                        label_smoothing=label_smoothing)
        else:
            loss_temp = logit_emo.new_zeros(())
        if adv_logits is not None and adv_labels is not None and len(
                adv_logits) == len(adv_labels) and len(adv_logits) > 0:
            adv_losses = []
            for logit_adv, label_adv in zip(adv_logits, adv_labels):
                if logit_adv is None or label_adv is None:
                    continue
                if logit_adv.size(0) == 0:
                    continue
                adv_losses.append(F.cross_entropy(logit_adv, label_adv))
            if len(adv_losses) > 0:
                adv_losses_tensor = torch.stack(adv_losses)
                loss_adv = adv_losses_tensor.mean()
            else:
                loss_adv = logit_emo.new_zeros(())
        else:
            loss_adv = logit_emo.new_zeros(())

        if loss_type == 'auto':
            awl = AutomaticWeightedLoss(3)
            task_loss = awl(loss_emo, loss_sen, loss_sft)
        elif loss_type == 'epoch':
            task_loss = (epoch / epochs) * (lambd[0] * loss_emo) + (
                1 - epoch / epochs) * (lambd[1] * loss_sen +
                                       lambd[2] * loss_sft)
        elif loss_type == 'emo_sen_sft':
            task_loss = lambd[0] * loss_emo + lambd[1] * loss_sen + lambd[
                2] * loss_sft
        elif loss_type == 'emo_sen':
            task_loss = lambd[0] * loss_emo + lambd[1] * loss_sen
        elif loss_type == 'emo_sft':
            task_loss = lambd[0] * loss_emo + lambd[2] * loss_sft
        elif loss_type == 'emo':
            task_loss = loss_emo
        elif loss_type == 'sen_sft':
            task_loss = lambd[1] * loss_sen + lambd[2] * loss_sft
        elif loss_type == 'sen':
            task_loss = loss_sen
        else:
            raise NotImplementedError

        cur_gamma, cur_lambda = aux_weight_scheduler(epoch,
                                                     contrastive_weight,
                                                     decouple_weight,
                                                     peak_epochs=5,
                                                     peak_scale=1.5)
        adv_weight_eff = float(modality_adv_weight) * alpha
        if epoch + 1 > 30:
            adv_weight_eff = adv_weight_eff * 1.1
        loss = task_loss + cur_gamma * loss_supcon + cur_lambda * loss_decouple + pair_aux_weight_04 * loss_pair_04 + pair_aux_weight_25 * loss_pair_25 + pair04_margin_weight * loss_pair04_margin + proto_aux_weight * loss_proto + hier_aux_weight * loss_hier_ce + hier_kl_weight * loss_hier_kl + temporal_aux_weight * loss_temp + adv_weight_eff * loss_adv

        preds_emo.append(torch.argmax(prob_emo, 1).detach())
        labels_emo.append(label_emo.detach())
        preds_sen.append(torch.argmax(prob_sen, 1).detach())
        labels_sen.append(label_sen.detach())
        preds_sft.append(torch.argmax(prob_sft, 1).detach())
        labels_sft.append(label_sft.detach())
        vids += data[-1]
        losses.append(loss.detach())
        adv_loss_values.append(loss_adv.detach())

        if train:
            loss.backward()
            optimizer.step()

        if collect_features:
            extracted_feats.append(extracted_feature.detach())

    if preds_emo != []:
        preds_emo = torch.cat(preds_emo, dim=0).cpu().numpy()
        labels_emo = torch.cat(labels_emo, dim=0).cpu().numpy()
        if classify == 'regression':
            labels_sen = np.array([])
            preds_sen = np.array([])
            labels_sft = np.array([])
            preds_sft = np.array([])
        else:
            preds_sen = torch.cat(preds_sen, dim=0).cpu().numpy()
            labels_sen = torch.cat(labels_sen, dim=0).cpu().numpy()
            preds_sft = torch.cat(preds_sft, dim=0).cpu().numpy()
            labels_sft = torch.cat(labels_sft, dim=0).cpu().numpy()
        if collect_features and extracted_feats != []:
            extracted_feats = torch.cat(extracted_feats, dim=0).cpu().numpy()
        else:
            extracted_feats = np.array([])
    else:
        labels_emo = np.array(labels_emo)
        preds_emo = np.array(preds_emo)
        labels_sen = np.array(labels_sen)
        preds_sen = np.array(preds_sen)
        labels_sft = np.array(labels_sft)
        preds_sft = np.array(preds_sft)
        extracted_feats = np.array([])

    vids = np.array(vids)
    avg_loss = round(torch.stack(losses).mean().item(), 4)
    avg_loss_adv = round(torch.stack(adv_loss_values).mean().item(), 4)
    if classify == 'regression':
        if dataset in ['MOSEIREG', 'CMUMOSEIREG', 'MOSIREG']:
            binary_truth = (labels_emo >= 0)
            binary_preds = (preds_emo >= 0)
            msa = float(accuracy_score(binary_truth, binary_preds))
            f1_bin = float(
                f1_score(binary_truth, binary_preds, average='weighted'))
            mae = float(np.mean(np.abs(preds_emo - labels_emo))
                        ) if labels_emo.size > 0 else 0.0
            return avg_loss, avg_loss_adv, labels_emo, preds_emo, round(
                msa, 4), round(f1_bin, 4), np.array([]), np.array([]), round(
                    mae, 4), 0.0, 0.0, 0.0, vids, initial_feats, extracted_feats
        if labels_emo.size > 1 and np.std(labels_emo) > 0 and np.std(preds_emo) > 0:
            corr = float(np.corrcoef(labels_emo, preds_emo)[0][1])
        else:
            corr = 0.0
        mae = float(np.mean(np.abs(preds_emo - labels_emo))
                    ) if labels_emo.size > 0 else 0.0
        try:
            r_square = float(r2_score(labels_emo, preds_emo))
        except Exception:
            r_square = 0.0
        return avg_loss, avg_loss_adv, labels_emo, preds_emo, round(
            corr, 4), round(r_square, 4), np.array([]), np.array([]), round(
                mae, 4), 0.0, 0.0, 0.0, vids, initial_feats, extracted_feats

    if classify == 'unsupervised':
        return avg_loss, avg_loss_adv, np.array([]), np.array([]), 0.0, 0.0, np.array(
            []), np.array([]), 0.0, 0.0, 0.0, 0.0, vids, initial_feats, extracted_feats

    avg_acc_emo = round(accuracy_score(labels_emo, preds_emo) * 100, 2)
    avg_f1_emo = round(
        f1_score(labels_emo, preds_emo, average='weighted') * 100, 2)
    avg_acc_sen = round(accuracy_score(labels_sen, preds_sen) * 100, 2)
    avg_f1_sen = round(
        f1_score(labels_sen, preds_sen, average='weighted') * 100, 2)
    avg_acc_sft = round(accuracy_score(labels_sft, preds_sft) * 100, 2)
    avg_f1_sft = round(
        f1_score(labels_sft, preds_sft, average='weighted') * 100, 2)

    return avg_loss, avg_loss_adv, labels_emo, preds_emo, avg_acc_emo, avg_f1_emo, labels_sen, preds_sen, avg_acc_sen, avg_f1_sen, avg_acc_sft, avg_f1_sft, vids, initial_feats, extracted_feats
