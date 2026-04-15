import torch
import torch.nn as nn
import torch.nn.functional as F
from module import HeterGConv_Edge, HeterGConvLayer, SenShift_Feat, ModalityDiscriminator
from utils import batch_to_all_tva


class CrossModalRefinement(nn.Module):

    def __init__(self, hidden_dim, heads=4, dropout=0.1, grl_alpha=1.0):
        super(CrossModalRefinement, self).__init__()
        self.attn_t_from_v = nn.MultiheadAttention(hidden_dim,
                                                   heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.attn_t_from_a = nn.MultiheadAttention(hidden_dim,
                                                   heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.attn_v_from_t = nn.MultiheadAttention(hidden_dim,
                                                   heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.attn_v_from_a = nn.MultiheadAttention(hidden_dim,
                                                   heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.attn_a_from_t = nn.MultiheadAttention(hidden_dim,
                                                   heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.attn_a_from_v = nn.MultiheadAttention(hidden_dim,
                                                   heads,
                                                   dropout=dropout,
                                                   batch_first=True)

        self.mix_t = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim),
                                   nn.ReLU(), nn.Dropout(dropout))
        self.mix_v = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim),
                                   nn.ReLU(), nn.Dropout(dropout))
        self.mix_a = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim),
                                   nn.ReLU(), nn.Dropout(dropout))

        self.shared_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU(), nn.Dropout(dropout),
                                         nn.Linear(hidden_dim, hidden_dim))
        self.private_t = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, hidden_dim))
        self.private_v = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, hidden_dim))
        self.private_a = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, hidden_dim))

        self.out_t = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                   nn.ReLU(), nn.Dropout(dropout))
        self.out_v = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                   nn.ReLU(), nn.Dropout(dropout))
        self.out_a = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                   nn.ReLU(), nn.Dropout(dropout))
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.modality_discriminator = ModalityDiscriminator(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            alpha=grl_alpha,
            dropout=dropout,
            output_dim=3,
        )

    def set_grl_alpha(self, alpha):
        self.modality_discriminator.set_alpha(alpha)

    def _attend(self, attn_layer, query, context, key_padding_mask):
        out, _ = attn_layer(query,
                            context,
                            context,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)
        return out

    def _flatten_valid(self, x, valid_mask):
        flat_mask = valid_mask.reshape(-1)
        flat_x = x.reshape(-1, x.size(-1))
        return flat_x[flat_mask]

    def _decouple_loss(self, shared_t, shared_v, shared_a, private_t, private_v,
                       private_a, valid_mask):
        st = self._flatten_valid(shared_t, valid_mask)
        sv = self._flatten_valid(shared_v, valid_mask)
        sa = self._flatten_valid(shared_a, valid_mask)
        pt = self._flatten_valid(private_t, valid_mask)
        pv = self._flatten_valid(private_v, valid_mask)
        pa = self._flatten_valid(private_a, valid_mask)

        if st.size(0) == 0:
            return shared_t.new_zeros(())

        ortho = ((F.normalize(pt, dim=-1) * F.normalize(st, dim=-1)
                  ).sum(dim=-1).abs().mean() + (
                      F.normalize(pv, dim=-1) *
                      F.normalize(sv, dim=-1)).sum(dim=-1).abs().mean() + (
                          F.normalize(pa, dim=-1) *
                          F.normalize(sa, dim=-1)).sum(dim=-1).abs().mean()) / 3
        common_align = (F.mse_loss(st, sv) + F.mse_loss(st, sa) +
                        F.mse_loss(sv, sa)) / 3
        private_sep = (F.cosine_similarity(F.normalize(pt, dim=-1),
                                           F.normalize(pv, dim=-1),
                                           dim=-1).pow(2).mean() +
                       F.cosine_similarity(F.normalize(pt, dim=-1),
                                           F.normalize(pa, dim=-1),
                                           dim=-1).pow(2).mean() +
                       F.cosine_similarity(F.normalize(pv, dim=-1),
                                           F.normalize(pa, dim=-1),
                                           dim=-1).pow(2).mean()) / 3
        return ortho + 0.5 * common_align + 0.5 * private_sep

    def forward(self, feat_t, feat_v, feat_a, umask, return_details=False):
        feat_t = feat_t.transpose(0, 1)
        feat_v = feat_v.transpose(0, 1)
        feat_a = feat_a.transpose(0, 1)
        key_padding_mask = ~(umask.transpose(0, 1) > 0)
        valid_mask = ~key_padding_mask
        valid_mask_f = valid_mask.unsqueeze(-1).float()

        t_from_v = self._attend(self.attn_t_from_v, feat_t, feat_v,
                                key_padding_mask)
        t_from_a = self._attend(self.attn_t_from_a, feat_t, feat_a,
                                key_padding_mask)
        v_from_t = self._attend(self.attn_v_from_t, feat_v, feat_t,
                                key_padding_mask)
        v_from_a = self._attend(self.attn_v_from_a, feat_v, feat_a,
                                key_padding_mask)
        a_from_t = self._attend(self.attn_a_from_t, feat_a, feat_t,
                                key_padding_mask)
        a_from_v = self._attend(self.attn_a_from_v, feat_a, feat_v,
                                key_padding_mask)

        mix_t = self.mix_t(torch.cat([feat_t, t_from_v, t_from_a], dim=-1))
        mix_v = self.mix_v(torch.cat([feat_v, v_from_t, v_from_a], dim=-1))
        mix_a = self.mix_a(torch.cat([feat_a, a_from_t, a_from_v], dim=-1))

        shared_t = self.shared_proj(mix_t)
        shared_v = self.shared_proj(mix_v)
        shared_a = self.shared_proj(mix_a)

        private_t = self.private_t(mix_t)
        private_v = self.private_v(mix_v)
        private_a = self.private_a(mix_a)

        shared_global = (shared_t + shared_v + shared_a) / 3

        refined_t = self.norm_t(feat_t + self.out_t(torch.cat(
            [private_t, shared_global], dim=-1))) * valid_mask_f
        refined_v = self.norm_v(feat_v + self.out_v(torch.cat(
            [private_v, shared_global], dim=-1))) * valid_mask_f
        refined_a = self.norm_a(feat_a + self.out_a(torch.cat(
            [private_a, shared_global], dim=-1))) * valid_mask_f

        decouple_loss = self._decouple_loss(shared_t, shared_v, shared_a,
                                            private_t, private_v, private_a,
                                            valid_mask)
        shared_flat_t = self._flatten_valid(shared_t, valid_mask)
        shared_flat_v = self._flatten_valid(shared_v, valid_mask)
        shared_flat_a = self._flatten_valid(shared_a, valid_mask)

        if self.modality_discriminator.alpha > 0.0:
            adv_logit_t = self.modality_discriminator(shared_flat_t)
            adv_logit_v = self.modality_discriminator(shared_flat_v)
            adv_logit_a = self.modality_discriminator(shared_flat_a)
            adv_label_t = torch.zeros(shared_flat_t.size(0),
                                      dtype=torch.long,
                                      device=shared_flat_t.device)
            adv_label_v = torch.ones(shared_flat_v.size(0),
                                     dtype=torch.long,
                                     device=shared_flat_v.device)
            adv_label_a = torch.full((shared_flat_a.size(0), ),
                                     2,
                                     dtype=torch.long,
                                     device=shared_flat_a.device)
            adv_outputs = {
                'adv_logits': [adv_logit_t, adv_logit_v, adv_logit_a],
                'adv_labels': [adv_label_t, adv_label_v, adv_label_a],
            }
        else:
            adv_outputs = {'adv_logits': [], 'adv_labels': []}
        if return_details:
            adv_outputs.update({
                'mix_flat': {
                    't': self._flatten_valid(mix_t, valid_mask).detach(),
                    'v': self._flatten_valid(mix_v, valid_mask).detach(),
                    'a': self._flatten_valid(mix_a, valid_mask).detach(),
                },
                'shared_flat': {
                    't': shared_flat_t.detach(),
                    'v': shared_flat_v.detach(),
                    'a': shared_flat_a.detach(),
                },
                'private_flat': {
                    't': self._flatten_valid(private_t, valid_mask).detach(),
                    'v': self._flatten_valid(private_v, valid_mask).detach(),
                    'a': self._flatten_valid(private_a, valid_mask).detach(),
                },
            })
        refined_t = refined_t.transpose(0, 1)
        refined_v = refined_v.transpose(0, 1)
        refined_a = refined_a.transpose(0, 1)
        return refined_t, refined_v, refined_a, decouple_loss, adv_outputs


class ModalityGating(nn.Module):

    def __init__(self, hidden_dim, drop=0.1):
        super(ModalityGating, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden_dim, 6))

    def forward(self, feats):
        fusion_in = torch.cat(feats, dim=-1)
        weights = torch.softmax(self.gate(fusion_in), dim=-1)
        out = 0
        for i, feat in enumerate(feats):
            out = out + weights[:, i:i + 1] * feat
        return out, weights


class FeatureGatedExpert(nn.Module):

    def __init__(self, hidden_dim, drop=0.1):
        super(FeatureGatedExpert, self).__init__()
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.audio_proj = nn.Linear(hidden_dim, hidden_dim)
        self.filter_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.corr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, feat_main, feat_text, feat_audio):
        feat_res = self.filter_mlp(feat_main)
        hint = 0.5 * (self.text_proj(feat_text) + self.audio_proj(feat_audio))
        gate = self.feature_gate(
            torch.cat([feat_main, feat_text, feat_audio], dim=-1))
        expert_feat = feat_main + feat_res + gate * hint
        expert_feat = F.normalize(expert_feat, p=2, dim=-1)
        corr_coef = self.corr_head(expert_feat).squeeze(-1)
        logits = torch.stack([-corr_coef, corr_coef], dim=-1)
        return logits, expert_feat, gate, corr_coef


class GraphSmile(nn.Module):

    def __init__(self, args, embedding_dims, n_classes_emo):
        super(GraphSmile, self).__init__()
        self.n_classes_emo = int(n_classes_emo)
        self.textf_mode = args.textf_mode
        self.no_cuda = args.no_cuda
        self.win_p = args.win[0]
        self.win_f = args.win[1]
        self.modals = args.modals
        self.shift_win = args.shift_win
        self.warmup_epochs = int(getattr(args, 'warmup_epochs', 8))
        self.refine_heads = int(getattr(args, 'refine_heads', 4))
        self.contrastive_dim = 128
        self.grl_alpha = float(getattr(args, 'grl_alpha', 1.0))

        if self.textf_mode == 'concat4' or self.textf_mode == 'sum4':
            self.used_t_indices = [0, 1, 2, 3]
        elif self.textf_mode == 'concat2' or self.textf_mode == 'sum2':
            self.used_t_indices = [0, 1]
        elif self.textf_mode == 'textf0':
            self.used_t_indices = [0]
        elif self.textf_mode == 'textf1':
            self.used_t_indices = [1]
        elif self.textf_mode == 'textf2':
            self.used_t_indices = [2]
        elif self.textf_mode == 'textf3':
            self.used_t_indices = [3]
        else:
            raise ValueError(f"unsupported: {self.textf_mode}")
        self.batchnorms_t = nn.ModuleList([
            nn.BatchNorm1d(embedding_dims[0]) for _ in self.used_t_indices])

        if self.textf_mode.startswith('concat'):
            in_dims_t = len(self.used_t_indices) * embedding_dims[0]
        else:
            in_dims_t = embedding_dims[0]
        self.dim_layer_t = nn.Sequential(nn.Linear(in_dims_t, args.hidden_dim),
                                         nn.LeakyReLU(), nn.Dropout(args.drop))
        self.dim_layer_v = nn.Sequential(
            nn.Linear(embedding_dims[1], args.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.drop),
        )
        self.dim_layer_a = nn.Sequential(
            nn.Linear(embedding_dims[2], args.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.drop),
        )
        self.cross_refine = CrossModalRefinement(args.hidden_dim,
                                                 self.refine_heads, args.drop,
                                                 self.grl_alpha)

        hetergconvLayer_tv = HeterGConvLayer(args.hidden_dim, args.drop,
                                             args.no_cuda)
        self.hetergconv_tv = HeterGConv_Edge(
            args.hidden_dim,
            hetergconvLayer_tv,
            args.heter_n_layers[0],
            args.drop,
            args.no_cuda,
            getattr(args, 'dropedge', 0.0),
            getattr(args, 'gcn_residual_drop', args.drop),
        )
        hetergconvLayer_ta = HeterGConvLayer(args.hidden_dim, args.drop,
                                             args.no_cuda)
        self.hetergconv_ta = HeterGConv_Edge(
            args.hidden_dim,
            hetergconvLayer_ta,
            args.heter_n_layers[1],
            args.drop,
            args.no_cuda,
            getattr(args, 'dropedge', 0.0),
            getattr(args, 'gcn_residual_drop', args.drop),
        )
        hetergconvLayer_va = HeterGConvLayer(args.hidden_dim, args.drop,
                                             args.no_cuda)
        self.hetergconv_va = HeterGConv_Edge(
            args.hidden_dim,
            hetergconvLayer_va,
            args.heter_n_layers[2],
            args.drop,
            args.no_cuda,
            getattr(args, 'dropedge', 0.0),
            getattr(args, 'gcn_residual_drop', args.drop),
        )

        self.modal_fusion = nn.Sequential(nn.Linear(args.hidden_dim,
                                                    args.hidden_dim),
                                         nn.LeakyReLU())
        self.modality_gating = ModalityGating(args.hidden_dim, args.drop)

        self.emo_output = nn.Linear(args.hidden_dim, n_classes_emo)
        self.sen_output = nn.Linear(args.hidden_dim, 3)
        self.senshift = SenShift_Feat(args.hidden_dim, args.drop,
                                      args.shift_win)
        self.use_pair_expert = bool(getattr(args, 'use_pair_expert', True))
        self.enable_pair_04 = self.n_classes_emo > 4
        self.enable_pair_25 = self.n_classes_emo > 5
        self.pair_logit_scale = float(getattr(args, 'pair_logit_scale', 0.3))
        self.use_proto_prior = bool(getattr(args, 'use_proto_prior', False))
        self.proto_alpha = float(getattr(args, 'proto_alpha', 0.25))
        self.proto_temp = float(getattr(args, 'proto_temp', 0.07))
        self.proto_momentum = float(getattr(args, 'proto_momentum', 0.95))
        self.use_hierarchical = bool(getattr(args, 'use_hierarchical', False))
        self.hier_alpha = float(getattr(args, 'hier_alpha', 0.20))
        self.hier_temp = float(getattr(args, 'hier_temp', 1.0))
        self.use_temporal_fusion = bool(
            getattr(args, 'use_temporal_fusion', False))
        self.temporal_logit_scale = float(
            getattr(args, 'temporal_logit_scale', 0.35))
        self.pair_head_04 = FeatureGatedExpert(args.hidden_dim, args.drop)
        self.pair_head_25 = FeatureGatedExpert(args.hidden_dim, args.drop)
        self.pair_gate = nn.Sequential(
            nn.Linear(args.hidden_dim, max(args.hidden_dim // 2, 32)),
            nn.ReLU(),
            nn.Dropout(args.drop),
            nn.Linear(max(args.hidden_dim // 2, 32), 2),
        )
        self.hier_adapter = nn.Sequential(
            nn.Linear(3, max(args.hidden_dim // 4, 16)),
            nn.ReLU(),
            nn.Linear(max(args.hidden_dim // 4, 16), n_classes_emo),
        )
        self.temporal_rnn = nn.GRU(args.hidden_dim,
                                   args.hidden_dim // 2,
                                   num_layers=1,
                                   batch_first=True,
                                   bidirectional=True)
        self.temporal_head = nn.Linear(args.hidden_dim, n_classes_emo)
        self.temporal_gate = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, max(args.hidden_dim // 2, 32)),
            nn.ReLU(),
            nn.Dropout(args.drop),
            nn.Linear(max(args.hidden_dim // 2, 32), 2),
        )
        self.contrast_proj = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU(),
            nn.Linear(args.hidden_dim, self.contrastive_dim))
        self.register_buffer('emo_prototypes',
                             torch.zeros(n_classes_emo, args.hidden_dim))
        self.register_buffer('emo_proto_counts', torch.zeros(n_classes_emo))
        self.train_stage = 'joint'

    def set_stage(self, stage='joint'):
        self.train_stage = stage
        front_modules = [
            self.dim_layer_t, self.dim_layer_v, self.dim_layer_a,
            self.cross_refine, self.contrast_proj
        ]
        graph_modules = [
            self.hetergconv_tv, self.hetergconv_ta, self.hetergconv_va,
            self.modal_fusion, self.modality_gating, self.emo_output,
            self.sen_output, self.senshift, self.pair_head_04,
            self.pair_head_25, self.pair_gate, self.hier_adapter,
            self.temporal_rnn, self.temporal_head, self.temporal_gate
        ]
        if stage == 'warmup':
            for module in front_modules:
                for p in module.parameters():
                    p.requires_grad = True
            for module in graph_modules:
                for p in module.parameters():
                    p.requires_grad = False
        else:
            for p in self.parameters():
                p.requires_grad = True

    def set_grl_alpha(self, alpha):
        self.grl_alpha = float(alpha)
        self.cross_refine.set_grl_alpha(self.grl_alpha)

    @torch.no_grad()
    def update_emotion_prototypes(self, feat_fusion, labels):
        if feat_fusion.numel() == 0:
            return
        feat_norm = F.normalize(feat_fusion.detach(), p=2, dim=-1)
        labels = labels.detach()
        for cls in labels.unique():
            cls_idx = int(cls.item())
            mask = labels == cls_idx
            if not mask.any():
                continue
            cls_feat = feat_norm[mask].mean(dim=0)
            cls_feat = F.normalize(cls_feat, p=2, dim=-1)
            old_count = self.emo_proto_counts[cls_idx].item()
            if old_count <= 0:
                self.emo_prototypes[cls_idx] = cls_feat
            else:
                self.emo_prototypes[cls_idx] = F.normalize(
                    self.proto_momentum * self.emo_prototypes[cls_idx] +
                    (1.0 - self.proto_momentum) * cls_feat,
                    p=2,
                    dim=-1)
            self.emo_proto_counts[cls_idx] = self.emo_proto_counts[
                cls_idx] + mask.sum().float()

    def get_proto_logits(self, feat_fusion):
        feat_norm = F.normalize(feat_fusion, p=2, dim=-1)
        proto_norm = F.normalize(self.emo_prototypes, p=2, dim=-1)
        valid_mask = (self.emo_proto_counts > 0).float().unsqueeze(0)
        proto_logits = torch.matmul(feat_norm, proto_norm.transpose(
            0, 1)) / max(self.proto_temp, 1e-6)
        proto_logits = proto_logits * valid_mask
        return proto_logits

    def get_coarse_from_emo(self, emo_logits):
        num_classes = emo_logits.size(1)
        if num_classes >= 6:
            coarse_neg = torch.logsumexp(emo_logits[:, [0, 3, 5]], dim=-1)
            coarse_neu = emo_logits[:, 2]
            coarse_pos = torch.logsumexp(emo_logits[:, [1, 4]], dim=-1)
        elif num_classes == 4:
            coarse_neg = torch.logsumexp(emo_logits[:, [0, 3]], dim=-1)
            coarse_neu = emo_logits[:, 2]
            coarse_pos = emo_logits[:, 1]
        else:
            coarse_neg = emo_logits[:, 0]
            coarse_neu = emo_logits[:, min(1, num_classes - 1)]
            coarse_pos = emo_logits[:, num_classes - 1]
        coarse = torch.stack([coarse_neg, coarse_neu, coarse_pos], dim=-1)
        return coarse

    def flatten_valid_seq(self, seq_feat, dia_lengths):
        valid_feats = []
        for b, cur_len in enumerate(dia_lengths):
            valid_feats.append(seq_feat[:cur_len, b, :])
        return torch.cat(valid_feats, dim=0)

    def temporal_forward(self, seq_feat, dia_lengths):
        seq_bt = seq_feat.transpose(0, 1)
        packed = nn.utils.rnn.pack_padded_sequence(seq_bt,
                                                   dia_lengths,
                                                   batch_first=True,
                                                   enforce_sorted=False)
        packed_out, _ = self.temporal_rnn(packed)
        out_bt, _ = nn.utils.rnn.pad_packed_sequence(packed_out,
                                                     batch_first=True)
        out_tb = out_bt.transpose(0, 1)
        feat_temp = self.flatten_valid_seq(out_tb, dia_lengths)
        logit_temp = self.temporal_head(feat_temp)
        return feat_temp, logit_temp

    def forward(self, feature_t0, feature_t1, feature_t2, feature_t3,
                feature_v, feature_a, umask, qmask, dia_lengths,
                return_analysis=False):

        all_t_features = [feature_t0, feature_t1, feature_t2, feature_t3]
        seq_len_t, batch_size_t, feature_dim_t = feature_t0.shape
        used_t_features = []
        for idx, bn in zip(self.used_t_indices, self.batchnorms_t):
            feat = all_t_features[idx]
            feat_bn = bn(feat.transpose(0, 1).reshape(-1, feature_dim_t))
            feat_bn = feat_bn.reshape(batch_size_t, seq_len_t, feature_dim_t).transpose(1, 0)
            used_t_features.append(feat_bn)

        if self.textf_mode in ['concat4', 'concat2']:
            merged_t_feat = torch.cat(used_t_features, dim=-1)
        elif self.textf_mode in ['sum4', 'sum2']:
            merged_t_feat = sum(used_t_features) / len(used_t_features)
        else:
            merged_t_feat = used_t_features[0]
        featdim_t = self.dim_layer_t(merged_t_feat)
        featdim_v, featdim_a = self.dim_layer_v(feature_v), self.dim_layer_a(
            feature_a)

        assert featdim_t.shape[0] == umask.shape[0]
        assert featdim_t.shape[1] == umask.shape[1]
        featdim_t, featdim_v, featdim_a, loss_decouple, adv_outputs = self.cross_refine(
            featdim_t, featdim_v, featdim_a, umask, return_details=return_analysis)
        adv_logits = list(adv_outputs['adv_logits'])
        adv_labels = list(adv_outputs['adv_labels'])
        featdim_m = (featdim_t + featdim_v + featdim_a) / 3.0
        feat_temp, logit_temp = self.temporal_forward(featdim_m, dia_lengths)
        emo_t, emo_v, emo_a = batch_to_all_tva(featdim_t, featdim_v, featdim_a,
                                               dia_lengths, self.no_cuda)
        assert emo_t.size(0) == sum(dia_lengths)
        assert emo_t.size(0) == emo_v.size(0) == emo_a.size(0)
        assert feat_temp.size(0) == emo_t.size(0)

        featheter_tv, heter_edge_index = self.hetergconv_tv(
            (emo_t, emo_v), dia_lengths, self.win_p, self.win_f, None, qmask)
        featheter_ta, heter_edge_index = self.hetergconv_ta(
            (emo_t, emo_a), dia_lengths, self.win_p, self.win_f,
            heter_edge_index, qmask)
        featheter_va, heter_edge_index = self.hetergconv_va(
            (emo_v, emo_a), dia_lengths, self.win_p, self.win_f,
            heter_edge_index, qmask)

        branch_feats = [
            self.modal_fusion(featheter_tv[0]),
            self.modal_fusion(featheter_ta[0]),
            self.modal_fusion(featheter_tv[1]),
            self.modal_fusion(featheter_va[0]),
            self.modal_fusion(featheter_ta[1]),
            self.modal_fusion(featheter_va[1]),
        ]
        feat_fusion, gate_weights = self.modality_gating(branch_feats)

        logit_emo_base = self.emo_output(feat_fusion)
        pair_logit_04, pair_feat_04, pair_feat_gate_04, pair_coef_04 = self.pair_head_04(
            feat_fusion, emo_t, emo_a)
        pair_logit_25, pair_feat_25, pair_feat_gate_25, pair_coef_25 = self.pair_head_25(
            feat_fusion, emo_t, emo_a)
        pair_gate = torch.softmax(self.pair_gate(feat_fusion), dim=-1)
        if self.use_pair_expert:
            delta_emo = torch.zeros_like(logit_emo_base)
            g04 = pair_gate[:, 0]
            g25 = pair_gate[:, 1]
            corr04 = self.pair_logit_scale * g04 * pair_coef_04
            corr25 = self.pair_logit_scale * g25 * pair_coef_25
            if self.enable_pair_04:
                delta_emo[:, 4] = delta_emo[:, 4] + corr04
                delta_emo[:, 0] = delta_emo[:, 0] - corr04
            if self.enable_pair_25:
                delta_emo[:, 5] = delta_emo[:, 5] + corr25
                delta_emo[:, 2] = delta_emo[:, 2] - corr25
            logit_emo = logit_emo_base + delta_emo
        else:
            logit_emo = logit_emo_base
        proto_logits = self.get_proto_logits(feat_fusion)
        if self.use_proto_prior:
            logit_emo = logit_emo + self.proto_alpha * proto_logits
        logit_sen = self.sen_output(feat_fusion)
        coarse_from_emo = self.get_coarse_from_emo(logit_emo_base)
        hier_probs = torch.softmax(logit_sen / max(self.hier_temp, 1e-6), dim=-1)
        hier_delta = self.hier_adapter(hier_probs)
        if self.use_hierarchical:
            logit_emo = logit_emo + self.hier_alpha * hier_delta
        temporal_gate = torch.softmax(
            self.temporal_gate(torch.cat([feat_fusion, feat_temp], dim=-1)),
            dim=-1)
        if self.use_temporal_fusion:
            logit_emo = logit_emo + self.temporal_logit_scale * (
                temporal_gate[:, 0:1] * logit_emo + temporal_gate[:, 1:2] *
                logit_temp - logit_emo)

        logit_shift = self.senshift(feat_fusion, feat_fusion, dia_lengths)
        feat_contrast = F.normalize(self.contrast_proj(feat_fusion),
                                    p=2,
                                    dim=-1)

        aux_outputs = {
            'logit_emo_base': logit_emo_base,
            'pair_logit_04': pair_logit_04,
            'pair_logit_25': pair_logit_25,
            'pair_feat_04': pair_feat_04,
            'pair_feat_25': pair_feat_25,
            'pair_feat_gate_04': pair_feat_gate_04,
            'pair_feat_gate_25': pair_feat_gate_25,
            'pair_coef_04': pair_coef_04,
            'pair_coef_25': pair_coef_25,
            'pair_gate': pair_gate,
            'proto_logits': proto_logits,
            'coarse_from_emo': coarse_from_emo,
            'hier_delta': hier_delta,
            'temp_logit_emo': logit_temp,
            'temporal_gate': temporal_gate,
            'adv_logits': adv_logits,
            'adv_labels': adv_labels,
        }
        if return_analysis:
            aux_outputs.update({
                'dcr_mix_flat': adv_outputs.get('mix_flat', {}),
                'dcr_shared_flat': adv_outputs.get('shared_flat', {}),
                'dcr_private_flat': adv_outputs.get('private_flat', {}),
                'modality_gate_weights': gate_weights.detach(),
                'branch_feats': [feat.detach() for feat in branch_feats],
                'heter_edge_index': heter_edge_index.detach(),
                'heter_edge_weight_tv': self.hetergconv_tv.edge_weight[:heter_edge_index.size(1)].detach(),
                'heter_edge_weight_ta': self.hetergconv_ta.edge_weight[:heter_edge_index.size(1)].detach(),
                'heter_edge_weight_va': self.hetergconv_va.edge_weight[:heter_edge_index.size(1)].detach(),
                'valid_utt_count': int(emo_t.size(0)),
                'feat_temp': feat_temp.detach(),
            })
        return logit_emo, logit_sen, logit_shift, feat_fusion, feat_contrast, loss_decouple, aux_outputs
