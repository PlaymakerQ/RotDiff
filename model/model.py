import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.constants as constants
from model.Lorentz_Rotation_Emb import LorentzRotationEmbedding, givens_rotations
from model.Lorentz_Attention import LorentzSelfAttention
from utils.others import get_previous_user_mask


class RotDiff(nn.Module):
    def __init__(self, opt):
        super(RotDiff, self).__init__()
        self.c = opt.gamma
        self.device = opt.device
        self.hidden_size = opt.d_emb
        self.n_node = opt.user_size
        self.num_negs = 20
        self.init_size = 1e-3
        self.drop = opt.drop
        self.dropout = nn.Dropout(self.drop)
        self.linear_diffusion = nn.Linear(self.hidden_size, self.n_node)
        self.linear_diffusion.weight.data.fill_(0.01)
        self.linear_social = nn.Linear(self.hidden_size, self.n_node)
        self.linear_diffusion.weight.data.fill_(0.01)
        self.att_rot_diffusion = nn.Embedding(3, self.hidden_size)
        self.att_rot_diffusion.weight.data = 2 * torch.rand((3, self.hidden_size), dtype=torch.float) - 1.0
        self.att_rot_diffusion.weight.data[:, ::2] = 1.0
        self.att_rot_diffusion.weight.data[:, 1::2] = 0.0
        self.att_rot_social = nn.Embedding(3, self.hidden_size)
        self.att_rot_social.weight.data = 2 * torch.rand((3, self.hidden_size), dtype=torch.float) - 1.0
        self.att_rot_social.weight.data[:, ::2] = 1.0
        self.att_rot_social.weight.data[:, 1::2] = 0.0
        self.pos_encode = torch.randn(1000, self.hidden_size, device=self.device)
        self._set_angles(step=1000)
        self.emb_all = nn.Embedding(self.n_node, self.hidden_size)
        self.emb_all.weight.data = self.init_size * torch.randn(
            (self.n_node, self.hidden_size))
        self.diffusion_emb = LorentzRotationEmbedding(opt, gamma=self.c)
        self.social_emb = LorentzRotationEmbedding(opt, gamma=self.c)
        self.lo_attention_diffusion = LorentzSelfAttention(
            dimension=self.hidden_size)
        self.lo_attention_social = LorentzSelfAttention(
            dimension=self.hidden_size)

    def train_model(self, input_cas):
        device = self.device
        input_cas = input_cas.to(device)
        seed = input_cas[:, :-1]
        tgt = input_cas[:, 1:]
        tgt_emb = self.emb_all(tgt).to(device)
        sample = tgt.contiguous().view(-1, 1)
        neg_tgt = self._get_neg_samples(sample)
        neg_tgt_emb = self.emb_all(neg_tgt)
        rot = self.diffusion_emb.get_all_embs()
        tgt_emb_rotd = self.do_rotation(tgt_emb, rot, 1)
        neg_tgt_emb_rotd = self.do_rotation(neg_tgt_emb, rot, 1)
        latent_dif = self.lorentz_self_att_diffusion(seed)
        latent_loss = self.cal_scores(latent_dif, tgt_emb_rotd, neg_tgt_emb_rotd)
        latent_soc = self.lorentz_self_att_social(seed)
        mask = get_previous_user_mask(seed, self.n_node)
        output = self.diffusion_prediction_both(latent_dif, latent_soc, mask)
        return output, latent_loss

    def forward(self, input_cas: torch.Tensor):
        device = self.device
        input_cas = input_cas.to(device)
        input_cas = input_cas[:, :-1]
        latent_dif = self.lorentz_self_att_diffusion(input_cas)
        latent_soc = self.lorentz_self_att_social(input_cas)
        mask = get_previous_user_mask(input_cas, self.n_node)
        output = self.diffusion_prediction_both(latent_dif, latent_soc, mask)
        return output

    def rotation_positional_encoding(self, user_emb):
        emb_shape = user_emb.size()
        B = user_emb.size(0)
        S = user_emb.size(1)
        seq_idx = torch.arange(S).expand(B, S).cuda()
        pos_emb = F.embedding(seq_idx, self.pos_encode)
        rot_pos_emb, user_emb = pos_emb.view(-1, self.hidden_size), user_emb.view(-1, self.hidden_size)
        pos_user_emb = givens_rotations(rot_pos_emb, user_emb).reshape(emb_shape)
        return pos_user_emb

    def lorentz_self_att_diffusion(self, input_seq: torch.Tensor):
        user_emb = self.emb_all(input_seq)
        rot_diff = self.diffusion_emb.get_all_embs()
        rot_diff = rot_diff
        user_emb = self.do_rotation(user_emb, rot_diff, tag=0)
        pos_user_emb = self.rotation_positional_encoding(user_emb)
        Q = self.do_rotation(x=pos_user_emb, r=self.att_rot_diffusion.weight, tag=0)
        K = self.do_rotation(x=pos_user_emb, r=self.att_rot_diffusion.weight, tag=1)
        V = self.do_rotation(x=pos_user_emb, r=self.att_rot_diffusion.weight, tag=2)
        mask = (input_seq == constants.PAD)
        latent_emb = self.lo_attention_diffusion(Q, K, V, mask=mask)
        latent_emb = self.dropout(latent_emb)

        return latent_emb

    def lorentz_self_att_social(self, input_seq: torch.Tensor):
        user_emb = self.emb_all(input_seq)
        rot_soc = self.social_emb.get_all_embs()
        rot_soc = rot_soc
        user_emb = self.do_rotation(user_emb, rot_soc, tag=0)
        pos_user_emb = self.rotation_positional_encoding(user_emb)
        Q = self.do_rotation(x=pos_user_emb, r=self.att_rot_social.weight, tag=0)
        K = self.do_rotation(x=pos_user_emb, r=self.att_rot_social.weight, tag=1)
        V = self.do_rotation(x=pos_user_emb, r=self.att_rot_social.weight, tag=2)
        mask = (input_seq == constants.PAD)
        latent_emb = self.lo_attention_social(Q, K, V, mask=mask)
        latent_emb = self.dropout(latent_emb)

        return latent_emb

    def diffusion_prediction_both(self, latent_dif, latent_soc, mask=None):
        emb_all = self.emb_all.weight
        diff_rot = self.diffusion_emb.get_all_embs()
        emb_all_rot_d = self.do_rotation(emb_all, diff_rot, tag=1)
        output_d1 = self.distance_for_prediction(latent_dif, emb_all_rot_d)
        out_l1 = self.linear_diffusion(latent_dif)
        output_d = output_d1 + out_l1
        soc_rot = self.social_emb.get_all_embs()
        emb_all_rot_s = self.do_rotation(emb_all, soc_rot, tag=1)
        output_s1 = self.distance_for_prediction(latent_soc, emb_all_rot_s)
        out_l2 = self.linear_social(latent_soc)
        output_s = output_s1 + out_l2
        output = 0.02 * output_s + output_d
        if mask is not None:
            output = (output + mask)
        output = output.view(-1, output.size(-1))

        return output

    def train_emb_d(self, graph):
        device = self.device
        emb_loss = self.diffusion_emb(graph.to(device), self.emb_all)

        return emb_loss

    def train_emb_s(self, graph):
        device = self.device
        emb_loss = self.social_emb(graph.to(device), self.emb_all)

        return emb_loss

    def cal_scores(self, x, pos, neg):
        pos_score = self.distance_for_contrastive(x, pos)
        neg_score = self.distance_for_contrastive(x, neg)
        pos_loss = F.logsigmoid(pos_score).sum()
        neg_loss = F.logsigmoid(-neg_score).sum()
        loss = - (pos_loss + neg_loss)

        return loss

    def distance_for_prediction(self, x, y):
        c = self.c
        x = x.unsqueeze(2)
        y = y.unsqueeze(1)
        x2_srt = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + c)
        y2_srt = -torch.sqrt(torch.sum(y * y, dim=-1, keepdim=True) + c)
        u = torch.cat((x, x2_srt), -1)
        v = torch.cat((y, y2_srt), -1)
        vt = v.squeeze(1).T
        uv = u @ vt
        result = - 2 * c - 2 * uv
        result = result.neg()

        return result.squeeze(2)

    def distance_for_contrastive(self, x, y):
        b = x.size(0)
        s = x.size(1)
        c = self.c
        x = x.view(b * s, -1, self.hidden_size)
        y = y.view(b * s, -1, self.hidden_size)
        x2_srt = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + c)
        y2_srt = -torch.sqrt(torch.sum(y * y, dim=-1, keepdim=True) + c)
        u = torch.cat((x, x2_srt), -1)
        v = torch.cat((y, y2_srt), -1)
        vt = v.permute(0, 2, 1)
        uv = torch.bmm(u, vt)
        result = - 2 * c - 2 * uv
        score = result.neg()
        return score

    def do_rotation(self, x, r, tag=0):
        ori_size = x.size()
        x = x.view(-1, self.hidden_size)
        rot_idx = torch.ones(x.size(0), dtype=torch.long).to(self.device)
        rot_idx = rot_idx * tag
        rot_m = F.embedding(rot_idx, r)
        rotated_x = givens_rotations(rot_m, x).reshape(ori_size)
        return rotated_x

    def _set_angles(self, step=20):
        num = self.pos_encode.size(0)
        repeat = int(num / step)
        base_angle = math.pi / step
        angles = torch.arange(step) * base_angle
        angles = angles.unsqueeze(0).expand(repeat, step).reshape(num, 1)
        msin = torch.sin(angles)
        mcos = torch.cos(angles)
        self.pos_encode[:, ::2] = mcos
        self.pos_encode[:, 1::2] = msin

    def _get_neg_samples(self, golds):
        neg_samples = torch.randint(self.n_node, (golds.shape[0], self.num_negs)).to(self.device)

        return neg_samples
