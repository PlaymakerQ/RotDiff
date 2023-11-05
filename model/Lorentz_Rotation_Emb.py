import torch
import torch.nn as nn
import torch.nn.functional as F


class LorentzRotationEmbedding(nn.Module):

    def __init__(self, opt, init_size=1e-3, num_neg_sample_size=20, gamma=1.0):
        super(LorentzRotationEmbedding, self).__init__()
        self.gamma = gamma
        self.data_type = torch.float
        self.init_size = init_size
        self.n_nodes = opt.user_size
        self.embedding_dim = opt.d_emb
        self.num_negs = num_neg_sample_size
        self.device = opt.device
        self.bias_fr = nn.Embedding(self.n_nodes, 1)
        self.bias_fr.weight.data = torch.zeros((self.n_nodes, 1), dtype=self.data_type)
        self.bias_to = nn.Embedding(self.n_nodes, 1)
        self.bias_to.weight.data = torch.zeros((self.n_nodes, 1), dtype=self.data_type)
        self.rel_diag = nn.Embedding(2, self.embedding_dim)
        self.rel_diag.weight.data = 2 * torch.rand((2, self.embedding_dim), dtype=self.data_type) - 1.0
        self.rel_diag.weight.data[:, ::2] = 1.0
        self.rel_diag.weight.data[:, 1::2] = 0.0

    def score(self, frs, tos, emb):
        context_vecs = self.get_fr_embedding(frs, emb)
        target_gold_vecs = self.get_to_embedding(tos, emb)
        dist_score = self.similarity_score(context_vecs, target_gold_vecs)
        bias_frs = self.bias_fr(frs)
        bias_tos = self.bias_to(tos).permute(0, 2, 1)
        score = dist_score + bias_frs + bias_tos

        return score

    def get_neg_samples(self, golds):
        neg_samples = torch.randint(self.n_nodes, (golds.shape[0], self.num_negs)).to(self.device)

        return neg_samples

    def get_fr_embedding(self, frs, emb):
        context_vecs = emb(frs)
        dim = context_vecs.size(2)
        context_frs = torch.zeros(frs.size(), dtype=torch.long).cuda()
        rel_diag_vecs = self.rel_diag(context_frs)
        r, x = rel_diag_vecs.view(-1, dim), context_vecs.view(-1, dim)
        context_rot = givens_rotations(r, x).view(context_vecs.size())

        return context_rot

    def get_to_embedding(self, tos, emb):
        context_vecs = emb(tos)
        dim = context_vecs.size(2)
        context_tos = torch.ones(tos.size(), dtype=torch.long).cuda()
        rel_diag_vecs = self.rel_diag(context_tos)
        r, x = rel_diag_vecs.view(-1, dim), context_vecs.view(-1, dim)
        context_rot = givens_rotations(r, x).view(context_vecs.size())

        return context_rot

    def similarity_score(self, context_vecs, target_vecs):
        c = self.gamma
        x = context_vecs
        y = target_vecs
        x2_srt = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + c)
        y2_srt = -torch.sqrt(torch.sum(y * y, dim=-1, keepdim=True) + c)
        u = torch.cat((x, x2_srt), -1)
        v = torch.cat((y, y2_srt), -1)
        vt = v.permute(0, 2, 1)
        uv = torch.bmm(u, vt)
        result = - 2 * c - 2 * uv
        score = result.neg()
        return score

    def forward(self, graph, emb):
        frs = graph[:, 0:1]
        tos = graph[:, 1:2]
        to_negs = self.get_neg_samples(tos)
        positive_score = self.score(frs, tos, emb)
        negative_score = self.score(frs, to_negs, emb)
        positive_loss = F.logsigmoid(positive_score).sum()
        negative_loss = F.logsigmoid(-negative_score).sum()
        batch_loss = - (positive_loss + negative_loss)

        return batch_loss

    def get_all_embs(self):
        r = self.rel_diag.weight
        return r


def givens_rotations(r, x):
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)

    return x_rot.view((r.shape[0], -1))
