import os
import datetime
import torch
import numpy as np


def init_dir(opt):
    dataset = opt.data_name
    dt = datetime.datetime.now()
    date, time = dt.strftime("%m_%d"), dt.strftime('%H_%M_%S_')
    epoch = '_epoch-' + str(opt.epoch)
    lr = '_lr-' + str(opt.lr)
    batch_size = '_batch-' + str(opt.batch_size)
    dim = '_d_emb-' + str(opt.d_emb)
    f_name = time + opt.model + epoch + lr + batch_size + dim
    save_path = os.path.join('save', date, dataset, f_name)

    return save_path


def set_dir(save_path, file_name='', new=False):
    if new is True:
        os.makedirs(save_path)

    path = os.path.join(save_path, file_name)

    return path


def save_model(state, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, 'model.pt')
    torch.save(state, save_dir)


def get_previous_user_mask(seq, user_size):
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    return masked_seq
