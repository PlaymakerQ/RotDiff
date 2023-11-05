import os
import logging
import random
import pickle
import torch
import numpy as np
import networkx as nx
from utils import constants


class Options(object):

    def __init__(self, data_name='douban'):
        base_dir = os.path.join('data', data_name)
        self.data_name = data_name
        self.data = os.path.join(base_dir, 'cascades.txt')
        self.net_data = os.path.join(base_dir, 'edges.txt')
        self.u2idx_dict = os.path.join(base_dir, 'u2idx.pickle')


def Split_data(data_name, train_rate=0.8, valid_rate=0.1, seed=123, with_EOS=True):
    options = Options(data_name)
    u2idx: dict

    if os.path.exists(options.u2idx_dict):
        with open(options.u2idx_dict, 'rb') as file:
            u2idx = pickle.load(file)

    else:
        user_size, u2idx = build_user_idx(options)

    t_cascades = []
    timestamps = []
    for line in open(options.data):
        u_in_cas = set()
        if len(line.strip()) == 0:
            continue
        timestamplist = []
        userlist = []
        chunks = line.strip().split(',')
        for chunk in chunks:
            if len(chunk.split()) == 2:
                user, timestamp = chunk.split()
            elif len(chunk.split()) == 3:
                root, user, timestamp = chunk.split()
                if root in u2idx:
                    u_in_cas.add(root)
                    userlist.append(u2idx[root])
                    timestamplist.append(float(timestamp))
            if user in u2idx:
                if user not in u_in_cas:
                    u_in_cas.add(user)
                    userlist.append(u2idx[user])
                    timestamplist.append(float(timestamp))

        if 1 < len(userlist) <= 500:
            if with_EOS:
                userlist.append(constants.EOS)
                timestamplist.append(constants.EOS)
            t_cascades.append(userlist)
            timestamps.append(timestamplist)

    order = [i[0] for i in sorted(enumerate(timestamps), key=lambda x: x[1])]
    t_cascades[:] = [t_cascades[i] for i in order]
    train_idx_ = int(train_rate * len(t_cascades))
    train = t_cascades[0:train_idx_]
    valid_idx_ = int((train_rate + valid_rate) * len(t_cascades))
    valid = t_cascades[train_idx_:valid_idx_]
    test = t_cascades[valid_idx_:]
    random.seed(seed)
    random.shuffle(train)
    user_size = len(u2idx)
    total_len = sum(len(i) - 1 for i in t_cascades)
    logging.info("Data Information:")
    logging.info(" - Datasets: {}".format(options.data_name))
    logging.info(" - Total number of users in cascades: %d" % (user_size - 2))
    logging.info(" - Total size: {}, Train size: {}, Valid size: {}, Test size: {}."
                 .format(len(t_cascades), len(train), len(valid), len(test)))
    ave_l, max_l = total_len / len(t_cascades), max(len(cas) for cas in t_cascades)
    min_l = min(len(cas) for cas in t_cascades)
    logging.info(
        " - Average length: {:.2f}, Maximum length: {:.2f}, Minimum length: {:.2f}".format(ave_l, max_l, min_l))

    return user_size, train, valid, test


def build_user_idx(options):
    user_set = set()
    u2idx = {}
    
    with open(options.net_data) as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            i, j = line.strip('\n').split(',')
            user_set.add(i)
            user_set.add(j)

    pos = 0
    u2idx['<blank>'] = pos
    pos += 1
    u2idx['</s>'] = pos
    pos += 1

    for user in user_set:
        u2idx[user] = pos
        pos += 1

    user_size = len(user_set) + 2

    with open(options.u2idx_dict, 'wb') as handle:
        pickle.dump(u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return user_size, u2idx


class DataLoader(object):

    def __init__(self, casades, batch_size=64):
        self.cascades = casades
        self._batch_size = batch_size
        self._n_batch = int(np.ceil(len(self.cascades) / self._batch_size))
        self._iter_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):

        def pad_to_longest(insts):

            m_len = 200
            max_cas = max(len(inst) for inst in insts)
            if max_cas < m_len:
                max_len = max_cas
                inst_data = np.array([
                    inst + [constants.PAD] * (max_len - len(inst))
                    for inst in insts])
            else:
                max_len = m_len
                inst_data = np.array([
                    inst + [constants.PAD] * (max_len - len(inst)) if len(inst) < max_len else inst[:max_len]
                    for inst in insts])

            inst_data_tensor = torch.LongTensor(inst_data)

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1
            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size
            seq_insts = self.cascades[start_idx:end_idx]
            seq_data = pad_to_longest(seq_insts)
            return seq_data
        else:
            self._iter_count = 0
            raise StopIteration()


def build_diffusion_graph(cascades):
    edge = []
    for line in cascades:
        for i in range(len(line)):
            if line[i] == 1 or line[i] == 0:
                break
            if i < len(line) - 1 and line[i + 1] != 1:
                edge.append([line[i], line[i + 1]])

    tor_edge = torch.LongTensor(edge)

    return tor_edge


def build_social_graph(data_name, seed=123):
    opts = Options(data_name)
    G = nx.DiGraph()

    with open(opts.u2idx_dict, 'rb') as f:
        u2idx = pickle.load(f)

    with open(opts.net_data) as f:
        for edge in f:
            if len(edge.strip()) == 0:
                continue
            src, tar = edge.strip('\n').split(',')
            if src in u2idx and tar in u2idx:
                src, tar = u2idx[src], u2idx[tar]
                G.add_edge(src, tar)
    soci_G = list(G.edges)
    random.seed(seed)
    random.shuffle(soci_G)
    soci_g = torch.LongTensor(soci_G)

    return soci_g



