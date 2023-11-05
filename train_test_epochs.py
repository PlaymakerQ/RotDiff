import torch
import utils.constants as constants
from optim.radam import *
from utils.metrics import compute_metric
from init import *


def get_performance(crit, pred, gold):
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(constants.PAD).data).sum().float()

    return loss, n_correct


def sep_test_epoch(model, validation_data, k_list=[10, 50, 100]):
    model.eval()
    scores = {}
    n_total_words = 0
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    with torch.no_grad():
        for i, batch in enumerate(validation_data):
            tgt = batch
            y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()
            pred = model(tgt)
            y_pred = pred.detach().cpu().numpy()
            scores_batch, scores_len = compute_metric(y_pred, y_gold, k_list)
            n_total_words += scores_len
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    return scores


def sep_valid_epoch(model, loss_func, valid_data, k_list=[10, 50, 100]):
    r""" Model validation. """
    model.eval()
    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    scores = {}
    num_total = 0.0
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    with torch.no_grad():
        for i, batch in enumerate(valid_data):
            tgt = batch
            gold = tgt[:, 1:]
            n_words = gold.data.ne(constants.PAD).sum().float()
            n_total_words += n_words
            pred = model(tgt)
            loss, n_correct = get_performance(loss_func, pred, gold.to(pred.device))
            n_total_correct += n_correct
            total_loss += loss.item()
            y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()
            y_pred = pred.detach().cpu().numpy()
            scores_batch, scores_len = compute_metric(y_pred, y_gold, k_list)
            num_total += scores_len
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / num_total
        scores['map@' + str(k)] = scores['map@' + str(k)] / num_total

    return total_loss / n_total_words, n_total_correct / n_total_words, scores

def train_graph_emb(model, d_graph, s_graph, optimizer):
    model.train()
    num_dg = d_graph.size(0)
    num_sg = s_graph.size(0)
    dif_batch_size = int(num_dg * 0.4)
    soc_batch_size = int(num_sg * 0.4)
    dif_batch_start = 0
    soc_batch_start = 0
    dif_all_edges = d_graph[torch.randperm(num_dg)]
    soc_all_edges = s_graph[torch.randperm(num_sg)]
    social_w = opt.w
    total_loss = 0.0

    while dif_batch_start < num_dg:
        batch_edge = dif_all_edges[dif_batch_start: dif_batch_start + dif_batch_size]
        optimizer.zero_grad()
        loss = model.train_emb_d(batch_edge)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        dif_batch_start += dif_batch_size

    while soc_batch_start < num_sg:
        batch_edge = soc_all_edges[soc_batch_start: soc_batch_start + soc_batch_size]
        optimizer.zero_grad()
        loss = model.train_emb_s(batch_edge) * social_w
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        soc_batch_start += soc_batch_size

    return total_loss

def train_cascade(model, training_data, loss_func, optimizer):
    model.train()
    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0

    for i, batch in enumerate(training_data):
        tgt = batch
        gold = tgt[:, 1:]
        n_words = gold.data.ne(constants.PAD).sum().float()
        n_total_words += n_words
        optimizer.zero_grad()
        pred, l_loss = model.train_model(tgt)
        pred_loss, n_correct = get_performance(loss_func, pred, gold.to(pred.device))
        if l_loss is not None:
            pred_loss = 0.5 * l_loss + pred_loss
        pred_loss.backward()
        optimizer.step()
        n_total_correct += n_correct
        total_loss += pred_loss.item()

    return total_loss / n_total_words, n_total_correct / n_total_words


def pre_train(model, d_graph, s_graph):
    pre_epochs = 5
    optimizer = RiemannianAdam(model.parameters(), lr=0.01, stabilize=1)
    model.train()

    for i in range(pre_epochs):
       
        num_dg = d_graph.size(0)
        num_sg = s_graph.size(0)
        dif_all_edges = d_graph[torch.randperm(num_dg)]
        soc_all_edges = s_graph[torch.randperm(num_sg)]
        social_w = opt.w
        total_loss = 0.0
        optimizer.zero_grad()
        loss = model.train_emb_d(dif_all_edges)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        optimizer.zero_grad()
        loss = model.train_emb_s(soc_all_edges) * social_w
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        logging.info("Pre-train loss of epoch {} is : {:.4f}".format(i, total_loss))
