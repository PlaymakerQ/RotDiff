import time
import torch.nn as nn
from utils.dataloader import *
from train_test_epochs import *
from model.model import RotDiff
from utils.metrics import print_scores


def main_process(train_data, valid_data, social_graph, diffusion_graph):
    model = RotDiff(opt)
    model = model.to(opt.device)
    loss_func = nn.CrossEntropyLoss(reduction='sum', ignore_index=constants.PAD)
    loss_func = loss_func.to(opt.device)
    optimizer = RiemannianAdam(model.parameters(), lr=opt.lr, stabilize=1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    validation_history = 0.0
    logging.info(dash)

    best_scores = 0.0
    pre_train(model, diffusion_graph, social_graph)

    for epoch_i in range(opt.epoch):

        start = time.time()
        train_loss, train_accu = train_cascade(model, train_data, loss_func, optimizer)
        elapse_time = (time.time() - start) / 60
        logging.info(f"[Epoch {epoch_i:>02}]: "
                     f"Training loss: {train_loss:<4.3f}, "
                     f"accuracy: {train_accu:>2.3%}, "
                     f"elapse: {elapse_time:>3.3f} min.")
        train_graph_emb(model, diffusion_graph, social_graph, optimizer)
        valid_loss, valid_accu, scores = sep_valid_epoch(model, loss_func, valid_data)
        scheduler.step(valid_loss)
        scores = sep_test_epoch(model, test_data)
        if validation_history <= sum(scores.values()):
            validation_history = sum(scores.values())
            best_scores = scores

    print_scores(best_scores)


if __name__ == "__main__":
    user_size, train, valid, test = Split_data(opt.data_name, opt.train_rate, opt.valid_rate, seed=opt.seed)
    train_data = DataLoader(train, batch_size=opt.batch_size)
    valid_data = DataLoader(valid, batch_size=opt.batch_size)
    test_data = DataLoader(test, batch_size=opt.batch_size)
    opt.user_size = user_size
    social_graph = build_social_graph(opt.data_name, opt.seed)
    diffusion_graph = build_diffusion_graph(train)
    main_process(train_data, valid_data, social_graph, diffusion_graph)
