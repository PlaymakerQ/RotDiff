import os
import torch
import logging
import argparse
from utils.others import init_dir, set_dir
import numpy as np

parser = argparse.ArgumentParser(description="RotDiff")
parser.add_argument('--model', default='RotDiff')
parser.add_argument('--data_name', default='christianity')
parser.add_argument('--seed', type=int, default=123,
                    help="set random seed.")
parser.add_argument('--epoch', type=int, default=20,
                    help="number of epochs.")
parser.add_argument('--batch_size', type=int, default=64,
                    help="batch size.")
parser.add_argument('--train_rate', type=float, default=0.8)
parser.add_argument('--valid_rate', type=float, default=0.1)
parser.add_argument('--drop', type=float, default=0.3)
parser.add_argument('--lr', type=float, default=1.6e-2,
                    help="initial learning rate.")
parser.add_argument('--w', type=float, default=0.3)
parser.add_argument('--alpha', type=float, default=0.3)
parser.add_argument('--gamma', type=float, default=1.0,
                    help="curvature parameter of the Lorentz model")
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--d_emb', type=int, default=64,
                    help="dimension of embeddings")
parser.add_argument('--log', action='store_true',
                    help='save log')
parser.add_argument('--save', action='store_true',
                    help='save model')
opt = parser.parse_args()
init_dir = init_dir(opt)
if opt.save:
    model_dir = set_dir(init_dir, 'model.pt', True)

formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

console = logging.StreamHandler()
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)
logging.getLogger("").setLevel(logging.INFO)

if opt.log:
    log_dir = set_dir(init_dir, 'log', True)
    filename = os.path.join(log_dir, 'train.log')
    file = logging.FileHandler(filename)
    file.setFormatter(formatter)
    logging.getLogger("").addHandler(file)
    logging.info("Saving logs in: {}".format(init_dir))

logging.info(opt)

if not opt.disable_cuda and torch.cuda.is_available():
    opt.device = torch.device('cuda')
else:
    opt.device = torch.device('cpu')
logging.info('Device Information: {}'.format(opt.device))

torch.backends.cudnn.deterministic = True
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

dash = '-' * 90
