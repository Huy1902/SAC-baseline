import os
import numpy as np
import torch
import pandas as pd

from environment.RewardFunction import mean_with_cost
from environment.ML1MEnvironment import ML1MEnvironment
from utils import set_random_seed

path_to_data = "dataset/ml1m"
path_to_output = "output/ml1m/"


cuda = 0
if cuda >= 0 and torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    torch.cuda.set_device(cuda)
    device = f"cuda:{cuda}"
else:
    device = "cpu"

print(device)

item_info = np.load(os.path.join(path_to_data, "item_info.npy"))
user_info = np.load(os.path.join(path_to_data, "user_info.npy"))
train = pd.read_csv(os.path.join(path_to_data, "train.csv"), sep="@")
test = pd.read_csv(os.path.join(path_to_data, "test.csv"), sep="@")

params = dict()
params['train'] = train
params['val'] = test
params['item_meta'] = item_info
params['user_meta'] = user_info
params['n_worker'] = 0
params['max_seq_len'] = 50

params['loss_type'] = 'bce'
params['device'] = device
params['l2_coef'] = 0.001
params['lr'] = 0.0003
params['feature_dim'] = 16
params['hidden_dims'] = [256]
params['attn_n_head'] = 2
params['batch_size'] = 128
params['seed'] = 26
params['epoch'] = 2
params['dropout_rate'] = 0.2
params['model_path'] = os.path.join(path_to_output,
                          f"env/ml1m_user_env_lr{params['lr']}_reg{params['l2_coef']}.model")
set_random_seed(params['seed'])

params['loss_type'] = 'bce'
params['device'] = device
params['l2_coef'] = 0.001
params['lr'] = 0.0003
params['feature_dim'] = 16
params['hidden_dims'] = [256]
params['attn_n_head'] = 2
params['batch_size'] = 128
params['epoch'] = 2
params['dropout_rate'] = 0.2
params['max_step'] = 20
params['initial_temper'] = 20
params['reward_function'] = mean_with_cost
params['sasrec_n_layer'] = 2
params['sasrec_d_model'] = 32
params['sasrec_n_head'] = 4
params['sasrec_dropout'] = 0.1
params['sasrec_d_forward'] = 64
params['critic_hidden_dims'] = [256, 64]
params['critic_dropout_rate'] = 0.2
params['n_iter']= [50000]
params['slate_size'] = 10
params['noise_var'] = 0.1
params['q_laplace_smoothness'] = 0.5
params['topk_rate'] = 0
params['empty_start_rate'] = 0
params['buffer_size'] = 100000
params['start_timestamp'] = 2000
params['gamma'] = 0.9
params['train_every_n_step']= 1
params['initial_greedy_epsilon'] = 0
params['final_greedy_epsilon'] = 0
params['elbow_greedy'] = 0.1
params['check_episode'] = 10
params['with_eval'] = False

params['episode_batch_size'] = 32
params['batch_size'] = 64
params['actor_lr'] = 0.01
params['critic_lr'] = 0.01
params['actor_decay'] = 0.00001
params['critic_decay'] = 0.00001
params['target_mitigate_coef'] = 0.01
params['behavior_lr'] = 0.0003
params['behavior_decay'] = 0.00001
params['hyper_actor_coef'] = 0.1
params['advantage_bias'] = 0
params['entropy_coef'] = 0.0001

params['alpha'] = 0.02

env = ML1MEnvironment(params)
n_user = 2
users = env.sample_user(n_user, False)
print(users)