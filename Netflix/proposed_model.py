import pickle
from copy import deepcopy
from random import randint
from random import seed
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.nn import functional as F
import os
from random import randint
import os
from math import log2
import random
import time

torch.manual_seed(0)
my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)



# RMSE loss
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


criterion = RMSELoss()


# rnn
class rnn_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(rnn_model, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        combined = torch.cat((input[:,-1,:],hidden[:,-1,:]), 1)
        hidden = self.i2h(combined)
        output = self.i2o(hidden)
        output = self.sigmoid(output)
        return output


# nn
class simple_neural_network(torch.nn.Module):
    def __init__(self, input_dim):
        super(simple_neural_network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.i2o = nn.Linear(64, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        hidden_out = self.fc1(input)
        hidden_out = F.relu(hidden_out)
        hidden_out = self.fc2(hidden_out)
        hidden_out = F.relu(hidden_out)
        output = self.i2o(hidden_out)
        output = self.sigmoid(output)
        return output


class simple_meta_learning(torch.nn.Module):
    def __init__(self):
        super(simple_meta_learning, self).__init__()
        self.model = simple_neural_network(160)
        self.local_lr = 1e-4
        self.store_parameters()

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())

    def forward(self, support_set_x, support_set_y, query_set_x, num_local_update,
                rnn_hidden,optimizer):


        for idx in range(num_local_update):
            loss_list=[]
            batch_size = 3
            batch_num = math.ceil(len(support_set_x) / batch_size)
            for i in range(batch_num):
                try:
                    if i==(batch_num-1):
                        supp_xs = support_set_x[batch_size * i:]
                        supp_ys = support_set_y[batch_size * i:]
                    else:
                        supp_xs = support_set_x[batch_size * i:batch_size * (i + 1)]
                        supp_ys = support_set_y[batch_size * i:batch_size * (i + 1)]
                except IndexError:
                    continue
                user_rep = self.model(supp_xs)
                user_rep = torch.mean(user_rep, 0)

                support_set_y_pred_1 = torch.matmul(supp_xs, user_rep.t())
                loc_hidden=rnn_hidden
                support_set_y_pred_2 = torch.matmul(supp_xs, loc_hidden.t())
                support_set_y_pred = support_set_y_pred_1 + support_set_y_pred_2
                loss = criterion(support_set_y_pred.view(-1, 1), supp_ys)
                loss_list.append(loss)
            loss = torch.stack(loss_list).mean(0)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        user_rep = self.model(query_set_x)
        user_rep = torch.mean(user_rep, 0)

        query_set_y_pred_1 = torch.matmul(query_set_x, user_rep.t())
        query_set_y_pred_2 = torch.matmul(query_set_x, rnn_hidden.t())
        query_set_y_pred = query_set_y_pred_1 + query_set_y_pred_2

        self.model.load_state_dict(self.keep_weight)

        return query_set_y_pred.view(-1, 1),user_rep

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys,
                      num_local_update,rnn_hidden,optimizer):
        query_set_y_pred,time_spec = self.forward(support_set_xs, support_set_ys, query_set_xs,
                                                  num_local_update,rnn_hidden,optimizer)

        loss_q = criterion(query_set_y_pred.view(-1, 1), query_set_ys)
        return loss_q,time_spec,query_set_y_pred.view(-1, 1).detach().numpy().tolist()


def dataset_prep(mov_list, movie_dict):
    data_tensor = []
    for mov in mov_list:
        movie_info = movie_dict[mov]
        data_tensor.append(movie_info.float())
    return torch.stack(data_tensor)


def training_function(ml_ss, support_set_x, support_set_y, query_set_x,
                      query_set_y,rnn_input,optimizer):

    user_loss,time_spec,pred_q = ml_ss.global_update(support_set_x, support_set_y, query_set_x,
                                              query_set_y,1,rnn_input,optimizer)

    return user_loss,time_spec


def valid_funct(ml_ss, test_sup_set_x, test_sup_set_y, test_que_set_x, test_que_set_y,
                rnn_input, optimizer):
    user_los,time_spec,pred_q = ml_ss.global_update(test_sup_set_x, test_sup_set_y, test_que_set_x,
                                              test_que_set_y, 5,rnn_input,optimizer)
    return user_los,time_spec,pred_q


def data_generation(active_user_dict, active_label_dict, movie_dict, period):
    user_data={}
    for user, item, labels in zip(active_user_dict.keys(), active_user_dict.values(),
                                  active_label_dict.values()):
        temp_dict={}
        support_indx = []
        for _ in range(0, 5):
            indx = randint(0, len(item[period]) - 1)
            support_indx.append(indx)
        indexes = [i for i in range(0, len(item[period]) )]
        query_indx = list(set(indexes) - set(support_indx))
        support_movie = [item[period][m] for m in support_indx]
        query_movie = [item[period][m] for m in query_indx]
        support_label = [active_label_dict[user][period][m] for m in support_indx]
        query_label = [active_label_dict[user][period][m] for m in query_indx]

        support_tensor = dataset_prep(support_movie, movie_dict)
        support_label = torch.unsqueeze(torch.tensor(support_label).float(), 1)
        query_label = torch.unsqueeze(torch.tensor(query_label).float(), 1)
        query_tensor = dataset_prep(query_movie, movie_dict)
        temp_dict[0]=support_tensor
        temp_dict[1]=support_label
        temp_dict[2]=query_tensor
        temp_dict[3]=query_label
        user_data[user]=temp_dict

    return user_data


# main function
if __name__ == "__main__":
    path = os.getcwd()
    active_user_dict = pickle.load(open("{}/final_user_interaction.pkl".format(path), "rb"))
    active_label_dict = pickle.load(open("{}/final_user_rating.pkl".format(path), "rb"))
    movie_dict = pickle.load(open("{}/final_movie_dict.pkl".format(path), "rb"))

    #split train test users
    train_user = []
    test_user = []
    tt_user = list(active_user_dict.keys())
    for _ in range(0, 144):
        indx = randint(0, len(tt_user) - 1)
        test_user.append(tt_user[indx])
    train_user = list(set(tt_user) - set(test_user))

    #RNN model
    input_size = 160
    hidden_size = 160
    output_size = 160
    rnn_mod = rnn_model(input_size, hidden_size, output_size)
    intial_hidden=torch.zeros(hidden_size).float()
    user_dynamics={}
    for user in tt_user:
        user_dynamics[user] = torch.reshape(intial_hidden, (1, 160))

    #rnn optimizer
    rnn_optimizer=optim.Adam(rnn_mod.parameters(), lr=1e-4)

    periodic_data = {}
    pred_rating=[]
    true_rating=[]
    for period in range(1, 17):
        periodic_data[period] = data_generation(active_user_dict, active_label_dict, movie_dict,
                                                period)
    time_spec_rep = {}
    for period in range(1, 17):
        user_data=periodic_data[period]
        epoch = 0
        max_epoch=50
        previous_loss = 999
        prev_loss = 999
        training_loss_p = []
        x_tick = []

        # Meta learning model
        ml_ss = simple_meta_learning()

        # meta optimizer
        meta_optimizer = optim.Adam(ml_ss.parameters(), lr=1e-4,weight_decay=1e-4)

        while  epoch <= max_epoch:
            training_loss = []
            if period>1:
                # RNN implementation
                rnn_data = periodic_data[period - 1]
                rnn_loss = []
                ii=0
                for user in tt_user:
                    train_x = torch.cat((rnn_data[user][0], rnn_data[user][2]), dim=0)
                    train_y = torch.cat((rnn_data[user][1], rnn_data[user][3]), dim=0)
                    hidden_ = user_dynamics[user]
                    h_list = []
                    for l in range(len(train_x)):
                        h_list.append(hidden_)
                    hidden = torch.stack(h_list)
                    r_time_s=time.time()
                    hidden_r = rnn_mod(train_x, hidden)
                    hidden_r = torch.mean(hidden_r, dim=0)
                    hidden_r = torch.reshape(hidden_r, (1, 160))
                    train_xx = torch.cat((user_data[user][0], user_data[user][2]), dim=0)
                    train_yy = torch.cat((user_data[user][1], user_data[user][3]), dim=0)
                    y_pred_1 = torch.matmul(train_xx, hidden_r.t())
                    time_s = time_spec_rep[user]
                    y_pred_2 = torch.matmul(train_xx, time_s.t())
                    y_pred = y_pred_1 + y_pred_2
                    loss_rn = criterion(y_pred.view(-1, 1), train_yy)
                    rnn_optimizer.zero_grad()
                    loss_rn.backward(retain_graph=True)
                    rnn_optimizer.step()
                    rnn_loss.append(loss_rn)
                    r_time_e=time.time()
                    user_dynamics[user] = hidden_r
                
                rn_los = torch.stack(rnn_loss).mean(0)
                print('RNN time for one user=',(r_time_e-r_time_s))
                if epoch % 2 == 0:
                    print('RNN loss at epoch {}={}'.format(epoch, rn_los))



            #Meta-training
            for user in train_user:
                support_set_x = user_data[user][0]
                support_set_y = user_data[user][1]
                query_set_x = user_data[user][2]
                query_set_y = user_data[user][3]
                m_time_s=time.time()

                los_tr, time_spec = training_function(ml_ss, support_set_x, support_set_y,
                                                      query_set_x, query_set_y,
                                                      user_dynamics[user], meta_optimizer)
                time_spec_rep[user] = time_spec
                training_loss.append(los_tr)
                meta_optimizer.zero_grad()
                los_tr.backward(retain_graph=True)
                meta_optimizer.step()
                m_time_e=time.time()
                ml_ss.store_parameters()

            print('Meta time=',(m_time_e-m_time_s))
            t_loss = torch.stack(training_loss).mean(0)
            if epoch % 2 == 0:
                print('Meta Training Loss for epoch {}= {}'.format(epoch, t_loss))

            epoch += 1



        # Meta Test
        testing_loss = []
        query_list = []
        pred_query_list = []

        for user in test_user:
            support_set_x = user_data[user][0]
            support_set_y = user_data[user][1]
            query_set_x = user_data[user][2]
            query_set_y = user_data[user][3]
            query_list.append(query_set_y)

            loss, time_spec, pred_q = valid_funct(ml_ss, support_set_x, support_set_y, query_set_x,
                                                  query_set_y, user_dynamics[user], meta_optimizer)

            testing_loss.append(loss)
            time_spec_rep[user] = time_spec
            pred_query_list.append(pred_q)

        t_loss = sum(testing_loss) / len(testing_loss)
        print('\nMeta Test Loss at period {} = {}\n'.format(period, t_loss))

        #Compute percentage recommendation and top N recommendation
        pred_query_list=[l for sub in pred_query_list for l in sub]
        true_list=np.array([l for sub in query_list for l in sub])
        pred_list = np.array([l for sub in pred_query_list for l in sub])
        idx_true = true_list.argsort()[::-1]
        idx_pred = pred_list.argsort()[::-1]
        tot_len=len(idx_pred)

        rmse_result=[]
        for per in range(1,10):
            top_per=int(per*0.1*tot_len)
            y_hat=torch.from_numpy(np.array(pred_list[idx_true[:top_per]].reshape
                                            ((top_per, -1)))).float()
            y_tre=torch.from_numpy(true_list[idx_true[:top_per]].reshape((top_per, -1))).float()
            rms=criterion(y_hat,y_tre)
            rmse_result.append(rms)
        print('==RMSE==')
        print(rmse_result)

#         # ndcg
#         top_min = 20
#         top_max = 40
#         array_ndcg = []

#         for i in range(top_min, top_max, 2):
#             dcg1 = 0
#             dcg2 = 0
#             for j in range(0, i):
#                 dcg1 = dcg1 + 1 / log2(1 + idx_pred[j]+1)
#                 dcg2 = dcg2 + 1 / log2(1 + j + 1)
#             ndcg = dcg1 / dcg2
#             array_ndcg.append(ndcg)
#         print('==NDCG==')
#         print(array_ndcg)
