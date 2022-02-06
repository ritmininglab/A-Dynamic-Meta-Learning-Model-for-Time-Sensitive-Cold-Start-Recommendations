import pickle
import torch
import torch.nn as nn
from copy import deepcopy
from torch.nn import functional as F
import numpy as np
import math
import torch.optim as optim
from random import randint
import os
import random
from math import log2

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

# simple nn
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
        self.store_parameters()

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())

    def forward(self, support_set_x, support_set_y, query_set_x,
                num_local_update,optimizer):
        for idx in range(num_local_update):
            loss_list = []
            batch_size = 3
            batch_num = math.ceil(len(support_set_x) / batch_size)
            for i in range(batch_num):
                try:
                    if i == (batch_num - 1):
                        supp_xs = support_set_x[batch_size * i:]
                        supp_ys = support_set_y[batch_size * i:]
                    else:
                        supp_xs = support_set_x[batch_size * i:batch_size * (i + 1)]
                        supp_ys = support_set_y[batch_size * i:batch_size * (i + 1)]
                except IndexError:
                    continue
                user_rep = self.model(supp_xs)
                user_rep = torch.mean(user_rep, 0)
                support_set_y_pred = torch.matmul(supp_xs, user_rep.t())
                loss = criterion(support_set_y_pred.view(-1, 1), supp_ys)
                loss_list.append(loss)
            loss = torch.stack(loss_list).mean(0)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        user_rep = self.model(query_set_x)
        user_rep = torch.mean(user_rep, 0)
        query_set_y_pred = torch.matmul(query_set_x, user_rep.t())
        self.model.load_state_dict(self.keep_weight)
        return query_set_y_pred

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys,
                      num_local_update,optimizer):
        query_set_y_pred = self.forward(support_set_xs, support_set_ys, query_set_xs,
                                        num_local_update,optimizer)
        loss_q = criterion(query_set_y_pred.view(-1, 1), query_set_ys)
        return loss_q,query_set_y_pred.view(-1, 1).detach().numpy().tolist()


def dataset_prep(mov_list, movie_dict):
    data_tensor = []
    for mov in mov_list:
        movie_info = movie_dict[mov]
        data_tensor.append(movie_info.float())
    return torch.stack(data_tensor)


def data_generation(user,item, labels, movie_dict, period):
    user_data = {}
    tot_movies = []
    tot_rating = []
    temp_dict = {}
    support_indx = []

    if period>1 :
        for p in range(1, period):
            tot_movies.append(item[p])
            tot_rating.append(labels[p])
        tot_movies = [s for sublist in tot_movies for s in sublist]
        tot_rating = [s for sublist in tot_rating for s in sublist]

        for _ in range(0, 5):
            indx = randint(0, len(item[period]) - 1)
            support_indx.append(indx)
        indexes = [i for i in range(0, len(item[period]) - 1)]
        query_indx = list(set(indexes) - set(support_indx))
        support_movie = [item[period][m] for m in support_indx]+tot_movies
        support_label = [labels[period][m] for m in support_indx]+tot_rating
        query_movie = [item[period][m] for m in query_indx]
        query_label = [labels[period][m] for m in query_indx]


    else:
        for _ in range(0, 5):
            indx = randint(0, len(item[period]) - 1)
            support_indx.append(indx)
        indexes = [i for i in range(0, len(item[period]) - 1)]
        query_indx = list(set(indexes) - set(support_indx))
        support_movie = [item[period][m] for m in support_indx]
        query_movie = [item[period][m] for m in query_indx]
        support_label = [labels[period][m] for m in support_indx]
        query_label = [labels[period][m] for m in query_indx]

    support_tensor = dataset_prep(support_movie, movie_dict)
    support_label = torch.unsqueeze(torch.tensor(support_label).float(), 1)
    query_label = torch.unsqueeze(torch.tensor(query_label).float(), 1)
    query_tensor = dataset_prep(query_movie, movie_dict)
    temp_dict[0] = support_tensor
    temp_dict[1] = support_label
    temp_dict[2] = query_tensor
    temp_dict[3] = query_label
    user_data[user] = temp_dict

    return user_data


# main fumction
if __name__ == "__main__":
    path = os.getcwd()
    active_user_dict = pickle.load(open("{}/final_user_interaction.pkl".format(path), "rb"))
    active_label_dict = pickle.load(open("{}/final_user_rating.pkl".format(path), "rb"))
    movie_dict = pickle.load(open("{}/final_movie_dict.pkl".format(path), "rb"))


    test_user = []
    tot_user = list(active_user_dict.keys())
    for _ in range(0, 144):
        indx = randint(0, len(tot_user) - 1)
        test_user.append(tot_user[indx])
    train_user = list(set(tot_user) - set(test_user))

    for period in range(1, 17):
        # Meta training
        training_loss_p = []

        # Meta learning model
        ml_ss = simple_meta_learning()

        # Global optimizer
        optimizer = optim.Adam(ml_ss.parameters(), lr=1e-4, weight_decay=1e-4)
        maxIte=0
        user_data={}
        for user in tot_user:
            user_data[user] = data_generation(user, active_user_dict[user], active_label_dict[user],
                                        movie_dict, period)

        while maxIte<20:
            training_loss = []
            for user in train_user:
                support_set_x = user_data[user][user][0]
                support_set_y = user_data[user][user][1]
                query_set_x = user_data[user][user][2]
                query_set_y = user_data[user][user][3]
                los_t, pred_y = ml_ss.global_update(support_set_x, support_set_y,
                                                    query_set_x, query_set_y, 1, optimizer)
                training_loss.append(los_t)
                optimizer.zero_grad()
                los_t.backward(retain_graph=True)
                optimizer.step()
                # store global parameters
                ml_ss.store_parameters()

            tot_loss = torch.stack(training_loss).mean(0)
            print('\nMeta Training Loss at iteration {}={}'.format(maxIte, tot_loss))
            maxIte+=1

        # Meta Test
        testing_loss = []
        pred_query_list = []
        query_list = []
        for user in test_user:
            support_set_x = user_data[user][user][0]
            support_set_y = user_data[user][user][1]
            query_set_x = user_data[user][user][2]
            query_set_y = user_data[user][user][3]
            query_list.append(query_set_y)
            loss, pred_y = ml_ss.global_update(support_set_x, support_set_y,
                                               query_set_x, query_set_y, 5, optimizer)
            pred_query_list.append(pred_y)
            testing_loss.append(loss)
        t_loss = torch.stack(testing_loss).mean(0)
        print('\nMeta Test Loss at period {}={}'.format(period,t_loss))

        # Compute percentage recommendation or top N recommendation
        pred_query_list = [l for sub in pred_query_list for l in sub]
        true_list = np.array([l for sub in query_list for l in sub])
        pred_list = np.array([l for sub in pred_query_list for l in sub])
        idx_true = true_list.argsort()[::-1]
        idx_pred = pred_list.argsort()[::-1]
        tot_len = len(idx_pred)
        print(tot_len)

        rmse_result = []
        for per in range(1, 10):
            top_per = int(per * 0.1 * tot_len)
            y_hat = torch.from_numpy(np.array(pred_list[idx_true[:top_per]].reshape
                                              ((top_per, -1)))).float()
            y_tre = torch.from_numpy(true_list[idx_true[:top_per]].reshape((top_per, -1))).float()
            rms = criterion(y_hat, y_tre)
            rmse_result.append(rms)
        print(rmse_result)

        # ndcg
        top_min = 20
        top_max = 40
        array_ndcg = []

        for i in range(top_min, top_max, 2):
            dcg1 = 0
            dcg2 = 0
            for j in range(0, i):
                dcg1 = dcg1 + 1 / log2(1 + idx_pred[j])
                dcg2 = dcg2 + 1 / log2(1 + j + 1)
            ndcg = dcg1 / dcg2
            array_ndcg.append(ndcg)
        print('==NDCG==')
        print(array_ndcg)
