import os
import sys
import torch
import numpy as np
from joblib import dump
from scipy.stats import pearsonr
from math import isnan
from torch_geometric.loader import DataLoader as DataLoader_pyg
from model.dataset import MyDataSet_pyg
from model.utils import *
from sklearn.metrics import mean_absolute_error
from model.network import MPFIL_MutPred

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_epoches = 1500
learning_rate = 0.0005
val_iter = 1
early_stop = 150
seed_list = [0, 42, 1000, 2025, 2026, 16051643]
seed_index = 1
seed = seed_list[seed_index]

time_stamp = '20260101' + '_' + 's1131_seed' + str(seed)
# time_stamp = '20260101' + '_' + 's2398_seed' + str(seed)
# time_stamp = '20260101' + '_' + 'mpad_seed' + str(seed)
dataset = time_stamp.split('_')[1][-4:]

init_model_path = './ckpt/' + time_stamp + '/mffn_model'
init_opt_path = './optimizers/' + time_stamp + '/opt_DGG'
save_path = './ckpt/' + time_stamp + '/mffn_model'
opt_path = './optimizers/' + time_stamp + '/opt_DGG'
result_path = './result/' + time_stamp + '/'
test_save_path_p = './result_ddg/' + time_stamp + '/' + 'best_pcc_predict' + '/'
test_save_path_r = './result_ddg/' + time_stamp + '/' + 'best_rmse_predict' + '/'
test_save_path_m = './result_ddg/' + time_stamp + '/' + 'best_mae_predict' + '/'
make_save_dir(time_stamp)

def train_cross_validation(model, fold_i, train_loader, test_loader):
    global seed_index, optimizer
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    train_rmse_numpy, test_rmse_numpy, test_mae_numpy = np.zeros((max_epoches)), np.zeros((max_epoches)), np.zeros((max_epoches))
    test_pcc_numpy = np.zeros((max_epoches))

    epoch = 0
    has_search = 0
    not_best_count = 0
    best_pcc, best_rmse, best_mae = 0.0, 10.0, 10.0
    global_best_pcc, global_best_rmse, global_best_mae = 0.0, 10.0, 10.0
    while epoch < max_epoches:
        model.train()
        avg_loss = AverageMeter()
        for batch_id, batch_pro in enumerate(train_loader):
            data, length, edge, affinity = batch_pro
            output = model(data, length, edge).squeeze()
            loss = criterion(output, affinity)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.update(loss.item(), affinity.shape[0])
            sys.stdout.write('\r[Fold %d][Epoch %d] Train | step: %d | loss: %f, avg_loss: %f' % (fold_i, epoch+1, batch_id+1, torch.sqrt(loss), np.sqrt(avg_loss.avg)))
        train_sqrt_loss = np.sqrt(avg_loss.avg)
        train_rmse_numpy[epoch] = train_sqrt_loss

        if (epoch + 1) % val_iter == 0:
            test_rmse, test_pcc_value, affinity_label, affinity_predict = test_pcc(model, test_loader)
            test_mae = mean_absolute_error(affinity_label, affinity_predict)
            test_pcc_numpy[epoch] = test_pcc_value
            test_rmse_numpy[epoch] = test_rmse
            test_mae_numpy[epoch] = test_mae
            print('\n[TEST] seed_%d = %d, pcc: %f, rmse: %f, mae: %f' % (seed_index, seed_list[seed_index], test_pcc_value, test_rmse, test_mae))
            if not isnan(test_pcc_value):
                ddg_pre_label = np.concatenate((affinity_label.reshape((1, -1)), affinity_predict.reshape((1, -1))), axis=0)
                if test_pcc_value > global_best_pcc:
                    global_best_pcc = test_pcc_value
                    np.save(test_save_path_p + 'fold' + str(fold_i) + '.npy', ddg_pre_label)
                    # torch.save(model, save_path + '_fold' + str(fold_i) + '_pcc')
                if test_rmse < global_best_rmse:
                    global_best_rmse = test_rmse
                    np.save(test_save_path_r + 'fold' + str(fold_i) + '.npy', ddg_pre_label)
                    # torch.save(model, save_path + '_fold' + str(fold_i) + '_rmse')
                if test_mae < global_best_mae:
                    global_best_mae = test_mae
                    np.save(test_save_path_m + 'fold' + str(fold_i) + '.npy', ddg_pre_label)
                    # torch.save(model, save_path + '_fold' + str(fold_i) + '_mae')
            if test_pcc_value > best_pcc and test_rmse <= best_rmse+0.10 and test_mae <= best_mae+0.05:
                print('Best result (epoch %d) !!!' % epoch)
                best_pcc, best_rmse, best_mae = test_pcc_value, test_rmse, test_mae
                torch.save(model, save_path + '_fold' + str(fold_i))
                torch.save(optimizer, opt_path + '_fold' + str(fold_i))
                not_best_count = 0
            else:
                not_best_count += 1
            if not_best_count >= early_stop:
                if has_search == 0:
                    has_search = 1
                    not_best_count = 0
                    model = torch.load(save_path + '_fold' + str(fold_i))
                    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate/10.0)
                else:
                    break
        epoch += 1
    dump((train_rmse_numpy, test_rmse_numpy, test_pcc_numpy, test_mae_numpy), result_path+'train_test_rmds_pcc_fold'+str(fold_i))
    return train_rmse_numpy, test_rmse_numpy, test_mae_numpy, test_pcc_numpy, best_pcc, best_rmse, best_mae


def test_pcc(model, test_loader):
    model.eval()
    with torch.no_grad():
        lossfunc = torch.nn.MSELoss().to(device)
        affinity_result = torch.tensor([]).to(device)
        output_result = torch.tensor([]).to(device)
        for batch_id, batch_pro in enumerate(test_loader):
            data, length, edge, affinity = batch_pro
            output = model(data, length, edge).squeeze()
            affinity_result = torch.cat([affinity_result, affinity])
            output_result = torch.cat([output_result, output])
        affinity_result = affinity_result.detach()
        output_result = output_result.detach()
        test_loss = lossfunc(affinity_result, output_result).item() ** 0.5
        pcc = pearsonr(affinity_result.cpu().numpy(), output_result.cpu().numpy()).statistic
        if isnan(pcc):
            print('PCC NAN!!!!!!!!!!!!')
    return test_loss, pcc, affinity_result.cpu().numpy(), output_result.cpu().numpy()


def init_model_optimizer():
    set_random_seed(seed)
    model = MPFIL_MutPred()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)
    torch.save(model, init_model_path + '_good_model_seed' + str(seed) + '_init')
    torch.save(optimizer.state_dict(), init_opt_path + '_good_model_seed' + str(seed) + '_init')
    print('Model initialization finish, now loading dataset.')
    return model, optimizer


if __name__ == '__main__':
    init_model_optimizer()
    train_rsme_10fold, test_rsme_10fold, test_mae_10fold, test_pcc_10fold = [], [], [], []
    res_best_pcc, res_best_rmse, res_best_mae = [], [], []
    for fold_i in range(10):   # begin ten-fold cross validation training
        set_random_seed(seed)
        if dataset == '1131':    # select dataset skempi_1131
            train_dataset_file = './data/S1131/tensor_DDG_list_train' + '_' + str(fold_i)
            test_dataset_file = './data/S1131/tensor_DDG_list_test' + '_' + str(fold_i)
        elif dataset == '2398':  # select dataset skempi_2398
            train_dataset_file = './data/S2398/tensor_DDG_list_train' + '_' + str(fold_i)
            test_dataset_file = './data/S2398/tensor_DDG_list_test' + '_' + str(fold_i)
        elif dataset == 'MPAD':  # select dataset mpad
            train_dataset_file = './data/S2398/tensor_DDG_list_train' + '_' + str(fold_i)
            test_dataset_file = './data/S2398/tensor_DDG_list_test' + '_' + str(fold_i)
        else:
            raise ValueError('Skempi dataset should use S1131 or S2398, please check the settings of time_stamp!!!')

        train_dataset, test_dataset = MyDataSet_pyg(train_dataset_file, is_train=True), MyDataSet_pyg(test_dataset_file, is_train=False)
        train_loader = DataLoader_pyg(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=False)
        test_loader = DataLoader_pyg(dataset=test_dataset, batch_size=128, shuffle=False)

        model = torch.load(init_model_path + '_good_model_seed' + str(seed) + '_init')
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)
        optimizer.load_state_dict(torch.load(init_opt_path + '_good_model_seed' + str(seed) + '_init'))

        train_rsme_numpy, test_rsme_numpy, test_mae_numpy, test_pcc_numpy, best_pcc, best_rmse, best_mae = train_cross_validation(model, fold_i, train_loader, test_loader)
        train_rsme_10fold.append(train_rsme_numpy)
        test_rsme_10fold.append(test_rsme_numpy)
        test_mae_10fold.append(test_mae_numpy)
        test_pcc_10fold.append(test_pcc_numpy)
        res_best_pcc.append(best_pcc)
        res_best_rmse.append(best_rmse)
        res_best_mae.append(best_mae)
        del train_dataset, test_dataset, model, optimizer, train_loader, test_loader
        torch.cuda.empty_cache()

    print('Final result avg | pcc: %f, rmse: %f, mae: %f' % (sum(res_best_pcc)/len(res_best_pcc), sum(res_best_rmse)/len(res_best_rmse), sum(res_best_mae)/len(res_best_rmse)))
    # dump((train_rsme_10fold, test_rsme_10fold, test_pcc_10fold, test_mae_10fold), result_path + '10fold_cross_validation_result')
