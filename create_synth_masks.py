import pandas as pd
import numpy as np



import sys
import warnings
import random
import sklearn
import sklearn.neural_network
from src.nn.models.Linear_binarized import Linear_bin
from src.nn.models.Model_AE import AE_binarize
from src.nn.models.Model_linear import NN_linear
from src.nn.nn_model_ref_v2 import NN_Model_Ref_v2
from alibi.explainers import CEM
from sympy import *
warnings.filterwarnings('ignore',category=FutureWarning)
import torch
import os
from src.nn.nn_model_ref import NN_Model_Ref
from sympy.logic import SOPform, POSform
from sympy import symbols
from alibi.explainers import AnchorTabular
from src.data_cipher.create_data import Create_data_binary
from src.utils.initialisation_run import init_all_for_run, init_cipher
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from src.utils.config import Config
import argparse
from src.utils.utils import str2bool, two_args_str_int, two_args_str_float, str2list, transform_input_type
import torch.nn.utils.prune as prune
import math

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation
import numpy as np
from pickle import dump

#NN
from tqdm import tqdm















def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr);
    return (res);


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True);
    return (res);

def make_classifier(input_size=84, d1=1024, d2=512, final_activation='sigmoid'):
    # Input and preprocessing layers
    inp = Input(shape=(input_size,));
    dense1 = Dense(d1)(inp);
    dense1 = BatchNormalization()(dense1);
    dense1 = Activation('relu')(dense1);
    dense2 = Dense(input_size, activation=final_activation)(dense1);
    out = BatchNormalization()(dense2);
    #dense2 = Activation('relu')(dense2);
    #out = Dense(1, activation=final_activation)(dense2);
    model = Model(inputs=inp, outputs=out);
    return (model);

def train_speck_distinguisher(n_feat, X, Y, X_eval, Y_eval, epoch, bs, name_ici="", wdir= "./"):
    # create the network
    net = make_classifier(input_size=n_feat);
    net.compile(optimizer='adam', loss='mse', metrics=['acc']);
    # generate training and validation data
    # set up model checkpoint
    check = make_checkpoint(wdir + 'NN_classifier' + str(6) + "_"+ name_ici + '.h5');
    # create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001));
    # train and evaluate
    h = net.fit(X, Y, epochs=epoch, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr, check]);
    np.save(wdir + 'h_acc_' + str(np.max(h.history['val_acc'])) + "_"+ name_ici +  '.npy', h.history['val_acc']);
    np.save(wdir + 'h_loss' + str(6) + "_"+ name_ici + '.npy', h.history['val_loss']);
    dump(h.history, open(wdir + 'hist' + str(6) + "_"+ name_ici +  '.p', 'wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    net3 = make_classifier(input_size=n_feat);
    net3.load_weights(wdir + 'NN_classifier' + str(6) + "_"+ name_ici +  '.h5')
    return (net3, h);


def generate_binary(n):

  # 2^(n-1)  2^n - 1 inclusive
  bin_arr = range(0, int(math.pow(2,n)))
  bin_arr = [bin(i)[2:] for i in bin_arr]

  # Prepending 0's to binary strings
  max_len = len(max(bin_arr, key=len))
  bin_arr = [i.zfill(max_len) for i in bin_arr]

  return bin_arr

def incremente_dico(nn_model_ref, index_sample, df_dico_second_tot, res2, df_dico_name_tot):
    for index_interet in range(5):
        input_name = nn_model_ref.net.x_input[index_sample][:, 4+index_interet:7+index_interet].detach().cpu().numpy().transpose()
        input_name2 = np.hstack([input_name[i, :] for i in range(input_name.shape[0])])
        input_name3 = '_'.join(map(str, input_name2))
        if input_name3 in list(df_dico_second_tot.keys()):
            for k in range(len(res2[5+index_interet])):
                assert (res2[5+index_interet][k] == df_dico_second_tot[input_name3][k])
        else:
            df_dico_second_tot[input_name3] = res2[5+index_interet]
            df_dico_name_tot[input_name3] = input_name2


    input_name = nn_model_ref.net.x_input[index_sample][:, :2].detach().cpu().numpy().transpose()
    input_name2 = np.hstack([input_name[i, :] for i in range(input_name.shape[0])])
    input_name3 = '_'.join(map(str, input_name2))
    input_name4 = "PAD_PAD_PAD_PAD_" + input_name3
    input_name6 = np.append(np.array(["PAD", "PAD", "PAD", "PAD"]), input_name2, axis=0)
    if input_name4 in list(df_dico_second_tot.keys()):
        for k in range(len(res2[0])):
            assert (res2[0][k] == df_dico_second_tot[input_name4][k])
    else:
        df_dico_second_tot[input_name4] = res2[0]
        df_dico_name_tot[input_name4] = input_name6

    input_name = nn_model_ref.net.x_input[index_sample][:, -2:].detach().cpu().numpy().transpose()
    input_name2 = np.hstack([input_name[i, :] for i in range(input_name.shape[0])])
    input_name3 = '_'.join(map(str, input_name2))
    input_name4 =  input_name3 + "_PAD_PAD_PAD_PAD"
    input_name6 = np.append(input_name2, np.array(["PAD", "PAD", "PAD", "PAD"]), axis=0)
    if input_name4 in list(df_dico_second_tot.keys()):
        for k in range(len(res2[0])):
            assert (res2[-1][k] == df_dico_second_tot[input_name4][k])
    else:
        df_dico_second_tot[input_name4] = res2[-1]
        df_dico_name_tot[input_name4] = input_name6


    return df_dico_second_tot, df_dico_name_tot



def DPLL(exp, M):
    print(exp)
    if len(exp) ==0:
        return (M)
    if len(exp) >0:
        #int_random =random.randint(0,len(exp)-1)
        clause = exp[0].replace(" ", "")
        if '|' not in clause:
            M = increment(M, clause)
            exp = exp[1:].copy()
            return DPLL(exp, M)
        else:
            vars = clause.split('|')
            #int_random = random.randint(0, len(vars) - 1)
            #for var in vars:
            var = vars[0]
            M = increment(M, var.replace(" ", "").replace(")", "").replace("(", ""))
            #exp = exp[1:].copy()
            exp = nettoyer(exp, var.replace(" ", "").replace(")", "").replace("(", "")).copy()
            return DPLL(exp, M)

def increment(M, el_c):
    if "V0" in el_c:
        if "~" in el_c:
            F = 4
        else:
            F = 1
    elif "V1" in el_c:
        if "~" in el_c:
            F = 5
        else:
            F = 2
    elif "DL" in el_c:
        if "~" in el_c:
            F = 3
        else:
            F = 0
    if "[i]" in el_c:
        offset = 1
    elif "[i+1]" in el_c:
        offset = 2
    elif "[i-1]" in el_c:
        offset = 0
    M[F][offset] = 1
    return M

def nettoyer(exp, var):
    exp2 = []
    flag_neg = "~" in var
    if not flag_neg:
        var_neg = "~" + var
    for var2 in exp:
        if not flag_neg:
            if var not in var2 or var_neg in var2:
                exp2.append(var2)
            #if var_neg in var2:
            #    exp2.append(var2.replace(var_neg, "").replace(" ", "").replace(")", "").replace("(", "").replace("|", ""))
            else:
                var3 = var2 + " | "
                var4 = " | " + var2
                var5 = var.replace(var3, "").replace(var4, "")
                if "|" in var:
                    exp2.append(var5)
                else:
                    exp2.append(var5.replace("(", "").replace(")", "").replace(" ", ""))
        else:
            if var not in var2:
                exp2.append(var2)
            else:

                var3 = var + "|"
                var4 = "|" + var
                var5 = var2.replace(" ", "").replace(var3, "").replace(var4, "")


                if "|" in var5:
                    exp2.append(var5.replace(" ", ""))
                else:
                    exp2.append(var5.replace("(", "").replace(")", "").replace(" ", ""))



    exp2.sort(key=lambda x: len(x.split("|")), reverse=False)
    return exp2

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# initiate the parser

print("TODO: MULTITHREADING + ASSERT PATH EXIST DEPEND ON CONDITION + save DDT + deleate some dataset")

config = Config()
parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=config.general.seed, type=two_args_str_int, choices=[i for i in range(100)])
parser.add_argument("--device", default=config.general.device, type=two_args_str_int, choices=[i for i in range(100)])
parser.add_argument("--logs_tensorboard", default=config.general.logs_tensorboard)
parser.add_argument("--models_path", default=config.general.models_path)
parser.add_argument("--models_path_load", default=config.general.models_path_load)
parser.add_argument("--cipher", default=config.general.cipher, choices=["speck", "simon", "aes228", "aes224", "simeck", "gimli"])
parser.add_argument("--nombre_round_eval", default=config.general.nombre_round_eval, type=two_args_str_int)
parser.add_argument("--inputs_type", default=config.general.inputs_type, type=transform_input_type)
parser.add_argument("--word_size", default=config.general.word_size, type=two_args_str_int)
parser.add_argument("--alpha", default=config.general.alpha, type=two_args_str_int)
parser.add_argument("--beta", default=config.general.beta, type=two_args_str_int)
parser.add_argument("--type_create_data", default=config.general.type_create_data, choices=["normal", "real_difference"])


parser.add_argument("--retain_model_gohr_ref", default=config.train_nn.retain_model_gohr_ref, type=str2bool)
parser.add_argument("--load_special", default=config.train_nn.load_special, type=str2bool)
parser.add_argument("--finetunning", default=config.train_nn.finetunning, type=str2bool)
parser.add_argument("--model_finetunne", default=config.train_nn.model_finetunne, choices=["baseline", "cnn_attention", "multihead", "deepset", "baseline_bin"])
parser.add_argument("--load_nn_path", default=config.train_nn.load_nn_path)
parser.add_argument("--countinuous_learning", default=config.train_nn.countinuous_learning, type=str2bool)
parser.add_argument("--curriculum_learning", default=config.train_nn.curriculum_learning, type=str2bool)
parser.add_argument("--nbre_epoch_per_stage", default=config.train_nn.nbre_epoch_per_stage, type=two_args_str_int)
parser.add_argument("--type_model", default=config.train_nn.type_model, choices=["baseline", "cnn_attention", "multihead", "deepset", "baseline_bin"])
parser.add_argument("--nbre_sample_train", default=config.train_nn.nbre_sample_train, type=two_args_str_int)
parser.add_argument("--nbre_sample_eval", default=config.train_nn.nbre_sample_eval, type=two_args_str_int)
parser.add_argument("--num_epochs", default=config.train_nn.num_epochs, type=two_args_str_int)
parser.add_argument("--batch_size", default=config.train_nn.batch_size, type=two_args_str_int)
parser.add_argument("--loss_type", default=config.train_nn.loss_type, choices=["BCE", "MSE", "SmoothL1Loss", "CrossEntropyLoss", "F1"])
parser.add_argument("--lr_nn", default=config.train_nn.lr_nn, type=two_args_str_float)
parser.add_argument("--weight_decay_nn", default=config.train_nn.weight_decay_nn, type=two_args_str_float)
parser.add_argument("--momentum_nn", default=config.train_nn.momentum_nn, type=two_args_str_float)
parser.add_argument("--base_lr", default=config.train_nn.base_lr, type=two_args_str_float)
parser.add_argument("--max_lr", default=config.train_nn.max_lr, type=two_args_str_float)
parser.add_argument("--demicycle_1", default=config.train_nn.demicycle_1, type=two_args_str_int)
parser.add_argument("--optimizer_type", default=config.train_nn.optimizer_type, choices=["Adam", "AdamW", "SGD"])
parser.add_argument("--scheduler_type", default=config.train_nn.scheduler_type, choices=["CyclicLR", "None"])
parser.add_argument("--numLayers", default=config.train_nn.numLayers, type=two_args_str_int)
parser.add_argument("--out_channel0", default=config.train_nn.out_channel0, type=two_args_str_int)
parser.add_argument("--out_channel1", default=config.train_nn.out_channel1, type=two_args_str_int)
parser.add_argument("--hidden1", default=config.train_nn.hidden1, type=two_args_str_int)
parser.add_argument("--kernel_size0", default=config.train_nn.kernel_size0, type=two_args_str_int)
parser.add_argument("--kernel_size1", default=config.train_nn.kernel_size1, type=two_args_str_int)
parser.add_argument("--num_workers", default=config.train_nn.num_workers, type=two_args_str_int)
parser.add_argument("--clip_grad_norm", default=config.train_nn.clip_grad_norm, type=two_args_str_float)
parser.add_argument("--end_after_training", default=config.train_nn.end_after_training, type=str2bool)



parser.add_argument("--load_masks", default=config.getting_masks.load_masks, type=str2bool)
parser.add_argument("--file_mask", default=config.getting_masks.file_mask)
parser.add_argument("--nbre_max_masks_load", default=config.getting_masks.nbre_max_masks_load, type=two_args_str_int)
parser.add_argument("--nbre_generate_data_train_val", default=config.getting_masks.nbre_generate_data_train_val, type=two_args_str_int)
parser.add_argument("--nbre_necessaire_val_SV", default=config.getting_masks.nbre_necessaire_val_SV, type=two_args_str_int)
parser.add_argument("--nbre_max_batch", default=config.getting_masks.nbre_max_batch, type=two_args_str_int)
parser.add_argument("--liste_segmentation_prediction", default=config.getting_masks.liste_segmentation_prediction)
parser.add_argument("--liste_methode_extraction", default=config.getting_masks.liste_methode_extraction, type=transform_input_type)
parser.add_argument("--liste_methode_selection", default=config.getting_masks.liste_methode_selection, type=transform_input_type)
parser.add_argument("--hamming_weigth", default=config.getting_masks.hamming_weigth, type=str2list)
parser.add_argument("--thr_value", default=config.getting_masks.thr_value, type=str2list)
parser.add_argument("--research_new_masks", default=config.getting_masks.research_new_masks, type=str2bool)
parser.add_argument("--save_fig_plot_feature_before_mask", default=config.getting_masks.save_fig_plot_feature_before_mask, type=str2bool)
parser.add_argument("--end_after_step2", default=config.getting_masks.end_after_step2, type=str2bool)


parser.add_argument("--create_new_data_for_ToT", default=config.make_ToT.create_new_data_for_ToT, type=str2bool)
parser.add_argument("--create_ToT_with_only_sample_from_cipher", default=config.make_ToT.create_ToT_with_only_sample_from_cipher, type=str2bool)
parser.add_argument("--nbre_sample_create_ToT", default=config.make_ToT.nbre_sample_create_ToT, type=two_args_str_int)

parser.add_argument("--create_new_data_for_classifier", default=config.make_data_classifier.create_new_data_for_classifier, type=str2bool)
parser.add_argument("--nbre_sample_train_classifier", default=config.make_data_classifier.nbre_sample_train_classifier, type=two_args_str_int)
parser.add_argument("--nbre_sample_val_classifier", default=config.make_data_classifier.nbre_sample_val_classifier, type=two_args_str_int)

parser.add_argument("--retrain_nn_ref", default=config.compare_classifer.retrain_nn_ref, type=str2bool)
parser.add_argument("--num_epch_2", default=config.compare_classifer.num_epch_2, type=two_args_str_int)
parser.add_argument("--batch_size_2", default=config.compare_classifer.batch_size_2, type=two_args_str_int)
parser.add_argument("--classifiers_ours", default=config.compare_classifer.classifiers_ours, type=str2list)
parser.add_argument("--retrain_with_import_features", default=config.compare_classifer.retrain_with_import_features, type=str2bool)
parser.add_argument("--keep_number_most_impactfull", default=config.compare_classifer.keep_number_most_impactfull, type=two_args_str_int)
parser.add_argument("--num_epch_our", default=config.compare_classifer.num_epch_our, type=two_args_str_int)
parser.add_argument("--batch_size_our", default=config.compare_classifer.batch_size_our, type=two_args_str_int)
parser.add_argument("--alpha_test", default=config.compare_classifer.alpha_test, type=two_args_str_float)
parser.add_argument("--quality_of_masks", default=config.compare_classifer.quality_of_masks, type=str2bool)
parser.add_argument("--end_after_step4", default=config.compare_classifer.end_after_step4, type=str2bool)
parser.add_argument("--eval_nn_ref", default=config.compare_classifer.eval_nn_ref, type=str2bool)
parser.add_argument("--compute_independance_feature", default=config.compare_classifer.compute_independance_feature, type=str2bool)
parser.add_argument("--save_data_proba", default=config.compare_classifer.save_data_proba, type=str2bool)

parser.add_argument("--model_to_prune", default=config.prunning.model_to_prune)
parser.add_argument("--values_prunning", default=config.prunning.values_prunning, type=str2list)
parser.add_argument("--layers_NOT_to_prune", default=config.prunning.layers_NOT_to_prune, type=str2list)
parser.add_argument("--save_model_prune", default=config.prunning.save_model_prune, type=str2bool)
parser.add_argument("--logs_layers", default=config.prunning.logs_layers, type=str2bool)
parser.add_argument("--nbre_sample_eval_prunning", default=config.prunning.nbre_sample_eval_prunning, type=two_args_str_int)
parser.add_argument("--inputs_type_prunning", default=config.general.inputs_type, type=transform_input_type)
parser.add_argument("--a_bit", default=config.train_nn.a_bit, type=two_args_str_int)


args = parser.parse_args()

args.load_special = True
args.finetunning = False
args.logs_tensorboard = args.logs_tensorboard.replace("test", "create_synth_masks")
args.inputs_type = args.inputs_type_prunning

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
print("---" * 100)
writer, device, rng, path_save_model, path_save_model_train, name_input = init_all_for_run(args)


print("LOAD CIPHER")
print()
cipher = init_cipher(args)
creator_data_binary = Create_data_binary(args, cipher, rng)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("---" * 100)
print("TABLE OF TRUTH")

nn_model_ref = NN_Model_Ref(args, writer, device, rng, path_save_model, cipher, creator_data_binary, path_save_model_train)
nn_model_ref.load_nn()


flag2 = True
acc_retain=[]
global_sparsity = 0.95
parameters_to_prune = []
for name, module in nn_model_ref.net.named_modules():
    if len(name):
        if name not in ["layers_batch", "layers_conv"]:
            flag = True
            for layer_forbidden in args.layers_NOT_to_prune:
                if layer_forbidden in name:
                    flag = False
            if flag:
                parameters_to_prune.append((module, 'weight'))
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=global_sparsity,
)
tot_sparsity = 0
tot_weight = 0
for name, module in nn_model_ref.net.named_modules():
    if len(name):
        if name not in ["layers_batch", "layers_conv"]:
            flag = True
            for layer_forbidden in args.layers_NOT_to_prune:
                if layer_forbidden in name:
                    flag = False
            if flag:
                tot_sparsity += 100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())
                tot_weight += float(module.weight.nelement()) - float(torch.sum(module.weight == 0))

                if args.logs_layers:
                    print(
                    "Sparsity in {}.weight: {:.2f}%".format(str(name),
                        100. * float(torch.sum(module.weight == 0))
                        / float(module.weight.nelement())
                        )
                    )
if flag2:
    nn_model_ref.eval_all(["val"])





df_exp_1 = pd.read_csv("results/table_of_truth_v2/speck/5/ctdata0l^ctdata1l_ctdata0r^ctdata1r^ctdata0l^ctdata1l_ctdata0l^ctdata0r_ctdata1l^ctdata1r/2020_07_21_13_51_48_837739/expression_bool_per_filter.csv")
df_exp_2 = pd.read_csv("results/table_of_truth_v2/speck/5/ctdata0l^ctdata1l_ctdata0r^ctdata1r^ctdata0l^ctdata1l_ctdata0l^ctdata0r_ctdata1l^ctdata1r/2020_07_21_13_51_48_837739/expression_bool_per_filter_withpadbegin.csv")
df_exp_3 = pd.read_csv("results/table_of_truth_v2/speck/5/ctdata0l^ctdata1l_ctdata0r^ctdata1r^ctdata0l^ctdata1l_ctdata0l^ctdata0r_ctdata1l^ctdata1r/2020_07_21_13_51_48_837739/expression_bool_per_filter_withpadend.csv")

unique_feature = df_exp_1.columns[1:]
print("Nbre de features: ", 16*len(unique_feature))


X_train = np.zeros((16*len(unique_feature), nn_model_ref.X_train_nn_binaire.shape[0]), dtype = np.bool)
X_val = np.zeros((16*len(unique_feature), nn_model_ref.X_val_nn_binaire.shape[0]), dtype = np.bool)

#X_train = np.zeros((16*len(unique_feature), nn_model_ref.X_train_nn_binaire[nn_model_ref.Y_train_nn_binaire==1].shape[0]), dtype = np.bool)
#X_val = np.zeros((16*len(unique_feature), nn_model_ref.X_val_nn_binaire[nn_model_ref.Y_val_nn_binaire==1].shape[0]), dtype = np.bool)


offset = 0
for index in range(16):
    print(index)
    if index == 0:
        df = df_exp_2
    elif index == 15:
        df = df_exp_3
    else:
        df = df_exp_1
    unique_feature = df.columns[1:]
    for index_f, filter in enumerate(unique_feature):
        exp = df[filter].values[0].split("|")
        for index_clause, clause in enumerate(exp):
            clause = clause.replace("(","").replace(")","").replace("i-1", str(index - 1)).replace("i+1", str(index + 1)).replace("i", str(index))
            masks_ici_1 = np.zeros((4,16), dtype = np.int)
            masks_ici_2 = np.zeros((4,16), dtype = np.int)
            elements = clause.split("&")
            for el in elements:
                if "DL" in el:
                    index_row = 0
                elif "V0" in el:
                    index_row = 2
                elif "V1" in el:
                    index_row = 3
                index_columns = int(el.split("[")[-1].split("]")[0])
                masks_ici_1[index_row][index_columns] = 1
                if "~" not in el:
                    masks_ici_2[index_row][index_columns] = 1
            masks_ici_1 = masks_ici_1.reshape(-1,1).squeeze()
            masks_ici_2 = masks_ici_2.reshape(-1,1).squeeze()
            X2 = masks_ici_1 & nn_model_ref.X_train_nn_binaire#[nn_model_ref.Y_train_nn_binaire==1]
            X3 = X2 ^ masks_ici_2
            X4 = np.sum(X3, axis = 1)
            X_f = (X4==0)*1
            if index_clause==0:
                X_train_ici = X_f
            else:
                X_train_ici = X_f | X_train_ici
            X2 = masks_ici_1 & nn_model_ref.X_val_nn_binaire#[nn_model_ref.Y_val_nn_binaire==1]
            X3 = X2 ^ masks_ici_2
            X4 = np.sum(X3, axis=1)
            X_f = (X4 == 0) * 1
            if index_clause == 0:
                X_val_ici = X_f
            else:
                X_val_ici = X_f | X_val_ici
        X_train[offset] = X_train_ici
        X_val[offset] = X_val_ici
        offset+=1

print(offset)
"""
df_clause_unique1 = pd.read_csv("results/table_of_truth_v2/speck/5/ctdata0l^ctdata1l_ctdata0r^ctdata1r^ctdata0l^ctdata1l_ctdata0l^ctdata0r_ctdata1l^ctdata1r/2020_07_21_13_51_48_837739/clause_unique.csv")
df_clause_unique2 = pd.read_csv("results/table_of_truth_v2/speck/5/ctdata0l^ctdata1l_ctdata0r^ctdata1r^ctdata0l^ctdata1l_ctdata0l^ctdata0r_ctdata1l^ctdata1r/2020_07_21_13_51_48_837739/clause_unique_withpadbegin.csv")
df_clause_unique3 = pd.read_csv("results/table_of_truth_v2/speck/5/ctdata0l^ctdata1l_ctdata0r^ctdata1r^ctdata0l^ctdata1l_ctdata0l^ctdata0r_ctdata1l^ctdata1r/2020_07_21_13_51_48_837739/clause_unique_withpadend.csv")



row_v2 = []
unique_feature = []
for r in df_clause_unique1["0"]:
    if r is not None:
        if type(r) is not float:
            if "(" not in r:
                if r not in row_v2:
                    row_v2.append(r)
                    for index in range(1, 15):
                        unique_feature.append(r.replace("i-1",str(index-1)).replace("i+1",str(index+1)).replace("i",str(index)))

for r in df_clause_unique2["0"]:
    if r is not None:
        if type(r) is not float:
            if "(" not in r:
                if r not in row_v2:
                    row_v2.append(r)
                    unique_feature.append(r)

for r in df_clause_unique3["0"]:
    if r is not None:
        if type(r) is not float:
            if "(" not in r:
                if r not in row_v2:
                    row_v2.append(r)
                    unique_feature.append(r)

print("NBRE DE CLAUSE UNIQUE:", len(unique_feature))





feature_masks = {}

X_train = np.zeros((len(unique_feature), nn_model_ref.X_train_nn_binaire.shape[0]), dtype = np.bool)
X_val = np.zeros((len(unique_feature), nn_model_ref.X_val_nn_binaire.shape[0]), dtype = np.bool)


#X_train = np.zeros((len(unique_feature), nn_model_ref.X_train_nn_binaire[nn_model_ref.Y_train_nn_binaire==1].shape[0]), dtype = np.bool)
#X_val = np.zeros((len(unique_feature), nn_model_ref.X_val_nn_binaire[nn_model_ref.Y_val_nn_binaire==1].shape[0]), dtype = np.bool)

for index_clause, clause in tqdm(enumerate(unique_feature)):
    feature_masks[clause] = []
    masks_ici_1 = np.zeros((4,16), dtype = np.int)
    masks_ici_2 = np.zeros((4,16), dtype = np.int)
    elements = clause.split("&")
    for el in elements:
        if "DL" in el:
            index_row = 0
        elif "V0" in el:
            index_row = 2
        elif "V1" in el:
            index_row = 3
        index_columns = int(el.split("[")[-1].split("]")[0])
        masks_ici_1[index_row][index_columns] = 1
        if "~" not in el:
            masks_ici_2[index_row][index_columns] = 1
    masks_ici_1 = masks_ici_1.reshape(-1,1).squeeze()
    masks_ici_2 = masks_ici_2.reshape(-1,1).squeeze()
    #for indexm_input_ici, m_input_ici in enumerate(masks_ici_1):
    #masks_ici_1_int = int("".join(str(i) for i in masks_ici_1), 2)
    #masks_ici_2_int = int("".join(str(i) for i in masks_ici_2), 2)
    feature_masks[clause].append(masks_ici_1)
    feature_masks[clause].append(masks_ici_2)
    X2 = masks_ici_1 & nn_model_ref.X_train_nn_binaire#[nn_model_ref.Y_train_nn_binaire==1]
    X3 = X2 ^ masks_ici_2
    X4 = np.sum(X3, axis = 1)
    X_f = (X4==0)*1
    X_train[index_clause] = X_f
    X2 = masks_ici_1 & nn_model_ref.X_val_nn_binaire#[nn_model_ref.Y_val_nn_binaire==1]
    X3 = X2 ^ masks_ici_2
    X4 = np.sum(X3, axis=1)
    X_f = (X4 == 0) * 1
    X_val[index_clause] = X_f


"""

X_train_f = X_train.transpose()
X_val_f = X_val.transpose()
"""
num_samples = np.sum(nn_model_ref.Y_train_nn_binaire)
print(X_train_f.shape)
vals, counts = np.unique(X_train, axis=1, return_counts=True)
print(vals.shape, counts.shape)
sv = dict(zip(vals,  counts / num_samples))
print(sv)
"""

Y_eval_proba = nn_model_ref.Y_val_nn_binaire
Y_train_proba = nn_model_ref.Y_train_nn_binaire


#net = AE_binarize(args, X_train_f.shape[1], h1= 250).to(device)

net = NN_linear(args, X_train_f.shape[1]).to(device)

nn_model_ref.net = net
nn_model_ref.X_train_nn_binaire = X_train_f
nn_model_ref.X_val_nn_binaire = X_val_f
#nn_model_ref.Y_train_nn_binaire = X_train_f
#nn_model_ref.Y_val_nn_binaire = X_val_f

"""
args.load_nn_path = "./results/create_synth_masks_v2/speck/5/ctdata0l^ctdata1l_ctdata0r^ctdata1r^ctdata0l^ctdata1l_ctdata0l^ctdata0r_ctdata1l^ctdata1r/2020_07_21_17_26_59_603174/0.9966710913033485_bestacc.pth"
nn_model_ref.net.load_state_dict(torch.load(args.load_nn_path,
                map_location=device)['state_dict'], strict=False)
nn_model_ref.net.to(device)
nn_model_ref.net.eval()
"""
nn_model_ref.train_from_scractch("AE")


for global_sparsity in [0, 0.2, 0.4]:
    print(global_sparsity)
    flag2 = True
    acc_retain=[]
    parameters_to_prune = []
    for name, module in nn_model_ref.net.named_modules():
        if len(name):
            if name not in ["layers_batch", "layers_conv"]:
                flag = True
                for layer_forbidden in args.layers_NOT_to_prune:
                    if layer_forbidden in name:
                        flag = False
                if flag:
                    parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=global_sparsity,
    )
    tot_sparsity = 0
    tot_weight = 0
    for name, module in nn_model_ref.net.named_modules():
        if len(name):
            if name not in ["layers_batch", "layers_conv"]:
                flag = True
                for layer_forbidden in args.layers_NOT_to_prune:
                    if layer_forbidden in name:
                        flag = False
                if flag:
                    tot_sparsity += 100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())
                    tot_weight += float(module.weight.nelement()) - float(torch.sum(module.weight == 0))

                    if args.logs_layers:
                        print(
                        "Sparsity in {}.weight: {:.2f}%".format(str(name),
                            100. * float(torch.sum(module.weight == 0))
                            / float(module.weight.nelement())
                            )
                        )
    if flag2:
        nn_model_ref.eval_all(["val"])
    print(nn_model_ref.net.fc1.weight)
    print(nn_model_ref.net.fc1.weight_mask)
    print(nn_model_ref.net.fc1.weight.shape)
    masks_imporanta = nn_model_ref.net.fc1.weight_mask.detach().int().numpy()
    masks_imporanta_coef = nn_model_ref.net.fc1.weight.detach().numpy()


"""
all_feat = list(feature_masks.keys())
for index_masks, masks in enumerate(masks_imporanta):
    print(np.sum(masks))
    for index_m_ici, m_ici in enumerate(masks):
        if m_ici:
            print(all_feat[index_m_ici], masks_imporanta_coef[index_masks][index_m_ici])
"""
