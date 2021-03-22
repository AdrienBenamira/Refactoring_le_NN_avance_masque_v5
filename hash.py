import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import math
import numpy as np
from torch.autograd import Variable
import numpy as np

import sys
import warnings
import random
import sklearn
import sklearn.neural_network

from src.nn.DataLoader import DataLoader_cipher_binary
from src.nn.models.Linear_binarized import Linear_bin
from src.nn.models.Model_AE import AE_binarize
from src.nn.models.Model_linear import NN_linear
from src.nn.nn_model_ref_v2 import NN_Model_Ref_v2
from alibi.explainers import CEM
from sympy import *
warnings.filterwarnings('ignore',category=FutureWarning)
import torch
import os
import collections
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
from sympy.parsing.sympy_parser import parse_expr
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation
import numpy as np
from pickle import dump
from sympy.logic.boolalg import to_dnf
#NN
from sympy.logic import simplify_logic
from tqdm import tqdm
from sympy import *














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
    dense2 = Dense(d2)(dense1);
    dense2 = BatchNormalization()(dense2);
    dense2 = Activation('relu')(dense2);
    out = Dense(1, activation=final_activation)(dense2);
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


def get_truth_table_embedding(nn_model_ref, dimension_embedding = 16, bz = 500):
    arr2 = generate_binary(dimension_embedding)
    l = []
    for x in arr2:
        l += [np.array([int(d) for d in x])]
    l2 = np.array(l)
    # l3 = l2.reshape(-1, nbre_input, nbre_temps_chaque_input)
    # l4 = np.transpose(l3, axes=(0,2,1))
    dico_tt_embeding_output = {}
    dico_tt_embeding_output_name = {}
    dico_tt_embeding_feature = {}
    dico_tt_embeding_feature_name = {}
    end_ind_bz = l2.shape[0] // bz + 1
    for index_end_bz in range(end_ind_bz):
        input_array_embedding = l2[index_end_bz * bz:(index_end_bz + 1) * bz]
        x_input_f2 = torch.Tensor(input_array_embedding)
        outputs_feature = torch.sigmoid(nn_model_ref.net.decoder(x_input_f2.to(device)))
        nn_model_ref.net.embedding = x_input_f2.to(device)
        outputs = nn_model_ref.net.classify()
        preds = (outputs.squeeze(1) > nn_model_ref.t.to(device)).int().cpu().detach().numpy() * 1
        preds_feat = (outputs_feature.squeeze(1) > nn_model_ref.t.to(device)).int().cpu().detach().numpy() * 1
        for index_input in range(len(input_array_embedding)):
            input_name2 = input_array_embedding[index_input]
            input_name3 = '_'.join(map(str, input_name2))
            dico_tt_embeding_output[input_name3] = preds[index_input]
            dico_tt_embeding_output_name[input_name3] = input_name2
            preds_feat_str = []
            for index_feat, value_feat in enumerate(preds_feat[index_input]):
                if value_feat:
                    preds_feat_str.append("Feature_"+str(index_feat))
            dico_tt_embeding_feature[input_name3] = preds_feat_str
            dico_tt_embeding_feature_name[input_name3] = input_name2
    del l2, l, x_input_f2
    return dico_tt_embeding_output, dico_tt_embeding_output_name, dico_tt_embeding_feature, dico_tt_embeding_feature_name

def get_final_expression_0_1(df_final):
    df_final_1 = df_final[df_final['Output'] == 1].values
    conditions_or_1 = []
    for conditions_or in df_final_1:
        str_1 = ""
        already_seen = []
        for el1 in conditions_or[1:]:
            if el1 is not None:
                if el1 not in already_seen:
                    already_seen.append(el1)
                    str_1 += el1 + " & "
        element_final_1 = str_1[:-2]
        if element_final_1 not in conditions_or_1:
            conditions_or_1.append(element_final_1)
    df_final_0 = df_final[df_final['Output'] == 0].values
    conditions_or_0 = []
    for conditions_or in df_final_0:
        str_0 = ""
        already_seen = []
        for el0 in conditions_or[1:]:
            if el0 is not None:
                if el0 not in already_seen:
                    already_seen.append(el0)
                    str_0 += el0 + " & "
        conditions_or_0.append(str_0[:-2])
        element_final_0 = str_0[:-2]
        if element_final_0 not in conditions_or_0:
            conditions_or_0.append(element_final_0)
    return conditions_or_1, conditions_or_0


def get_final_expression_0_1_version1(df_final):
    df_final_1 = df_final[df_final['Output'] == 1].values
    conditions_or_1 = []
    for conditions_or in df_final_1:
        str_1 = ""
        already_seen = []
        for el1 in conditions_or[1:]:
            if el1 is not None:
                if el1 not in already_seen:
                    already_seen.append(el1)
                    if not "(" in el1:
                        str_1 += "("+el1 + ") & "
                    else:
                        str_1 += el1 + " & "
        element_final_1 = str_1[:-2]
        #if element_final_1 not in conditions_or_1:
        conditions_or_1.append(element_final_1)
    df_final_0 = df_final[df_final['Output'] == 0].values
    conditions_or_0 = []
    for conditions_or in df_final_0:
        str_0 = ""
        already_seen = []
        for el0 in conditions_or[1:]:
            if el0 is not None:
                if el0 not in already_seen:
                    already_seen.append(el0)
                    if not "(" in el0:
                        str_0 += "(" + el0 + ") & "
                    else:
                        str_0 += el0 + " & "
        conditions_or_0.append(str_0[:-2])
        element_final_0 = str_0[:-2]
        #if element_final_0 not in conditions_or_0:
        conditions_or_0.append(element_final_0)
    return conditions_or_1, conditions_or_0

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
args.logs_tensorboard = args.logs_tensorboard.replace("test", "train_AE")
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



#LOAD NN
#------------------------------------------------------------------------------------------------------

def compress(train, test, model, classes=10):
    retrievalB = list([])
    retrievalL = list([])
    for batch_step, (data, target) in enumerate(train):
        #var_data = Variable(data.cuda())
        var_data = Variable(data)

        _,_, code = model(var_data)
        retrievalB.extend(code.cpu().data.numpy())
        retrievalL.extend(target)

    queryB = list([])
    queryL = list([])
    for batch_step, (data, target) in enumerate(test):
        #var_data = Variable(data.cuda())
        var_data = Variable(data)
        _,_, code = model(var_data)
        queryB.extend(code.cpu().data.numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.eye(classes)[np.array(retrievalL)]

    queryB = np.array(queryB)
    queryL = np.eye(classes)[np.array(queryL)]
    return retrievalB, retrievalL, queryB, queryL


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1] # max inner product value
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qB, rB, queryL, retrievalL):
    """
       :param qB: {-1,+1}^{mxq} query bits
       :param rB: {-1,+1}^{nxq} retrieval bits
       :param queryL: {0,1}^{mxl} query label
       :param retrievalL: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = queryL.shape[0]
    map = 0
    for iter in range(num_query):
        # gnd : check if exists any retrieval items with same label
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        # tsum number of items with same label
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        # sort gnd by hamming dist
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum) # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qB, rB, queryL, retrievalL, topk):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param queryL: {0,1}^{mxl} query label
    :param retrievalL: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


#------------------------------------------------------------------------------------------------------

# Hyper Parameters
num_epochs = 40
batch_size = 16
epoch_lr_decrease = 20
learning_rate = 0.001
encode_length = 16
num_classes = 1

print("---" * 100)
print("TABLE OF TRUTH")

global_sparsity = 0.95
df_expression_bool_m = pd.read_csv("./results/expression_bool_per_filter_POS_v2.csv")
df_expression_bool_m_begin = pd.read_csv("./results/expression_bool_per_filter_POS_withpadbegin.csv")
df_expression_bool_m_end = pd.read_csv("./results/expression_bool_per_filter_POS_withpadend.csv")
#df_expression_bool_m = pd.read_csv("./results/expression_bool_per_filter.csv")
#for round_ici in [5, 6, 7, 8, 4]:


nn_model_ref = NN_Model_Ref_v2(args, writer, device, rng, path_save_model, cipher, creator_data_binary, path_save_model_train)
nn_model_ref.load_nn()

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

flag_test = False
acc_retain=[]
nn_model_ref.eval_all(df_expression_bool_m, ["train", "val"])

dictionnaire_feature_name = {}

X_eval_proba_feat = nn_model_ref.all_intermediaire_val
Y_eval_proba = nn_model_ref.Y_val_nn_binaire
X_train_proba_feat = nn_model_ref.all_intermediaire
Y_train_proba = nn_model_ref.Y_train_nn_binaire



print(X_train_proba_feat.shape[1], X_train_proba_feat.shape[1] / 16)




from torch.utils.data import DataLoader

#net = AE_binarize(args, X_train_proba_feat.shape[1]).to(device)
#nn_model_ref.net = net
nn_model_ref.X_train_nn_binaire = X_train_proba_feat
nn_model_ref.X_val_nn_binaire = X_eval_proba_feat

data_train = DataLoader_cipher_binary(X_train_proba_feat, nn_model_ref.Y_train_nn_binaire, device)
dataloader_train = DataLoader(data_train, batch_size=nn_model_ref.batch_size,
                              shuffle=True, num_workers=args.num_workers)
data_val = DataLoader_cipher_binary(nn_model_ref.X_val_nn_binaire, nn_model_ref.Y_val_nn_binaire, nn_model_ref.device)
dataloader_val = DataLoader(data_val, batch_size=nn_model_ref.batch_size,
                              shuffle=False, num_workers=nn_model_ref.args.num_workers)
nn_model_ref.dataloaders = {'train': dataloader_train, 'val': dataloader_val}

#nn_model_ref.train_from_scractch_2("AE")

def uniform_quantize():
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
      return grad_output

  return qfn().apply


# new layer


class activation_quantize_fn(nn.Module):
  def __init__(self):
    super(activation_quantize_fn, self).__init__()
    self.uniform_q = uniform_quantize()

  def forward(self, x):
    activation_q = self.uniform_q(x)
    # print(np.unique(activation_q.detach().numpy()))
    return activation_q


class CNN(nn.Module):
    def __init__(self, inputShape, encode_length, num_classes):
        super(CNN, self).__init__()
        self.act_q = activation_quantize_fn()
        #self.alex = torchvision.models.alexnet(pretrained=True)
        #self.alex.classifier = nn.Sequential(*list(self.alex.classifier.children())[:6])
        self.fc_plus = nn.Linear(inputShape, encode_length)
        self.fc = nn.Linear(encode_length, num_classes, bias=False)

    def forward(self, x):
        #x = self.alex.features(x)
        #x = x.view(x.size(0), 256 * 6 * 6)
        #x = self.alex.classifier(x)
        x = self.fc_plus(x)
        code = self.act_q(x)
        output = self.fc(code)

        return output, x, code


cnn = CNN(X_train_proba_feat.shape[1], encode_length=encode_length, num_classes=num_classes)
# cnn.load_state_dict(torch.load('temp.pkl'))


# Loss and Optimizer
#criterion = nn.CrossEntropyLoss().cuda()
criterion = nn.MSELoss()

optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // epoch_lr_decrease))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


best = 0.0

# Train the Model
for epoch in range(num_epochs):
    #cnn.cuda().train()
    cnn.train()
    adjust_learning_rate(optimizer, epoch)
    for i, (images, labels) in enumerate(nn_model_ref.dataloaders["train"]):
        #images = Variable(images.cuda())
        #labels = Variable(labels.cuda())


        images = Variable(images)
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs, feature, _ = cnn(images)
        loss1 = criterion(outputs.squeeze(1), labels)
        # loss2 = F.mse_loss(torch.abs(feature), Variable(torch.ones(feature.size()).cuda()))
        #loss2 = torch.mean(torch.abs(torch.pow(torch.abs(feature) - Variable(torch.ones(feature.size()).cuda()), 3)))
        loss2 = torch.mean(torch.abs(torch.pow(torch.abs(feature) - Variable(torch.ones(feature.size())), 3)))

        loss = loss1 + 0.1 * loss2
        loss.backward()
        optimizer.step()


        if (i + 1) % (len(dataloader_train) // batch_size / 2) == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(dataloader_train) // batch_size,
                     loss1.item(), loss2.item()))

    # Test the Model
    cnn.eval()  # Change model to 'eval' mode
    correct = 0
    total = 0
    for images, labels in nn_model_ref.dataloaders["val"]:
        #images = Variable(images.cuda(), volatile=True)
        images = Variable(images, volatile=True)
        outputs, _, _ = cnn(images)
        preds = (outputs.squeeze(1) > nn_model_ref.t.to(nn_model_ref.device)).float().cpu() * 1
        #_, predicted = torch.max(outputs.cpu().data, 1)
        total += labels.size(0)
        correct += (preds == labels).sum()

    print('Test Accuracy of the model: %.2f %%' % (100.0 * correct / total))

    if 1.0 * correct / total > best:
        best = 1.0 * correct / total
        torch.save(cnn.state_dict(), 'temp.pkl')

    print('best: %.2f %%' % (best * 100.0))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cifar2.pkl')

# Calculate MAP
# cnn.load_state_dict(torch.load('temp.pkl'))
cnn.eval()
retrievalB, retrievalL, queryB, queryL = compress(nn_model_ref.dataloaders["train"], nn_model_ref.dataloaders["val"], cnn)
print(np.shape(retrievalB))
print(np.shape(retrievalL))
print(np.shape(queryB))
print(np.shape(queryL))

print('---calculate map---')
result = calculate_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL)
print(result)