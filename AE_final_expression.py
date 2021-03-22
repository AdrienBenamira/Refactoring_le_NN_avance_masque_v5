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
print("---" * 100)
print("TABLE OF TRUTH")

global_sparsity = 0.6
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




masks_imporanta = nn_model_ref.net.fc1.weight_mask.detach().int().numpy()
masks_imporanta_coef = nn_model_ref.net.fc1.weight.detach().numpy()


cpt = 0
for index_masks, masks in enumerate(masks_imporanta):
    if np.sum(masks):
        #print(index_masks, np.sum(masks))
        cpt +=1
        #if np.sum(masks)<100:
        #    for index_m_ici, m_ici in enumerate(masks):
        #        if m_ici:
        #            print(index_m_ici, masks_imporanta_coef[index_masks][index_m_ici])
print(cpt, np.sum(masks_imporanta))
cpt = 0
masks_imporanta = nn_model_ref.net.fc2.weight_mask.detach().int().numpy()
masks_imporanta_coef = nn_model_ref.net.fc2.weight.detach().numpy()
for index_masks, masks in enumerate(masks_imporanta):
    if np.sum(masks):
        #print(index_masks, np.sum(masks))
        cpt +=1
        #if np.sum(masks)<100:
        #    for index_m_ici, m_ici in enumerate(masks):
        #        if m_ici:
        #            print(index_m_ici, masks_imporanta_coef[index_masks][index_m_ici])
print(cpt, np.sum(masks_imporanta))
dictionnaire_feature_name = {}

print(ok)

X_eval_proba_feat = nn_model_ref.all_intermediaire_val
Y_eval_proba = nn_model_ref.Y_val_nn_binaire
X_train_proba_feat = nn_model_ref.all_intermediaire
Y_train_proba = nn_model_ref.Y_train_nn_binaire



print(X_train_proba_feat.shape[1], X_train_proba_feat.shape[1] / 16)






net = AE_binarize(args, X_train_proba_feat.shape[1]).to(device)
nn_model_ref.net = net
nn_model_ref.X_train_nn_binaire = X_train_proba_feat
nn_model_ref.X_val_nn_binaire = X_eval_proba_feat
nn_model_ref.train_from_scractch_2("AE")


#LOAD NN

"""net = AE_binarize(args, X_train_proba_feat.shape[1]).to(device)
nn_model_ref.net = net
nn_model_ref.args.load_nn_path = "results/0.920685_bestacc.pth"
nn_model_ref.load_nn()"""


"""
from sklearn.decomposition import MiniBatchDictionaryLearning

dico = MiniBatchDictionaryLearning(n_components=64, alpha=0.1,
                                                  n_iter=50, batch_size=30)

dico.fit(X_train_proba_feat[Y_train_proba==1])

Embedding_train = dico.transform(X_train_proba_feat[Y_train_proba==1])
print(dico.components_.shape, Embedding_train.shape)
X_approx =np.around(np.dot(dico.components_.transpose(),  Embedding_train.transpose()))
mses = ((X_approx.transpose()-X_train_proba_feat[Y_train_proba==1])**2).mean(axis=1) *100
print(mses.shape)
print(mses)
print(np.mean(mses), np.std(mses))

print()


Embedding_val = dico.transform(X_eval_proba_feat[Y_eval_proba==1])
print(dico.components_.shape, Embedding_val.shape)
X_approx =np.around(np.dot(dico.components_.transpose(),  Embedding_val.transpose()))
mses = ((X_approx.transpose()-X_eval_proba_feat[Y_eval_proba==1])**2).mean(axis=1) *100
print(mses.shape)
print(mses)
print(np.mean(mses), np.std(mses))

print()


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


Embedding_train = dico.transform(X_train_proba_feat)
Embedding_val = dico.transform(X_eval_proba_feat)

clf = LinearRegression()
clf.fit(Embedding_train, Y_train_proba)
score = clf.score(Embedding_val, Y_eval_proba)

print(score)


clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(Embedding_train, Y_train_proba)
score = clf.score(Embedding_val, Y_eval_proba)

print(score)"""


offset_feat = 0
for index_col, col in enumerate(df_expression_bool_m.columns):
    offset_feat = 15*index_col

    for time in range(16):
        if time==0:
            expPOS = df_expression_bool_m_begin[col].values[0]
            dictionnaire_feature_name["Feature_" + str(index_col + time + offset_feat)] = str(expPOS).replace("i",
                                                                                                                str(
                                                                                                                    time))
        elif time==15:
            expPOS = df_expression_bool_m_end[col].values[0]
            dictionnaire_feature_name["Feature_" + str(index_col + time + offset_feat)] = str(expPOS).replace("i",
                                                                                                                str(
                                                                                                                    time))
        else:
            expPOS = df_expression_bool_m[col].values[0]
            dictionnaire_feature_name["Feature_" + str(index_col + time + offset_feat)] = str(expPOS).replace("i",
                                                                                                                str(
                                                                                                                    time))

#print(dictionnaire_feature_name)

dico_tt_embeding_output, dico_tt_embeding_output_name, dico_tt_embeding_feature, dico_tt_embeding_feature_name = get_truth_table_embedding(
    nn_model_ref)




df2 = pd.DataFrame.from_dict(dico_tt_embeding_output, orient='index')
df2_name = pd.DataFrame.from_dict(dico_tt_embeding_output_name, orient='index')
output_name = ["Output"]
df2.columns = output_name
del dico_tt_embeding_output, dico_tt_embeding_output_name
df3 = pd.DataFrame.from_dict(dico_tt_embeding_feature, orient='index')
df3_name = pd.DataFrame.from_dict(dico_tt_embeding_feature_name, orient='index')
del dico_tt_embeding_feature, dico_tt_embeding_feature_name
nfeat = df3.shape[1] - 1

index_to_del = df3[df3[0].isnull()].index.tolist()
df_final = pd.concat([df2, df3], join="inner", axis=1)
df_final = df_final.drop(index_to_del)




uniaue_ele = pd.unique(df_final[[i for i in range(nfeat)]].values.ravel('K'))

print("Uniaue element", uniaue_ele)

for u_e in uniaue_ele:
    if u_e is not None:
        df_final = df_final.replace(u_e, dictionnaire_feature_name[str(u_e)])



print("SAVE ALL")

df_final.to_csv(path_save_model + "final_all.csv")
conditions_or_1, conditions_or_0 = get_final_expression_0_1_version1(df_final)

dico_conditions_or_1 = collections.Counter(conditions_or_1)
dico_conditions_or_0 = collections.Counter(conditions_or_0)
intersection_0_1 = list(set(conditions_or_1) & set(conditions_or_0))

dictionnaire_final = {}
dico_conditions_or_1_list_ici = list(dico_conditions_or_1.keys())
for _, key_1 in tqdm(enumerate(dico_conditions_or_1_list_ici)):
    nbre_0 = 0
    nbre_1 = dico_conditions_or_1[key_1]
    if key_1 in intersection_0_1:
        nbre_0 = dico_conditions_or_0[key_1]
    if nbre_0<nbre_1:
        key_1_clean = key_1
        for time in range(16):
            key_1_clean = key_1_clean.replace(" ", "").replace("[" + str(time) + "+1]",
                                                               "[" + str(time + 1) + "]").replace(
                "[" + str(time) + "-1]", "[" + str(time - 1) + "]")
        liste_f = []
        liste_f2 = []
        liste_ou = key_1_clean.split("&")
        for x in liste_ou:
            x2 = x.replace("(", "").replace(")", "")
            x3 = x.replace("(", "").replace(")", "").replace("~", "")
            liste_f += x2.split("|")
            liste_f2 += x3.split("|")
        dico_count_var_clause = collections.Counter(liste_f)
        list_count_var_clause = list(dico_count_var_clause.keys())
        dico_count_var_clause_2 = collections.Counter(liste_f2)
        list_count_var_clause_2 = list(dico_count_var_clause_2.keys())
        exp_ici = key_1_clean.replace("[", "").replace("]", "")
        # exp_icibis = exp_ici.replace("&", "*").replace("|", "+").replace("~", "N")
        #exp_ici2 = parse_expr(exp_ici, evaluate=False)
        # exp_ici3 = to_dnf(exp_ici2)
        dictionnaire_final[key_1_clean] = [nbre_1, nbre_0, nbre_0 / nbre_1, dico_count_var_clause,
                                           list_count_var_clause,
                                           len(list_count_var_clause), dico_count_var_clause_2, list_count_var_clause_2,
                                           len(list_count_var_clause_2)]#, exp_ici2]

df_final = pd.DataFrame.from_dict(dictionnaire_final, orient='index')
df_final.columns = ["Nbre_1", "Nbre_0", "Nbre_0/Nbre_1 (lower better)", "Expr count", "Expr unique", "Nbre Expr unique", "Var count", "Var unique", "Nbre Var unique"]#, "Jolie EXPRESSION"]
print(df_final)
df_final.to_csv(path_save_model + "classification_all.csv")





