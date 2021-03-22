import sys
import warnings
import random
import sklearn
from torch.nn import functional as F
import sklearn.neural_network
from src.nn.models.Linear_binarized import Linear_bin
from src.nn.models.Model_AE import AE_binarize
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
from src.utils.utils import str2bool, two_args_str_int, two_args_str_float, str2list, transform_input_type, str2hexa
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
parser.add_argument("--hidden2", default=config.train_nn.hidden2, type=two_args_str_int)

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
parser.add_argument("--diff", default=config.train_nn.diff, type=str2hexa)

args = parser.parse_args()

args.load_special = False
args.finetunning = False
args.logs_tensorboard = args.logs_tensorboard.replace("test", "table_of_truth")
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
global_sparsity = 0.2
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



t = nn_model_ref.net.fc1.weight_mask.sum(0).detach().cpu().numpy().tolist()
index_filter_time_keep = [i for i, x in enumerate(t) if x!=0.0]

liste_feature = []
for j in range(args.out_channel0):
    liste_feature += ["F"+str(j)+"_" + str(i) for i in range(16)]

liste_feature_importance = [liste_feature[q] for q in index_filter_time_keep]

print("FEATURES IMPORTANTES: ", liste_feature_importance)
print("NBRE FEATURES IMPORTANTES: ", len(liste_feature_importance))



#nn_model_ref.eval_all(["val"])

x_input = torch.zeros((4,16))
arr2 = generate_binary(9)
l = []
for x in arr2:
    l += [np.array([int(d) for d in x])]
l2 = np.array(l)
print(l2.shape)
l3 = l2.reshape(-1, 3, 3)
#print(l3)
#print(l3.shape)
l4 = np.transpose(l3, axes=(0,2,1))
#print(l4)
#print(l4.shape)
#print("-"*10)

#print(l5)
#print(l5.shape)

V0 = l4[:,1,:]
V1 = l4[:, 2, :]
Dv = V0^V1
x_input_f = np.insert(l4, 1, Dv, 1)

l5 = x_input_f[:,:,1:]

#print(x_input_f)
#print(x_input_f.shape)

rest = np.zeros((512,4,6))
rest2 = np.zeros((512, 4, 7))

rest = np.zeros((512,4,6))
rest[:,:,:2] = l5

rest2[:,:,5:] = l5

x_input_f1b = np.append(rest, x_input_f, axis=2)
x_input_f2 = np.append(x_input_f1b, rest2, axis=2)


x_input_f2 = torch.Tensor(x_input_f2)
df_dico_name_tot = {}
df_dico_second_tot = {}

nn_model_ref.net(x_input_f2.to(device))
for index in range(nn_model_ref.net.x_input.shape[0]):
    res = []
    for index_x, x in enumerate(nn_model_ref.net.x_input[index]):
        res.append(x.detach().cpu().numpy())
    res2 = np.array(res).transpose()
    #print(res2)
    df_dico_second_tot, df_dico_name_tot = incremente_dico(nn_model_ref, index, df_dico_second_tot, res2, df_dico_name_tot)



df = pd.DataFrame.from_dict(df_dico_second_tot)
df_name = pd.DataFrame.from_dict(df_dico_name_tot)



df2 = df.T
df2_name = df_name.T

print(df2.head(5))
print(df2_name.head(5))


#df2.to_csv(path_save_model + "table_of_tructh_0922.csv")
#df2_name.to_csv(path_save_model + "table_of_tructh_0922_name.csv")



#df3 = pd.read_csv(path_save_model + "table_of_tructh_0922.csv")
#df_name3 = pd.read_csv(path_save_model + "table_of_tructh_0922.csv")





#df3 = df2.rename(columns={"Unnamed: 0": "Key"})
df2.columns=["Filter_" + str(i) for i in range(4)]
df2_name.columns=["DL[i-1]","DV[i-1]","V0[i-1]","V1[i-1]","DL[i]","DV[i]","V0[i]","V1[i]","DL[i+1]","DV[i+1]","V0[i+1]","V1[i+1]"]

#df2_name.columns=["DL[i-1]","DV[i-1]","V0[i-1]","V1[i-1]","DL[i]","DV[i]","V0[i]","V1[i]","DL[i+1]","DV[i+1]","V0[i+1]","V1[i+1]"]
print(df2.head(5))
print(df2_name.head(5))
df_m = pd.concat([df2_name,df2], axis = 1)
print(df_m.head())




df_m.to_csv(path_save_model + "table_of_truth_0985_final_with_pad.csv")

df_m2=df_m.drop(df_m.index[df_m["DL[i-1]"] == "PAD"])
df_m_f=df_m2.drop(df_m2.index[df_m2["DL[i+1]"] == "PAD"])

#print (df_m_f)
df_m_f= df_m_f.reset_index()
print (df_m_f.head(5))

df_m_f.to_csv(path_save_model + "table_of_truth_0985_final_without_pad.csv")




dictionnaire_res_fin_expression = {}
dictionnaire_res_fin_expression_POS = {}

dictionnaire_perfiler = {}
dictionnaire_perfiler_POS = {}

doublon = []

expPOS_tot =[]

cpteur = 0

for index_f in range(args.out_channel0):
#for index_f in range(1):
#for index_f in range(1):
    print("Fliter ", index_f)
    #if "F"+str(index_f) in list(dico_important.keys()):
    index_intere = df_m_f.index[df_m_f['Filter_'+str(index_f)] == 1].tolist()
    print()
    if len(index_intere) ==0:
        print("Empty")
    else:
        dictionnaire_res_fin_expression["Filter "+ str(index_f)] = []
        dictionnaire_res_fin_expression_POS["Filter " + str(index_f)] = []
        condtion_filter = []
        for col in ["DL[i-1]", "V0[i-1]", "V1[i-1]", "DL[i]", "V0[i]", "V1[i]", "DL[i+1]", "V0[i+1]", "V1[i+1]"]:
            s = df_m_f[col].values
            my_dict = {"0.0": 0.0, "1.0": 1.0, 0.0: 0.0, 1.0: 1.0}
            s2 = np.array([my_dict[zi] for zi in s])
            condtion_filter.append(s2[index_intere])

        condtion_filter2 = np.array(condtion_filter).transpose()
        condtion_filter3 = [x.tolist() for x in condtion_filter2]
        assert len(condtion_filter3) == len(index_intere)
        assert len(condtion_filter3[0]) == 9
        w1, x1, y1, w2, x2, y2, w3, x3, y3 = symbols('DL[i-1], V0[i-1], V1[i-1], DL[i], V0[i], V1[i], DL[i+1], V0[i+1], V1[i+1]')
        minterms = condtion_filter3
        exp =SOPform([w1, x1, y1, w2, x2, y2, w3, x3, y3], minterms)
        if str(exp) == 'True':
            print(exp, "True")
        else:
            print(exp)
            doublon.append(exp)
            dictionnaire_res_fin_expression["Filter " + str(index_f)].append(exp)
            expV2 = str(exp).split(" | ")
            dictionnaire_perfiler["Filter " + str(index_f)] = [str(exp)] + [x.replace("(", "").replace(")", "") for x in expV2]
            print()
            expPOS = POSform([w1, x1, y1, w2, x2, y2, w3, x3, y3], minterms)
            print(expPOS)
            expPOS_tot.append(str(expPOS))
            dictionnaire_res_fin_expression_POS["Filter " + str(index_f)].append(expPOS)
            expV2POS = str(expPOS).split(" & ")
            print(len(expV2POS))

            cpteur += 2**len(expV2POS)

            dictionnaire_perfiler_POS["Filter " + str(index_f)] = [str(expV2POS)] + [x.replace("(", "").replace(")", "") for
                                                                            x in expV2POS]
            #dictionnaire_res_fin_expression["Filter " + str(index_f)].append(exp)




        print()

print(cpteur)

df_filtre = pd.DataFrame.from_dict(dictionnaire_perfiler, orient='index').T
row = pd.unique(df_filtre[[index_f for index_f in df_filtre.columns]].values.ravel('K'))
df_filtre.to_csv(path_save_model + "dictionnaire_perfiler.csv")
df_row = pd.DataFrame(row)
df_row.to_csv(path_save_model + "clause_unique.csv")
df_expression_bool_m = pd.DataFrame.from_dict(dictionnaire_res_fin_expression, orient='index').T
df_expression_bool_m.to_csv(path_save_model + "expression_bool_per_filter.csv")


df_filtre = pd.DataFrame.from_dict(dictionnaire_perfiler_POS, orient='index').T
row3 = pd.unique(df_filtre[[index_f for index_f in df_filtre.columns]].values.ravel('K'))
df_filtre.to_csv(path_save_model + "dictionnaire_perfiler_POS.csv")
df_row = pd.DataFrame(row3)
df_row.to_csv(path_save_model + "clause_unique_POS.csv")
df_expression_bool_m = pd.DataFrame.from_dict(dictionnaire_res_fin_expression_POS, orient='index').T
df_expression_bool_m.to_csv(path_save_model + "expression_bool_per_filter_POS.csv")


#df_expression_bool.to_csv(path_save_model + "time_important_per_filter.csv")


#------------------------------------------------------------------------------------------------------------------------------

df_m3=df_m.drop(df_m.index[df_m["DL[i-1]"] != "PAD"])
df_m3= df_m3.reset_index()
print (df_m3.head(5))

df_m3.to_csv(path_save_model + "table_of_truth_0985_final_with_only_pad_begin.csv")



dictionnaire_res_fin_expression = {}
dictionnaire_res_fin_expression_POS = {}

dictionnaire_perfiler = {}
dictionnaire_perfiler_POS = {}

doublon = []

expPOS_tot =[]

cpteur = 0

for index_f in range(args.out_channel0):
#for index_f in range(5):
    print("Fliter ", index_f)
    #if "F"+str(index_f) in list(dico_important.keys()):
    index_intere = df_m3.index[df_m3['Filter_'+str(index_f)] == 1].tolist()
    print()
    if len(index_intere) ==0:
        print("Empty")
    else:
        dictionnaire_res_fin_expression["Filter "+ str(index_f)] = []
        dictionnaire_res_fin_expression_POS["Filter " + str(index_f)] = []
        condtion_filter = []
        for col in ["DL[i]", "V0[i]", "V1[i]", "DL[i+1]", "V0[i+1]", "V1[i+1]"]:
            s = df_m3[col].values
            my_dict = {"0.0": 0.0, "1.0": 1.0, 0.0: 0.0, 1.0: 1.0}
            s2 = np.array([my_dict[zi] for zi in s])
            condtion_filter.append(s2[index_intere])

        condtion_filter2 = np.array(condtion_filter).transpose()
        condtion_filter3 = [x.tolist() for x in condtion_filter2]
        assert len(condtion_filter3) == len(index_intere)
        assert len(condtion_filter3[0]) == 6
        w2, x2, y2, w3, x3, y3 = symbols('DL[0], V0[0], V1[0], DL[1], V0[1], V1[1]')
        minterms = condtion_filter3
        exp =SOPform([w2, x2, y2, w3, x3, y3], minterms)


        if str(exp) == 'True':
            print(exp, "True")
        else:
            print(exp)
            doublon.append(exp)
            dictionnaire_res_fin_expression["Filter " + str(index_f)].append(exp)
            expV2 = str(exp).split(" | ")
            dictionnaire_perfiler["Filter " + str(index_f)] = [str(exp)] + [x.replace("(", "").replace(")", "") for x in expV2]
            print()
            expPOS = POSform([w2, x2, y2, w3, x3, y3], minterms)
            print(expPOS)
            expPOS_tot.append(str(expPOS))
            dictionnaire_res_fin_expression_POS["Filter " + str(index_f)].append(expPOS)
            expV2POS = str(expPOS).split(" & ")
            print(len(expV2POS))

            cpteur += 2**len(expV2POS)

            dictionnaire_perfiler_POS["Filter " + str(index_f)] = [str(expV2POS)] + [x.replace("(", "").replace(")", "") for
                                                                            x in expV2POS]
            #dictionnaire_res_fin_expression["Filter " + str(index_f)].append(exp)




    print()

print(cpteur)

df_filtre = pd.DataFrame.from_dict(dictionnaire_perfiler, orient='index').T
row = pd.unique(df_filtre[[index_f for index_f in df_filtre.columns]].values.ravel('K'))
df_filtre.to_csv(path_save_model + "dictionnaire_perfiler_withpadbegin.csv")
df_row = pd.DataFrame(row)
df_row.to_csv(path_save_model + "clause_unique_withpadbegin.csv")
df_expression_bool_m = pd.DataFrame.from_dict(dictionnaire_res_fin_expression, orient='index').T
df_expression_bool_m.to_csv(path_save_model + "expression_bool_per_filter_withpadbegin.csv")


df_filtre = pd.DataFrame.from_dict(dictionnaire_perfiler_POS, orient='index').T
row3 = pd.unique(df_filtre[[index_f for index_f in df_filtre.columns]].values.ravel('K'))
df_filtre.to_csv(path_save_model + "dictionnaire_perfiler_POS_withpadbegin.csv")
df_row = pd.DataFrame(row3)
df_row.to_csv(path_save_model + "clause_unique_POS_withpadbegin.csv")
df_expression_bool_m = pd.DataFrame.from_dict(dictionnaire_res_fin_expression_POS, orient='index').T
df_expression_bool_m.to_csv(path_save_model + "expression_bool_per_filter_POS_withpadbegin.csv")

#df_expression_bool = pd.DataFrame.from_dict(dico_important, orient='index').T
#df_expression_bool.to_csv(path_save_model + "time_important_per_filter_withpadbegin.csv")



#------------------------------------------------------------------------------------------------------------------------------

df_m3=df_m.drop(df_m.index[df_m["DL[i+1]"] != "PAD"])
df_m3= df_m3.reset_index()
print (df_m3.head(5))

df_m3.to_csv(path_save_model + "table_of_truth_0985_final_with_only_pad_end.csv")



dictionnaire_res_fin_expression = {}
dictionnaire_res_fin_expression_POS = {}

dictionnaire_perfiler = {}
dictionnaire_perfiler_POS = {}

doublon = []

expPOS_tot =[]

cpteur = 0

for index_f in range(args.out_channel0):
#for index_f in range(5):

    print("Fliter ", index_f)
    #if "F"+str(index_f) in list(dico_important.keys()):
    index_intere = df_m3.index[df_m3['Filter_'+str(index_f)] == 1].tolist()
    print()
    if len(index_intere) ==0:
        print("Empty")
    else:
        dictionnaire_res_fin_expression["Filter "+ str(index_f)] = []
        dictionnaire_res_fin_expression_POS["Filter " + str(index_f)] = []
        condtion_filter = []
        for col in ["DL[i-1]", "V0[i-1]", "V1[i-1]", "DL[i]", "V0[i]", "V1[i]"]:
            s = df_m3[col].values
            my_dict = {"0.0": 0.0, "1.0": 1.0, 0.0: 0.0, 1.0: 1.0}
            s2 = np.array([my_dict[zi] for zi in s])
            condtion_filter.append(s2[index_intere])

        condtion_filter2 = np.array(condtion_filter).transpose()
        condtion_filter3 = [x.tolist() for x in condtion_filter2]
        assert len(condtion_filter3) == len(index_intere)
        assert len(condtion_filter3[0]) == 6
        w2, x2, y2, w3, x3, y3 = symbols('DL[14], V0[14], V1[14], DL[15], V0[15], V1[15]')
        minterms = condtion_filter3
        exp =SOPform([w2, x2, y2, w3, x3, y3], minterms)


        if str(exp) == 'True':
            print(exp, "True")
        else:
            print(exp)
            doublon.append(exp)
            dictionnaire_res_fin_expression["Filter " + str(index_f)].append(exp)
            expV2 = str(exp).split(" | ")
            dictionnaire_perfiler["Filter " + str(index_f)] = [str(exp)] + [x.replace("(", "").replace(")", "") for x in expV2]
            print()
            expPOS = POSform([w2, x2, y2, w3, x3, y3], minterms)
            print(expPOS)
            expPOS_tot.append(str(expPOS))
            dictionnaire_res_fin_expression_POS["Filter " + str(index_f)].append(expPOS)
            expV2POS = str(expPOS).split(" & ")
            print(len(expV2POS))

            cpteur += 2**len(expV2POS)

            dictionnaire_perfiler_POS["Filter " + str(index_f)] = [str(expV2POS)] + [x.replace("(", "").replace(")", "") for
                                                                            x in expV2POS]
            #dictionnaire_res_fin_expression["Filter " + str(index_f)].append(exp)




    print()

print(cpteur)

df_filtre = pd.DataFrame.from_dict(dictionnaire_perfiler, orient='index').T
row = pd.unique(df_filtre[[index_f for index_f in df_filtre.columns]].values.ravel('K'))
df_filtre.to_csv(path_save_model + "dictionnaire_perfiler_withpadend.csv")
df_row = pd.DataFrame(row)
df_row.to_csv(path_save_model + "clause_unique_withpadend.csv")
df_expression_bool_m = pd.DataFrame.from_dict(dictionnaire_res_fin_expression, orient='index').T
df_expression_bool_m.to_csv(path_save_model + "expression_bool_per_filter_withpadend.csv")


df_filtre = pd.DataFrame.from_dict(dictionnaire_perfiler_POS, orient='index').T
row3 = pd.unique(df_filtre[[index_f for index_f in df_filtre.columns]].values.ravel('K'))
df_filtre.to_csv(path_save_model + "dictionnaire_perfiler_POS_withpadend.csv")
df_row = pd.DataFrame(row3)
df_row.to_csv(path_save_model + "clause_unique_POS_withpadend.csv")
df_expression_bool_m = pd.DataFrame.from_dict(dictionnaire_res_fin_expression_POS, orient='index').T
df_expression_bool_m.to_csv(path_save_model + "expression_bool_per_filter_POS_withpadend.csv")

#df_expression_bool = pd.DataFrame.from_dict(dico_important, orient='index').T
#df_expression_bool.to_csv(path_save_model + "time_important_per_filter_withpadend.csv")






x_input = torch.zeros((4,16))
arr2 = generate_binary(9)



l = []
for x in arr2:
    l += [np.array([int(d) for d in x])]
l2 = np.array(l)


nfilter = 32
input_1 = np.zeros((512, nfilter, 16), dtype=np.int)

for nf in range(nfilter):
    input_1[:,nf,0:9] = l2


ks =3
x_input_f2 = torch.Tensor(input_1)
df_dico_name_tot = {}
df_dico_second_tot = {}
cpt=0
x_output = F.relu(nn_model_ref.net.BN_conv_time2(nn_model_ref.net.conv_time2(x_input_f2)))

res_all = []
for index_batch in range(x_output.shape[0]):
    res = []
    for index_x, x in enumerate(x_output[index_batch]):
        res.append(x.detach().cpu().numpy())
    res2 = np.array(res).transpose()
    res = res2[0]
    res_all.append(res)
res_all = np.array(res_all).transpose()

new_res_binary = []
name_filter_ici = []
cpt = 0
for index_fici in range(32):
    unique_element_filter = np.unique(res_all[index_fici])
    new_filter = np.zeros((len(unique_element_filter), 512), dtype=np.int)
    for index_value, value in enumerate(unique_element_filter):
        bool = (res_all[index_fici] == value)
        new_filter[index_value] = bool
        name_filter_ici.append("Filter_" + str(index_fici) + "_" + str(index_value))
    cpt += unique_element_filter.shape[0]

    if index_fici==0:
        new_res_binary = new_filter
    else:
        new_res_binary = np.concatenate((new_res_binary, new_filter), axis=0)
new_res_binary = new_res_binary.transpose()


for index_batch in range(x_output.shape[0]):
    res = new_res_binary[index_batch]
    input_name2 = input_1[index_batch, :, 0:ks * ks]
    input_name2bis = input_name2.reshape(-1, ks * ks)[0]
    input_name3 = '_'.join(map(str, input_name2bis))

    if input_name3 in list(df_dico_second_tot.keys()):
        assert (df_dico_second_tot[input_name3] == res * 1).all()
        cpt += 1
    else:
        df_dico_second_tot[input_name3] = res * 1
        df_dico_name_tot[input_name3] = input_name2bis




df = pd.DataFrame.from_dict(df_dico_second_tot)
df_name = pd.DataFrame.from_dict(df_dico_name_tot)



df2 = df.T
df2_name = df_name.T

print(df2.head(5))
print(df2_name.head(5))


#df2.to_csv(path_save_model + "table_of_tructh_0922.csv")
#df2_name.to_csv(path_save_model + "table_of_tructh_0922_name.csv")



#df3 = pd.read_csv(path_save_model + "table_of_tructh_0922.csv")
#df_name3 = pd.read_csv(path_save_model + "table_of_tructh_0922.csv")





#df3 = df2.rename(columns={"Unnamed: 0": "Key"})
df2.columns=name_filter_ici
df2_name.columns=["F_k[l]","F_k[l+1]","F_k[l+2]","F_k[l+3]","F_k[l+4]","F_k[l+5]","F_k[l+6]","F_k[l+7]","F_k[l+8]"]



print(df2.head(5))
print(df2_name.head(5))
df_m = pd.concat([df2_name,df2], axis = 1)
df_m= df_m.reset_index()
print(df_m.head())
df_m.to_csv(path_save_model + "table_of_truth_2emecouche_final.csv")


dictionnaire_res_fin_expression = {}
dictionnaire_res_fin_expression_POS = {}
dictionnaire_perfiler = {}
dictionnaire_perfiler_POS = {}
doublon = []
expPOS_tot =[]

cpteur = 0

#for index_f in range(args.out_channel0):
for index_filter, value_filter in enumerate(name_filter_ici):
#for index_f in range(1):
    index_f = value_filter.split('_')[1]
    print("Fliter ", index_filter)
    #if "F"+str(index_f) in list(dico_important.keys()):
    index_intere = df_m.index[df_m[value_filter] == 1].tolist()
    print()
    if len(index_intere) ==0:
        print("Empty")
    else:
        dictionnaire_res_fin_expression[value_filter] = []
        dictionnaire_res_fin_expression_POS[value_filter] = []
        condtion_filter = []
        for col in ["F_k[l]","F_k[l+1]","F_k[l+2]","F_k[l+3]","F_k[l+4]","F_k[l+5]","F_k[l+6]","F_k[l+7]","F_k[l+8]"]:
            s = df_m[col].values
            my_dict = {"0.0": 0.0, "1.0": 1.0, 0.0: 0.0, 1.0: 1.0}
            s2 = np.array([my_dict[zi] for zi in s])
            condtion_filter.append(s2[index_intere])

        condtion_filter2 = np.array(condtion_filter).transpose()
        condtion_filter3 = [x.tolist() for x in condtion_filter2]
        assert len(condtion_filter3) == len(index_intere)
        assert len(condtion_filter3[0]) == 9
        w1, x1, y1, w2, x2, y2, w3, x3, y3 = symbols('F_k[l],F_k[l+1],F_k[l+2],F_k[l+3],F_k[l+4],F_k[l+5],F_k[l+6],F_k[l+7],F_k[l+8]')
        minterms = condtion_filter3
        exp =SOPform([w1, x1, y1, w2, x2, y2, w3, x3, y3], minterms)
        if str(exp) == 'True':
            print(exp, "True")
        else:
            print(str(exp).replace("k", str(index_f)))
            doublon.append(str(exp).replace("k", str(index_f)))
            dictionnaire_res_fin_expression[value_filter].append(str(exp).replace("k", str(index_f)))
            expV2 = str(exp).replace("k", str(index_f)).split(" | ")
            dictionnaire_perfiler[value_filter] = [str(exp).replace("k", str(index_f))] + [x.replace("(", "").replace(")", "") for x in expV2]
            print()
            #expPOS = POSform([w1, x1, y1, w2, x2, y2, w3, x3, y3], minterms)
            #print(str(expPOS).replace("k", str(index_f)))
            #expPOS_tot.append(str(expPOS).replace("k", str(index_f)))
            #dictionnaire_res_fin_expression_POS[value_filter].append(str(expPOS).replace("k", str(index_f)))
            #expV2POS = str(expPOS).replace("k", str(index_f)).split(" & ")
            #print(len(expV2POS))

            #cpteur += 2**len(expV2POS)

            #dictionnaire_perfiler_POS[value_filter] = [str(expPOS).replace("k", str(index_f))] + [x.replace("(", "").replace(")", "") for
            #                                                                x in expV2POS]
            #dictionnaire_res_fin_expression["Filter " + str(index_f)].append(exp)

    print()

print(cpteur)

df_filtre = pd.DataFrame.from_dict(dictionnaire_perfiler, orient='index').T
row = pd.unique(df_filtre[[index_f for index_f in df_filtre.columns]].values.ravel('K'))
df_filtre.to_csv(path_save_model + "dictionnaire_perfiler_2emecouche.csv")
df_row = pd.DataFrame(row)
df_row.to_csv(path_save_model + "clause_unique_2emecouche.csv")
df_expression_bool_m = pd.DataFrame.from_dict(dictionnaire_res_fin_expression, orient='index').T
df_expression_bool_m.to_csv(path_save_model + "expression_bool_per_filter_2emecouche.csv")


#df_filtre = pd.DataFrame.from_dict(dictionnaire_perfiler_POS, orient='index').T
#row3 = pd.unique(df_filtre[[index_f for index_f in df_filtre.columns]].values.ravel('K'))
#df_filtre.to_csv(path_save_model + "dictionnaire_perfiler_POS_2emecouche.csv")
#df_row = pd.DataFrame(row3)
#df_row.to_csv(path_save_model + "clause_unique_POS_2emecouche.csv")
#df_expression_bool_m = pd.DataFrame.from_dict(dictionnaire_res_fin_expression_POS, orient='index').T
#df_expression_bool_m.to_csv(path_save_model + "expression_bool_per_filter_POS_2emecouche.csv")

#df_expression_bool = pd.DataFrame.from_dict(dico_important, orient='index').T
#df_expression_bool.to_csv(path_save_model + "time_important_per_filter_2emecouche.csv")



print(ok)














all_masksPOS = [[], [], [],[], [], []]
for exp_iter in expPOS_tot:
    expression = exp_iter.split("&")
    M = np.zeros((6, 16), dtype=np.uint8)
    #for iterici in range(2*len(expression)):
    M2 = DPLL(expression, M)
    for offset in range(15):
        for index_m_f, m_f in enumerate(M2):
            liste_ici = m_f.tolist()
            result = int("".join(str(i) for i in liste_ici), 2)
            all_masksPOS[index_m_f].append(result>>offset)

print("NBRE DE MASKS CREE:", len(all_masksPOS[0]))

with open(path_save_model + "masks_allPOS.txt", "w") as file:
    for i in range(6):
        file.write(str(all_masksPOS[i]))
        file.write("\n")


row_v2 = []
for r in row:
    if r is not None:
        if "(" not in r:
            row_v2.append(r)

print("NBRE DE CLAUSE UNIQUE:", len(row_v2))
all_masks = [[], [], [],[], [], []]
for clause_ici in row_v2:
    element_clause = clause_ici.split("&")
    for index_mask in range(1,15):
        M = np.zeros((6,16), dtype = np.uint8)
        for el_c in element_clause:
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
                offset = 0
            elif "[i+1]" in el_c:
                offset = 1
            elif "[i-1]" in el_c:
                offset = -1
            M[F][index_mask + offset] = 1

        for index_m_f, m_f in enumerate(M):
            liste_ici = m_f.tolist()
            result = int("".join(str(i) for i in liste_ici), 2)
            all_masks[index_m_f].append(result)

print("NBRE DE MASKS CREE:", len(all_masks[0]))

with open(path_save_model + "masks_all.txt", "w") as file:
    for i in range(6):
        file.write(str(all_masks[i]))
        file.write("\n")



del nn_model_ref

for round_ici in [5, 6, 7, 8, 4]:

    args.nombre_round_eval = round_ici

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




    #print(X_eval_proba_feat[0].tolist())
    #print(nn_model_ref.all_intermediaire_val[0].tolist())

    #print(ok)

    X_eval_proba_feat = nn_model_ref.all_intermediaire_val
    Y_eval_proba = nn_model_ref.Y_val_nn_binaire
    X_train_proba_feat = nn_model_ref.all_intermediaire
    Y_train_proba = nn_model_ref.Y_train_nn_binaire

    print(X_train_proba_feat.shape[1], X_train_proba_feat.shape[1]/16)



    #net = Linear_bin(args, X_train_proba_feat.shape[1]).to(device)
    net = AE_binarize(args, X_train_proba_feat.shape[1]).to(device)

    nn_model_ref.net = net
    nn_model_ref.X_train_nn_binaire = X_train_proba_feat
    nn_model_ref.X_val_nn_binaire = X_eval_proba_feat
    nn_model_ref.Y_train_nn_binaire = X_train_proba_feat
    nn_model_ref.Y_val_nn_binaire = X_eval_proba_feat

    nn_model_ref.train_from_scractch("AE")

    print(ok)

    """

    net_retrain, h = train_speck_distinguisher(X_train_proba_feat.shape[1], X_train_proba_feat,
                                                       Y_train_proba, X_eval_proba_feat, Y_eval_proba,
                                                       bs=5000,
                                                       epoch=20, name_ici="test")"""


    """clf = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(1024,512))
    clf.fit(X_train_proba_feat, Y_train_proba)
    
    predict_fn = lambda x: clf.predict(x)
    print('Train accuracy: ', accuracy_score(Y_train_proba, predict_fn(X_train_proba_feat)))
    print('Test accuracy: ', accuracy_score(Y_eval_proba, predict_fn(X_eval_proba_feat)))
    
    #predict_fn = lambda x: net_retrain.predict(x)[0]
    explainer = AnchorTabular(predict_fn, [i for i in range(X_train_proba_feat.shape[1])], categorical_names={0:"R", 1:"S"}, seed=1)
    explainer.fit(X_train_proba_feat, disc_perc=[75])
    idx = 0
    #X = X_eval_proba_feat[idx].reshape((1,) + X_eval_proba_feat[idx].shape)
    #print('Prediction: ', explainer.predictor(X)[0])
    
    idx = 0
    print('Prediction: ', explainer.predictor(X_eval_proba_feat[idx].reshape(1, -1))[0])
    print('Label: ', Y_eval_proba[0])
    
    
    
    explanation = explainer.explain(X_eval_proba_feat[idx], threshold=0.95)
    #explanation = explainer.explain(X, threshold=0.95)
    print('Anchor: %s' % (' AND '.join(explanation.anchor)))
    print('Precision: %.2f' % explanation.precision)
    print('Coverage: %.2f' % explanation.coverage)

    print()
    """
    del nn_model_ref
