import itertools
import sys
import warnings
import random
import sklearn
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

import matplotlib.pyplot as plt

from matplotlib import colors

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
parser.add_argument("--hidden2", default=config.train_nn.hidden1, type=two_args_str_int)
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

args.load_special = True
args.finetunning = False
args.logs_tensorboard = args.logs_tensorboard.replace("test", "output_table_of_truth")

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





def get_all_masks_from_1clausefilter_allclauseinput(dico_clause, nbre_clause_input, str_filter, mask_filtre, impossibillty):
    all_element_str_filter = str_filter.split("&")
    dico_element = {}
    for index_f, fl in enumerate(all_element_str_filter):
        fl_clean = fl.replace("(", "").replace(")", "").replace(" ", "")
        index_fl_ici = fl_clean.split("[")[-1].split("]")[0]
        flag_inv = '~' in fl_clean
        dico_element[index_f] = [fl_clean, index_fl_ici, flag_inv]
    nbre_element_filter = len(all_element_str_filter)
    lst_all_possible0 = list(itertools.combinations(range(len(all_element_str_filter)), 2))


    dico_all_possible = {}
    for (index_el1, index_el2) in lst_all_possible0:
        flag1 = dico_element[index_el1][2]
        flag2 = dico_element[index_el2][2]
        flag3 = False
        #if flag1 and flag2:
        #    flag3 = True
        if (not flag1) and (not flag2):
            flag3 = True
        d1 = dico_element[index_el1][1]
        d2 = dico_element[index_el2][1]
        if len(d1) == 1:
            d1_int = 0
        else:
            if "-" in d1:
                d1_int = -1 * int(d1.replace("l-", "").replace(" ", ""))
            else:
                d1_int = int(d1.replace("l+", "").replace(" ", ""))
        if len(d2) == 1:
            d2_int = 0
        else:
            if "-" in d2:
                d2_int = -1 * int(d2.replace("l-", "").replace(" ", ""))
            else:
                d2_int = int(d2.replace("l+", "").replace(" ", ""))
        delta = max(abs(d2_int) - abs(d1_int), abs(d1_int) - abs(d2_int))
        if delta>2:
            flag3=False
        dico_all_possible[(index_el1, index_el2)] = [delta, flag3]



    lst_all_possible = list(itertools.product(range(nbre_clause_input), repeat=nbre_element_filter))

    lst_all_possible_f = lst_all_possible.copy()



    #filtre possibilites:
    #print(dico_all_possible)
    #print(impossibillty)


    #print(len(dico_all_possible))

    for pos in list(dico_all_possible.keys()):
        if dico_all_possible[pos][1]:
            delta = dico_all_possible[pos][0]
            for pos2 in list(impossibillty.keys()):
                flag_impossible = delta in impossibillty[pos2]['oppose:non']
                #print("il faut suppremer de possibilites tous les proposetions avec ", pos2, "en posiotion", pos)
                if flag_impossible:
                    lst_all_possible_f = [tuplet for tuplet in lst_all_possible_f if ((tuplet[pos[0]] != pos2[0]) and (tuplet[pos[1]] != pos2[1]))]

    #print(len(lst_all_possible_f))


    for possible_index, possible in enumerate(lst_all_possible_f):
        mask = [""]
        for index_possition, value_position in enumerate(possible):
            el = dico_element[index_possition]
            cl = dico_clause[value_position]
            mask2 = []
            if el[-1]:
                for m in mask:
                    for ccc in cl[1].split(" | "):
                        m2 = m
                        m2 += ccc + " & "
                        mask2.append(m2)

            else:
                for m in mask:
                    m += cl[0] + " & "
                mask2.append(m)
            mask = mask2
            for m in range(len(mask)):
                mask[m] = mask[m].replace("i", el[1])
        for m in range(len(mask)):
            mask[m] = mask[m][:-3]
        for i in range(9):
            for m in range(len(mask)):
                mask[m] = mask[m].replace(str(i)+"+1", str(i+1)).replace(str(i)+"-1", str(i-1)).replace("+0", "")


        #sanity check
        for m in range(len(mask)):
            mask_el = np.array(mask[m].split(" & "))
            mask_unique_el = np.unique(mask_el)
            mask_unique_el_pos = [el for el in mask_unique_el if "~" not in el]
            mask_unique_el_neg = [el.replace("~", "") for el in mask_unique_el if "~" in el]
            intersection = np.intersect1d(mask_unique_el_pos, mask_unique_el_neg)
            if (len(intersection)==0):
                mask_filtre.append(mask[m])






    return mask_filtre


exp_filter = pd.read_csv("/home/adriben/PycharmProjects/Refactoring_le_NN_avance_masque/results/table_of_truth_v2/speck/5/DL_DV_V0_V1/2020_07_27_12_02_34_340448/expression_bool_per_filter.csv")
exp_filter_2eme_couche = pd.read_csv("/home/adriben/PycharmProjects/Refactoring_le_NN_avance_masque/results/table_of_truth_v2/speck/5/DL_DV_V0_V1/2020_07_27_12_02_34_340448/expression_bool_per_filter_2emecouche.csv")


all_masks = {}


def get_possible(all_clause_str_input):
    #print(all_clause_str_input)
    dico_allclause = {}
    for c in all_clause_str_input:
        dico_allclause[c] = {}
        all_el = c.replace("(","").replace(")","").replace(" ","").split("&")
        elemen_pos=[]
        elemen_neg = []
        position_elemen_pos = {}
        position_elemen_neg = {}
        for el in all_el:
            index_fl_ici = el.split("[")[-1].split("]")[0]
            value_fl_ici = el.split("[")[0]
            if '~' in el:
                elemen_neg.append(value_fl_ici.replace("~", ""))
                if value_fl_ici.replace("~", "") in list(position_elemen_neg.keys()):
                    position_elemen_neg[value_fl_ici.replace("~", "")].append(index_fl_ici)
                else:
                    position_elemen_neg[value_fl_ici.replace("~", "")] = [index_fl_ici]
            else:
                elemen_pos.append(value_fl_ici)
                if value_fl_ici in list(position_elemen_pos.keys()):
                    position_elemen_pos[value_fl_ici].append(index_fl_ici)
                else:
                    position_elemen_pos[value_fl_ici] = [index_fl_ici]
        dico_allclause[c]["elemen_pos"] = elemen_pos
        dico_allclause[c]["elemen_neg"] = elemen_neg
        dico_allclause[c]["position_elemen_pos"] = position_elemen_pos
        dico_allclause[c]["position_elemen_neg"] = position_elemen_neg

    impossibility = {}
    for i_c1, c1 in enumerate(all_clause_str_input):
        for i_c2, c2 in enumerate(all_clause_str_input):
            if i_c2> i_c1:
                impossibility[(i_c1, i_c2)] = {}
                elemen_pos_c1 = dico_allclause[c1]["elemen_pos"]
                elemen_neg_c1 = dico_allclause[c1]["elemen_neg"]
                elemen_pos_c2 = dico_allclause[c2]["elemen_pos"]
                elemen_neg_c2 = dico_allclause[c2]["elemen_neg"]

                # cas meme signe
                impossibility[(i_c1, i_c2)]["oppose:non"] = []
                intersect = np.intersect1d(elemen_pos_c1, elemen_neg_c2)
                if len(intersect)>0:
                    for inter_el in intersect:
                        position_elemen_pos_c1 = dico_allclause[c1]["position_elemen_pos"][inter_el]
                        position_elemen_neg_c2 = dico_allclause[c2]["position_elemen_neg"][inter_el]
                        for d1 in position_elemen_pos_c1:
                            if len(d1)==1:
                                d1_int = 0
                            else:
                                if "-" in d1:
                                    d1_int = -1*int(d1.replace("i-","").replace(" ",""))
                                else:
                                    d1_int =  int(d1.replace("i+", "").replace(" ", ""))
                            for d2 in position_elemen_neg_c2:
                                if len(d2) == 1:
                                    d2_int = 0
                                else:
                                    if "-" in d2:
                                        d2_int = -1 * int(d2.replace("i-", "").replace(" ", ""))
                                    else:
                                        d2_int = int(d2.replace("i+", "").replace(" ", ""))
                                delta = max(abs(d2_int) - abs(d1_int), abs(d1_int) - abs(d2_int))
                                if delta not in impossibility[(i_c1, i_c2)]["oppose:non"]:
                                    if delta>0:
                                        impossibility[(i_c1, i_c2)]["oppose:non"].append(delta)

                """intersect = np.intersect1d(elemen_neg_c1, elemen_pos_c2)
                if len(intersect) > 0:
                    for inter_el in intersect:
                        position_elemen_neg_c1 = dico_allclause[c1]["position_elemen_neg"][inter_el]
                        position_elemen_pos_c2 = dico_allclause[c2]["position_elemen_pos"][inter_el]
                        for d1 in position_elemen_neg_c1:
                            if len(d1) == 1:
                                d1_int = 0
                            else:
                                if "-" in d1:
                                    d1_int = -1 * int(d1.replace("i-", "").replace(" ", ""))
                                else:
                                    d1_int = int(d1.replace("i+", "").replace(" ", ""))
                            for d2 in position_elemen_pos_c2:
                                if len(d2) == 1:
                                    d2_int = 0
                                else:
                                    if "-" in d2:
                                        d2_int = -1 * int(d2.replace("i-", "").replace(" ", ""))
                                    else:
                                        d2_int = int(d2.replace("i+", "").replace(" ", ""))
                                delta = max(abs(d2_int) - abs(d1_int), abs(d1_int) - abs(d2_int))

                                if delta not in impossibility[(i_c1, i_c2)]["oppose:non"]:
                                    if delta>0:
                                        impossibility[(i_c1, i_c2)]["oppose:non"].append(delta)"""



    return impossibility







for num_filter, filter in enumerate(exp_filter_2eme_couche.columns):
    if "Filter" in filter:
        #if int(filter.split("_")[1]) not in [3,5,6,7]:
        print()
        #all_masks[filter] = []
        M = []
        index_f = filter.split('_')[1]
        filter2 = "Filter "+str(index_f)
        str_input = exp_filter[filter2].values[0]
        all_clause_str_input = str_input.split("|")


        impossibillty = get_possible(all_clause_str_input)

        dico_clause = {}
        for index_c, cl in enumerate(all_clause_str_input):
            cl_clean = cl.replace("(", "").replace(")", "").replace(" ", "").replace("&", " & ")
            cl_clean_inv = ("~" + cl_clean.replace(" & ", " & ~")).replace("~~", "").replace(" & ", " | ")
            dico_clause[index_c] = [cl_clean, cl_clean_inv]
        nbre_clause_input = len(all_clause_str_input)
        str_filter_all = exp_filter_2eme_couche[filter].values[0]
        str_filter_all_lst = str_filter_all.split("|")
        for str_filter_i, str_filter in enumerate(str_filter_all_lst):
            print(str_filter_i+1, "/", len(str_filter_all_lst))
            M = get_all_masks_from_1clausefilter_allclauseinput(dico_clause, nbre_clause_input, str_filter, M, impossibillty)

        for clause_ici_index, clause_ici in enumerate(M):
            element_clause = clause_ici.split("&")
            M = 0.5 * np.ones((3, 9), dtype=np.uint8)
            for el_c in element_clause:
                if "DL[l]" in el_c:
                    if "~" in el_c:
                        M[0][0] = 0
                    else:
                        M[0][0] = 1
                for l_str in range(1,9):
                    if "DL[l+"+str(l_str)+"]" in el_c:
                        if "~" in el_c:
                            M[0][l_str] = 0
                        else:
                            M[0][l_str] = 1

                if "V0[l]" in el_c:
                    if "~" in el_c:
                        M[1][0] = 0
                    else:
                        M[1][0] = 1
                for l_str in range(1,9):
                    if "V0[l+"+str(l_str)+"]" in el_c:
                        if "~" in el_c:
                            M[1][l_str] = 0
                        else:
                            M[1][l_str] = 1

                if "V1[l]" in el_c:
                    if "~" in el_c:
                        M[2][0] = 0
                    else:
                        M[2][0] = 1
                for l_str in range(1, 9):
                    if "V1[l+" + str(l_str) + "]" in el_c:
                        if "~" in el_c:
                            M[2][l_str] = 0
                        else:
                            M[2][l_str] = 1
            image = M
            row_labels = ["DL", "V0", "V1"]
            col_labels = ["l"] + ["l+"+str(i) for i in range(1,9)]

            cmap = colors.ListedColormap(['black', 'grey', 'white'])
            bounds = [0, 0.25, 0.75, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)

            # tell imshow about color map so that only set colors are used
            img = plt.imshow(image, interpolation='nearest', origin='lower',
                             cmap=cmap, norm=norm)

            # make a color bar
            #plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 0.25, 0.75, 1])

            plt.xticks(range(9), col_labels)
            plt.yticks(range(3), row_labels)
            plt.title('Mask ' + str(clause_ici) + ' clause Filtre numeros ' + str(filter))
            path = "./results/imgs/" + str(filter)
            try:
                os.mkdir(path)
            except OSError as exc:  # Python >2.5
                pass
            plt.savefig(path + "/mask_"+str(clause_ici_index)+".png")
            plt.clf()
        print("Numeros filtre ", num_filter," nom filtre:", filter," nbre de masks ", len(M))
        del M


