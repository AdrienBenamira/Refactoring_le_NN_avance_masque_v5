import sys
import warnings

from src.nn.DataLoader import DataLoader_cipher_binary
from src.nn.nn_model_ref_3class import NN_Model_Ref_3class
from src.nn.nn_model_ref_8class import NN_Model_Ref_8class

warnings.filterwarnings('ignore',category=FutureWarning)
from collections import Counter
from src.get_masks.evaluate_quality_masks import Quality_masks
from src.classifiers.classifier_all import All_classifier, evaluate_all
from src.ToT.table_of_truth import ToT
from src.data_classifier.Generator_proba_classifier import Genrator_data_prob_classifier
from src.get_masks.get_masks import Get_masks
from src.nn.nn_model_ref import NN_Model_Ref
from src.data_cipher.create_data import Create_data_binary
from src.utils.initialisation_run import init_all_for_run, init_cipher
from src.utils.config import Config
import argparse
from src.utils.utils import str2bool, two_args_str_int, two_args_str_float, str2list, transform_input_type, str2hexa
import numpy as np
import pandas as pd
import torch.nn as nn
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
parser.add_argument("--a_bit", default=config.train_nn.a_bit, type=two_args_str_int)
parser.add_argument("--type_model", default=config.train_nn.type_model, choices=["baseline", "cnn_attention", "multihead", "deepset", "baseline_bin", "baseline_3class"])
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
parser.add_argument("--make_data_equilibre_3class", default=config.train_nn.make_data_equilibre_3class, type=str2bool)
parser.add_argument("--make_data_equilibre_8class", default=config.train_nn.make_data_equilibre_8class, type=str2bool)



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
parser.add_argument("--diff", default=config.train_nn.diff, type=str2hexa)


args = parser.parse_args()

args.logs_tensorboard = args.logs_tensorboard.replace("test", "test_3class")


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
print("STEP 1 : LOAD/ TRAIN NN REF")
print()
print("COUNTINUOUS LEARNING: "+ str(args.countinuous_learning) +  " | CURRICULUM LEARNING: " +  str(args.curriculum_learning) + " | MODEL: " + str(args.type_model))
print()

"""nombre_round_eval = args.nombre_round_eval
args.nombre_round_eval = nombre_round_eval - 2
nn_model_ref2 = NN_Model_Ref(args, writer, device, rng, path_save_model, cipher, creator_data_binary, path_save_model_train)
nn_model_ref2.epochs = 10
nn_model_ref2.train_general(name_input)
args.nombre_round_eval = nombre_round_eval - 1
nn_model_ref3 = NN_Model_Ref(args, writer, device, rng, path_save_model, cipher, creator_data_binary, path_save_model_train)
nn_model_ref3.epochs = 10
nn_model_ref3.net = nn_model_ref2.net
nn_model_ref3.train_general(name_input)
args.nombre_round_eval = nombre_round_eval
nn_model_ref = NN_Model_Ref(args, writer, device, rng, path_save_model, cipher, creator_data_binary, path_save_model_train)
nn_model_ref.net = nn_model_ref3.net"""

nn_model_ref = NN_Model_Ref_8class(args, writer, device, rng, path_save_model, cipher, creator_data_binary, path_save_model_train)
#nn_model_ref.load_nn()
#net_f = nn_model_ref.net.eval()



if args.create_new_data_for_ToT and args.create_new_data_for_classifier:
    del nn_model_ref.X_train_nn_binaire, nn_model_ref.X_val_nn_binaire, nn_model_ref.Y_train_nn_binaire, nn_model_ref.Y_val_nn_binaire
    del nn_model_ref.c0l_train_nn, nn_model_ref.c0l_val_nn, nn_model_ref.c0r_train_nn, nn_model_ref.c0r_val_nn
    del nn_model_ref.c1l_train_nn, nn_model_ref.c1l_val_nn, nn_model_ref.c1r_train_nn, nn_model_ref.c1r_val_nn

del nn_model_ref

args.nombre_round_eval = 5
nn_model_ref2 = NN_Model_Ref(args, writer, device, rng, path_save_model, cipher, creator_data_binary, path_save_model_train)


import time
import torch
val_phase = ['train', 'val']
from torch.utils.data import DataLoader
import glob
args.load_special = True
mypath = "./results/Res_96/*pth"
all_models_trained = {}
for filenames in glob.glob(mypath):
    print(filenames)
    #if filenames =="./results/Res_96/0.948724_bestacc.pth":
    nn_model_ref = NN_Model_Ref(args, writer, device, rng, path_save_model, cipher, creator_data_binary, path_save_model_train)
    args.load_nn_path = filenames
    nn_model_ref.load_nn()
    all_models_trained[filenames] = nn_model_ref.net
    del nn_model_ref

#net1 = all_models_trained["./results/Res_96/0.984361_bestacc.pth"].eval()
#net2 = all_models_trained["./results/Res_96/0.928219_bestacc.pth"].eval()
#net3 = all_models_trained["./results/Res_96/0.948724_bestacc.pth"].eval()
#net4 = all_models_trained["./results/Res_96/0.928219_bestacc.pth"].eval()

data_train = DataLoader_cipher_binary(nn_model_ref2.X_train_nn_binaire, nn_model_ref2.Y_train_nn_binaire, nn_model_ref2.device)
dataloader_train = DataLoader(data_train, batch_size=nn_model_ref2.batch_size,
                              shuffle=False, num_workers=nn_model_ref2.args.num_workers)
data_val = DataLoader_cipher_binary(nn_model_ref2.X_val_nn_binaire, nn_model_ref2.Y_val_nn_binaire, nn_model_ref2.device)
dataloader_val = DataLoader(data_val, batch_size=nn_model_ref2.batch_size,
                              shuffle=False, num_workers=nn_model_ref2.args.num_workers)
if len(val_phase)>1:
    nn_model_ref2.dataloaders = {'train': dataloader_train, 'val': dataloader_val}
else:
    nn_model_ref2.dataloaders = {'val': dataloader_val}
nn_model_ref2.load_general_train()


since = time.time()
n_batches = nn_model_ref2.batch_size
pourcentage = 3
#phase = "val"
#nn_model_ref2.intermediaires = {x:[] for x in val_phase }
#data_train = np.zeros((len(nn_model_ref2.X_train_nn_binaire), 16*nn_model_ref2.args.out_channel1),  dtype = np.uint8)
#data_val = np.zeros((len(nn_model_ref2.X_val_nn_binaire), 16*nn_model_ref2.args.out_channel1), dtype = np.uint8)
#x = nn_model_ref2.net.intermediare.detach().cpu().numpy().astype(np.uint8)
#data_train = np.zeros_like(x, dtype = np.uint8)
#data_val = np.zeros_like(x, dtype = np.uint8)
from tqdm import tqdm
nn_model_ref2.outputs_proba = {x: [] for x in val_phase}
nn_model_ref2.outputs_pred = {x: [] for x in val_phase}
for phase in val_phase:
    nn_model_ref2.net.eval()
    if nn_model_ref2.args.curriculum_learning:
        nn_model_ref2.dataloaders[phase].catgeorie = pourcentage
    running_loss = 0.0
    nbre_sample = 0
    correct1 = torch.zeros(1).long()
    TOT21 = torch.zeros(1).long()
    correct2 = torch.zeros(1).long()
    TOT22 = torch.zeros(1).long()
    correct3 = torch.zeros(1).long()
    TOT23 = torch.zeros(1).long()
    correct4 = torch.zeros(1).long()
    TOT24 = torch.zeros(1).long()
    tk0 = tqdm(nn_model_ref2.dataloaders[phase], total=int(len(nn_model_ref2.dataloaders[phase])))
    for i, data in enumerate(tk0):
        inputs, labels = data
        outputs = net_f(inputs.to(nn_model_ref2.device))
        _, predicted = torch.max(outputs.data, 1)

        outputs = net1(inputs.to(nn_model_ref2.device)[predicted==0])
        preds = (outputs.squeeze(1) > nn_model_ref2.t.to(nn_model_ref2.device)).int().cpu() * 1
        correct1 += (preds == labels[predicted==0].to(nn_model_ref2.device)).cpu().sum().item()
        TOT21 += labels[predicted==0].size(0)

        outputs = net2(inputs.to(nn_model_ref2.device)[predicted == 1])
        preds = (outputs.squeeze(1) > nn_model_ref2.t.to(nn_model_ref2.device)).int().cpu() * 1
        correct2 += (preds == labels[predicted == 1].to(nn_model_ref2.device)).cpu().sum().item()
        TOT22 += labels[predicted == 1].size(0)

        outputs = net3(inputs.to(nn_model_ref2.device)[predicted == 2])
        preds = (outputs.squeeze(1) > nn_model_ref2.t.to(nn_model_ref2.device)).int().cpu() * 1
        correct3 += (preds == labels[predicted == 2].to(nn_model_ref2.device)).cpu().sum().item()
        TOT23 += labels[predicted == 2].size(0)

        outputs = net4(inputs.to(nn_model_ref2.device)[predicted == 3])
        preds = (outputs.squeeze(1) > nn_model_ref2.t.to(nn_model_ref2.device)).int().cpu() * 1
        correct4 += (preds == labels[predicted == 3].to(nn_model_ref2.device)).cpu().sum().item()
        TOT24 += labels[predicted == 3].size(0)

        #preds = (outputs.squeeze(1) > nn_model_ref2.t.to(nn_model_ref2.device)).int().cpu() * 1
        #data_ici = nn_model_ref2.net.intermediare.detach().cpu().numpy().astype(np.uint8)
        #if phase == "train":
        #    data_train[i*nn_model_ref2.batch_size:(i+1)*nn_model_ref2.batch_size,:] = data_ici
        #else:
        #    data_val[i*nn_model_ref2.batch_size:(i+1)*nn_model_ref2.batch_size,:] = data_ici
        #del data_ici

        #nn_model_ref2.intermediaires[phase].append(nn_model_ref2.net.intermediare.detach().cpu().numpy().astype(np.uint8))
        #nn_model_ref2.outputs_proba[phase].append(outputs.detach().cpu().numpy().astype(np.float16))

        nbre_sample += n_batches
    epoch_loss = running_loss / nbre_sample
    acc1 = (correct1.item()) * 1.0 / TOT21.item()
    acc2 = (correct2.item()) * 1.0 / TOT22.item()
    acc3 = (correct3.item()) * 1.0 / TOT23.item()
    acc4 = (correct4.item()) * 1.0 / TOT24.item()
    #acc = (correct1.item() + correct2.item()) * 1.0 / (TOT22.item() + TOT21.item())
    acc = (correct1.item() + correct2.item()+correct3.item() + correct4.item()) * 1.0 / (TOT22.item() + TOT21.item()+TOT23.item() + TOT24.item())
    nn_model_ref2.acc = acc
    print('{} Loss: {:.4f}'.format(
        phase, epoch_loss))
    print('{} Acc: {:.4f}'.format(
        phase, acc1))
    print('{} Acc: {:.4f}'.format(
        phase, acc2))
    print('{} Acc: {:.4f}'.format(
        phase, acc3))
    print('{} Acc: {:.4f}'.format(
        phase, acc4))
    print('{} Acc FINAL: {:.4f}'.format(
        phase, acc))
    #print(desc)
    print()
    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print()
    num1 = int(nn_model_ref2.args.nbre_sample_train_classifier/nn_model_ref2.batch_size)
    num2 = int(nn_model_ref2.args.nbre_sample_val_classifier / nn_model_ref2.batch_size)
    if phase == "train":
        #scaler1 = StandardScaler()
        #del nn_model_ref2.dataloaders["train"]
        #data = data_train #np.array(nn_model_ref2.intermediaires[phase]).astype(np.uint8).reshape(num1 * nn_model_ref2.batch_size, -1)
        #data2 = scaler1.fit_transform(data)
        nn_model_ref2.all_intermediaire = data_train
        #nn_model_ref2.outputs_proba_train = np.array(nn_model_ref2.outputs_proba[phase]).astype(np.float16).reshape(num1 * nn_model_ref2.batch_size, -1)
        #nn_model_ref2.outputs_pred_train = np.array(nn_model_ref2.outputs_pred[phase]).astype(np.float16).reshape(num1 * nn_model_ref2.batch_size, -1)
        #if not nn_model_ref2.args.retrain_nn_ref:
            #del nn_model_ref2.all_intermediaire, data_train

    else:
        #scaler2 = StandardScaler()
        #data = data_val
        #data = np.array(nn_model_ref2.intermediaires[phase]).astype(np.uint8).reshape(num1 * nn_model_ref2.batch_size, -1)
        #data2 = scaler2.fit_transform(data)
        nn_model_ref2.all_intermediaire_val = data_val
        #nn_model_ref2.outputs_proba_val = np.array(nn_model_ref2.outputs_proba[phase]).astype(np.float16).reshape(num2 * nn_model_ref2.batch_size, -1)
        #nn_model_ref2.outputs_pred_val = np.array(nn_model_ref2.outputs_pred[phase]).astype(np.float16).reshape(
        #    num2 * nn_model_ref2.batch_size, -1)
        #if not nn_model_ref2.args.retrain_nn_ref:
            #del nn_model_ref2.all_intermediaire_val, data_val
        del nn_model_ref2.dataloaders[phase]
