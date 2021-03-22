import sys
import warnings


from src.nn.DataLoader import DataLoader_cipher_binary

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

parser.add_argument("--diff", default=config.train_nn.diff, type=str2hexa)
parser.add_argument("--retain_model_gohr_ref", default=config.train_nn.retain_model_gohr_ref, type=str2bool)
parser.add_argument("--load_special", default=config.train_nn.load_special, type=str2bool)
parser.add_argument("--finetunning", default=config.train_nn.finetunning, type=str2bool)
parser.add_argument("--model_finetunne", default=config.train_nn.model_finetunne, choices=["baseline", "cnn_attention", "multihead", "deepset", "baseline_bin"])
parser.add_argument("--load_nn_path", default=config.train_nn.load_nn_path)
parser.add_argument("--countinuous_learning", default=config.train_nn.countinuous_learning, type=str2bool)
parser.add_argument("--curriculum_learning", default=config.train_nn.curriculum_learning, type=str2bool)
parser.add_argument("--nbre_epoch_per_stage", default=config.train_nn.nbre_epoch_per_stage, type=two_args_str_int)
parser.add_argument("--a_bit", default=config.train_nn.a_bit, type=two_args_str_int)
parser.add_argument("--type_model", default=config.train_nn.type_model, choices=["baseline", "cnn_attention", "multihead", "deepset", "baseline_bin", "baseline_bin_v2"])
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
parser.add_argument("--limit", default=config.train_nn.limit, type=two_args_str_int)
parser.add_argument("--kstime", default=config.train_nn.kstime, type=two_args_str_int)
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



args = parser.parse_args()

from src.classifiers.nn_classifier_keras import train_speck_distinguisher
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


import glob
from torch.utils.data import DataLoader


args.load_special = True
mypath = "./results/Res_96/*pth"
all_models_trained = {}
for filenames in glob.glob(mypath):
    print(filenames)
    nn_model_ref = NN_Model_Ref(args, writer, device, rng, path_save_model, cipher, creator_data_binary, path_save_model_train)
    args.load_nn_path = filenames
    nn_model_ref.load_nn()
    all_models_trained[filenames] = nn_model_ref.net.eval()

    del nn_model_ref

all_models_trained["coef"] = [1, 1, 1, 1, 1, 1, 1, 1]
#[0.125,0.008, 0.06, 0.5, 0.03, 0.0125, 0.008, 0.25]


nn_model_ref = NN_Model_Ref(args, writer, device, rng, path_save_model, cipher, creator_data_binary, path_save_model_train)


data_train = DataLoader_cipher_binary(nn_model_ref.X_train_nn_binaire, nn_model_ref.Y_train_nn_binaire, nn_model_ref.device)
dataloader_train = DataLoader(data_train, batch_size=nn_model_ref.batch_size,
                              shuffle=False, num_workers=nn_model_ref.args.num_workers)
data_val = DataLoader_cipher_binary(nn_model_ref.X_val_nn_binaire, nn_model_ref.Y_val_nn_binaire, nn_model_ref.device)
dataloader_val = DataLoader(data_val, batch_size=nn_model_ref.batch_size,
                              shuffle=False, num_workers=nn_model_ref.args.num_workers)

nn_model_ref.dataloaders = {'train': dataloader_train, 'val': dataloader_val}
nn_model_ref.load_general_train()

import time
import torch
val_phase = ['train', 'val']
since = time.time()
n_batches = nn_model_ref.batch_size
pourcentage = 3
#phase = "val"
#nn_model_ref.intermediaires = {x:[] for x in val_phase }
data_train = np.zeros((len(nn_model_ref.X_train_nn_binaire), 64*8),  dtype = np.float16)
data_val = np.zeros((len(nn_model_ref.X_val_nn_binaire), 64*8), dtype = np.float16)

#data_train1 = np.zeros((len(nn_model_ref.X_train_nn_binaire), 8),  dtype = np.float16)
#data_val1 = np.zeros((len(nn_model_ref.X_val_nn_binaire), 8), dtype = np.float16)

#x = nn_model_ref.net.intermediare.detach().cpu().numpy().astype(np.uint8)
#data_train = np.zeros_like(x, dtype = np.uint8)
#data_val = np.zeros_like(x, dtype = np.uint8)
from tqdm import tqdm
nn_model_ref.outputs_proba = {x: [] for x in val_phase}
nn_model_ref.outputs_pred = {x: [] for x in val_phase}

methode1 = False

for phase in val_phase:
    nn_model_ref.net.eval()
    if nn_model_ref.args.curriculum_learning:
        nn_model_ref.dataloaders[phase].catgeorie = pourcentage
    running_loss = 0.0
    nbre_sample = 0
    TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(
        1).long()
    tk0 = tqdm(nn_model_ref.dataloaders[phase], total=int(len(nn_model_ref.dataloaders[phase])))
    for i, data in enumerate(tk0):
        inputs, labels = data
        coefall = 0
        for iter_filenames, filenames in enumerate(glob.glob(mypath)):
            coef = all_models_trained["coef"][iter_filenames]
            nn_model_ref.net = all_models_trained[filenames].eval()
            outputs = nn_model_ref.net(inputs.to(nn_model_ref.device))
            if methode1 :
                if iter_filenames==0:
                    outputs_f = coef* outputs
                else:
                    outputs_f += coef * outputs
                coefall +=coef


            #if phase == "train":
            #    data_train1[i*nn_model_ref.batch_size:(i+1)*nn_model_ref.batch_size,iter_filenames] = outputs.squeeze(1).detach().cpu().numpy()
            #else:
            #    data_val1[i*nn_model_ref.batch_size:(i+1)*nn_model_ref.batch_size,iter_filenames] = outputs.squeeze(1).detach().cpu().numpy()


            data_ici = nn_model_ref.net.intermediare2.detach().cpu().numpy()
            if phase == "train":
                data_train[i*nn_model_ref.batch_size:(i+1)*nn_model_ref.batch_size,64*iter_filenames:64*iter_filenames+64] = data_ici
            else:
                data_val[i*nn_model_ref.batch_size:(i+1)*nn_model_ref.batch_size,64*iter_filenames:64*iter_filenames+64] = data_ici


            del data_ici

            if methode1:
                outputs = outputs_f/(coefall)
            #print(outputs_f)




        #data_ici = nn_model_ref.net.intermediare.detach().cpu().numpy().astype(np.uint8)
        #if phase == "train":
        #    data_train[i*nn_model_ref.batch_size:(i+1)*nn_model_ref.batch_size,:] = data_ici
        #else:
        #    data_val[i*nn_model_ref.batch_size:(i+1)*nn_model_ref.batch_size,:] = data_ici
        #del data_ici

        #nn_model_ref.intermediaires[phase].append(nn_model_ref.net.intermediare.detach().cpu().numpy().astype(np.uint8))
        #nn_model_ref.outputs_proba[phase].append(outputs.detach().cpu().numpy().astype(np.float16))

        loss = nn_model_ref.criterion(outputs.squeeze(1), labels.to(nn_model_ref.device))
        desc = 'loss: %.4f; ' % (loss.item())
        preds = (outputs.squeeze(1) > nn_model_ref.t.to(nn_model_ref.device)).float().cpu() * 1
        #nn_model_ref.outputs_pred[phase].append(preds.detach().cpu().numpy().astype(np.float16))
        TP += (preds.eq(1) & labels.eq(1)).cpu().sum()
        TN += (preds.eq(0) & labels.eq(0)).cpu().sum()
        FN += (preds.eq(0) & labels.eq(1)).cpu().sum()
        FP += (preds.eq(1) & labels.eq(0)).cpu().sum()
        TOT = TP + TN + FN + FP
        desc += 'acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (
            (TP.item() + TN.item()) * 1.0 / TOT.item(), TP.item() * 1.0 / TOT.item(),
            TN.item() * 1.0 / TOT.item(), FN.item() * 1.0 / TOT.item(),
            FP.item() * 1.0 / TOT.item())
        running_loss += loss.item() * n_batches
        nbre_sample += n_batches

        #print(ok)





    epoch_loss = running_loss / nbre_sample
    acc = (TP.item() + TN.item()) * 1.0 / TOT.item()
    nn_model_ref.acc = acc
    print('{} Loss: {:.4f}'.format(
        phase, epoch_loss))
    print('{} Acc: {:.4f}'.format(
        phase, acc))
    #print(desc)
    print()
    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print()



from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit



#n_samples = data_train1.shape[0]
#cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
#clf = svm.SVC(kernel='linear', C=1)
#clf = RandomForestClassifier(max_depth=10, random_state=0)
#scores = cross_val_score(clf, data_train1, nn_model_ref.Y_train_nn_binaire, cv=cv)
#print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))


"""net2, h = train_speck_distinguisher(args,data_train1.shape[1], data_train1,
                                            nn_model_ref.Y_train_nn_binaire, data_val1, nn_model_ref.Y_val_nn_binaire,
                                            bs=args.batch_size_our,
                                            epoch=args.num_epch_our, name_ici="our_model",
                                            wdir=nn_model_ref.path_save_model, flag_3layes=False)"""

net2, h = train_speck_distinguisher(args,data_train.shape[1], data_train,
                                            nn_model_ref.Y_train_nn_binaire, data_val, nn_model_ref.Y_val_nn_binaire,
                                            bs=args.batch_size_our,
                                            epoch=args.num_epch_our, name_ici="our_model",
                                            wdir=nn_model_ref.path_save_model, flag_3layes=False)
print(ok)


