import os

#'[ctdata0l, ctdata0r, ctdata1l, ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l]'
#["baseline", "deepset", "cnn_attention", "multihead"]

"""
command = "python3 main_3class.py --nombre_round_eval 5 --type_model baseline_3class --make_data_equilibre_3class No"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 6 --type_model baseline_3class --make_data_equilibre_3class No"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 5 --type_model baseline_3class"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 6 --type_model baseline_3class"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 5 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_3class --make_data_equilibre_3class No"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 6 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_3class --make_data_equilibre_3class No"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 5 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_3class"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 6 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_3class"
os.system(command)




command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 5"
os.system(command)
command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 6"
os.system(command)
#command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 7"
#os.system(command)

command = "python3 main.py --nombre_round_eval 5"
os.system(command)
command = "python3 main.py --nombre_round_eval 6"
os.system(command)
#command = "python3 main.py --nombre_round_eval 7"
#os.system(command)

command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_bin_v2 --nombre_round_eval 5"
os.system(command)
command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_bin_v2 --nombre_round_eval 6"
os.system(command)
#command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_bin_v2 --nombre_round_eval 7"
#os.system(command)

command = "python3 main.py --type_model baseline_bin_v2 --nombre_round_eval 5"
os.system(command)
command = "python3 main.py --type_model baseline_bin_v2 --nombre_round_eval 6"
os.system(command)
#command = "python3 main.py --type_model baseline_bin_v2 --nombre_round_eval 7"
#os.system(command)

"""

"""for r in [5, 6, 7]:
    #command = "python3 main.py --nombre_round_eval "+str(r)
    #os.system(command)
    for N in [50, 100]:
        Ntrain = 10**7
        Ntrain2 = int(Ntrain/N)
        Nval2 = int(Ntrain/(10*N))
        command = "python3 main_N_batch.py --Nbatch "+str(N) + " --nombre_round_eval "+str(r+1) + " --nbre_sample_train " + str(Ntrain2)+ " --nbre_sample_eval " + str(Nval2)
        os.system(command)"""


"""for r in [5, 4, 3, 2]:
    for lks in [[1,9], [2,12], [3,15]]:
        command = "python3 main.py --numLayers "+str(r) + " --limit "+str(lks[0]) + " --kstime "+str(lks[1])
        os.system(command)"""


#self.diff = (0x8100, 0x8102) #98.4 sur 3 round
#self.diff = (0x8300, 0x8302) #XXX sur 3 round
#self.diff = (0x8700, 0x8702) #91.7 sur 3 round
#self.diff = (0x8f00, 0x8f02) #XXX sur 3 round
#self.diff = (0x9f00, 0x9f02) #XXX sur 3 round
#self.diff = (0xbf00, 0xbf02) #XXX sur 3 round
#self.diff = (0xff00, 0xff02) #XXX sur 3 round
#self.diff = (0x7f00, 0x7f02) #XXX sur 3 round

"""for r in [3,4]:
    for diff in ["python3 main.py --diff '(0x8100, 0x8102)'", "python3 main.py --diff '(0x8300, 0x8302)'",
                 "python3 main.py --diff '(0x8700, 0x8702)'", "python3 main.py --diff '(0x8f00, 0x8f02)'",
                 "python3 main.py --diff '(0x9f00, 0x9f02)'", "python3 main.py --diff '(0xbf00, 0xbf02)'",
                 "python3 main.py --diff '(0xff00, 0xff02)'", "python3 main.py --diff '(0x7f00, 0x7f02)'"]:
        print(diff + " --nombre_round_eval "+str(r))
        os.system(diff+ " --nombre_round_eval "+str(r))"""
"""
pathini= "/home/adriben/PycharmProjects/Refactoring_le_NN_avance_masque/results/masks_analyse/masks_all_THOMASTEST_todo/"
for filetodo in ["masks_all_THOMASTEST_1.txt", "masks_all_THOMASTEST_2.txt", "masks_all_THOMASTEST_3.txt",
                 "masks_all_THOMASTEST_12.txt", "masks_all_THOMASTEST_23.txt", "masks_all_THOMASTEST_123.txt"]:
    path_file = pathini+filetodo
    command = "python3 main.py --file_mask "+path_file
    os.system(command)"""


for round in ["7", "8"]:
    if round!="7":
        command = "python3 main_recovery_key.py --nombre_round_eval "+round
        os.system(command)
    for N in [10, 50, 100]:
        Ntrain = 10 ** 7
        Ntrain2 = int(Ntrain / N)
        Nval2 = int(Ntrain / (10 * N))
        command = "python3 main_N_batch2.py --Nbatch " + str(N) + " --nombre_round_eval " + str(
            round) + " --nbre_sample_train " + str(Ntrain2) + " --nbre_sample_eval " + str(Nval2)
        os.system(command)






"""pathini= "./results/masks_analyse/"
for filetodo in ["masks_all_THOMASTEST_ffff_1.txt", "masks_all_THOMASTEST_ffff_2.txt", "masks_all_THOMASTEST_ffff_12.txt",
                 "masks_all_THOMASTEST_ffff_3.txt", "masks_all_THOMASTEST_ffff_23.txt", "masks_all_THOMASTEST_ffff_123.txt"]:
    path_file = pathini+filetodo
    command = "python3 main.py --file_mask "+path_file
    os.system(command)"""