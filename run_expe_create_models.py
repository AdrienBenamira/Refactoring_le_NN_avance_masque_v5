import os

#'[ctdata0l, ctdata0r, ctdata1l, ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l]'
#["baseline", "deepset", "cnn_attention", "multihead"]


for round in ([5]):
    command1 = "python3 main.py --nombre_round_eval " + str(round) + " --inputs_type '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]' --device 2 --a_bit 1"
    command2 = "python3 main.py --nombre_round_eval " + str(round) + " --inputs_type '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]' --device 0 --a_bit 2"
    os.system(command1 + " & " + command2)
    command1 = "python3 main.py --nombre_round_eval " + str(
        round) + " --inputs_type '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]' --device 2 --a_bit 3"
    command2 = "python3 main.py --nombre_round_eval " + str(
        round) + " --inputs_type '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]' --device 0 --a_bit 4"
    os.system(command1 + " & " + command2)

for round in ([5]):
    command1 = "python3 main.py --nombre_round_eval " + str(round) + " --inputs_type '[ctdata0l^ctdata1l, inv(DeltaV), inv(V0)&inv(V1), inv(V0)&V1]' --device 2 --a_bit 1"
    command2 = "python3 main.py --nombre_round_eval " + str(round) + " --inputs_type '[ctdata0l^ctdata1l, inv(DeltaV), inv(V0)&inv(V1), inv(V0)&V1]' --device 0 --a_bit 2"
    os.system(command1 + " & " + command2)
    command1 = "python3 main.py --nombre_round_eval " + str(
        round) + " --inputs_type '[ctdata0l^ctdata1l, inv(DeltaV), inv(V0)&inv(V1), inv(V0)&V1]' --device 2 --a_bit 3"
    command2 = "python3 main.py --nombre_round_eval " + str(
        round) + " --inputs_type '[ctdata0l^ctdata1l, inv(DeltaV), inv(V0)&inv(V1), inv(V0)&V1]' --device 0 --a_bit 4"
    os.system(command1 + " & " + command2)

for round in ([5, 6]):
    command1 = "python3 main.py --nombre_round_eval " + str(round) + " --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --device 2 --a_bit 1"
    command2 = "python3 main.py --nombre_round_eval " + str(round) + " --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --device 0 --a_bit 2"
    os.system(command1 + " & " + command2)
    command1 = "python3 main.py --nombre_round_eval " + str(round) + " --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --device 2 --a_bit 3"
    command2 = "python3 main.py --nombre_round_eval " + str(round) + " --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --device 0 --a_bit 4"
    os.system(command1 + " & " + command2)


