import os


command = "python3 main.py --nombre_round_eval 5"
os.system(command)

command = "python3 main.py --nombre_round_eval 6"
os.system(command)

command = "python3 main.py --nombre_round_eval 5 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]'"
os.system(command)

command = "python3 main.py --nombre_round_eval 6 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]'"


command = "python3 main.py --inputs_type '[c0r^c1r, c0l^c1l, t0^t1]' --nombre_round_eval 8 --diff '(0, 0x0040)' --cipher simon"
os.system(command)

