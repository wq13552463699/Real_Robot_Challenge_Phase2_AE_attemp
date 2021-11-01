from subprocess import run

cmd = 'mpirun -np 8 python -u train.py 2>&1 | tee temp.log'

run(cmd, capture_output=True)