#!/bin/bash
#SBATCH --partition interactive
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 6:00:00
#SBATCH --job-name jupyter-notebook
#SBATCH --output logs/jupyter-notebook-%J.log
#SBATCH -c 2

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@${cluster}.hpc.yale.edu
   
For more info and how to connect from windows, 
   see research.computing.yale.edu/jupyter-nb
Here is the MobaXterm info:

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${cluster}.hpc.yale.edu
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
# e.g. farnam:
# module load Python/2.7.13-foss-2016b 

# DON'T USE ADDRESS BELOW. 
# DO USE TOKEN BELOW

source activate env
jupyter-notebook --no-browser --port=${port} --ip=${node}