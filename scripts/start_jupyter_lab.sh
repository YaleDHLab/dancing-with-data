#!/bin/bash
#SBATCH --partition interactive
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 6:00:00
#SBATCH --job-name jupyter-lab
#SBATCH --output logs/jupyter-lab-%J.log
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
source activate env
jupyter lab --no-browser --port=${port} --ip=${node}
