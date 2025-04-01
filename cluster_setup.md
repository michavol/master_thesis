# Setupt
VPN Host: sslvpn.ethz.ch/math-guest
VPN User: michavol@math-guest.ethz.ch
Sever: ssh michavol@ada-27.math.ethz.ch (also host for vscode remote)
ada-23 or ada-27 (better GPUs)
module load CUDA/12.3-local
module load modules/Python3/1.1

python -m venv .venv
pip install -r requirements.txt
source .venv/bin/activate

# Run Jobs
nice -n 7 <command line> #for running python scripts
nvidia-smi  #for checking gpu load
tmux        #for being able to close the connection
quota -v    #for checking memory consumption
FileZilla   #for data transfer

# File Zilla
Host: sftp://sftpmath.math.ethz.ch
User: michavol
Password: ...