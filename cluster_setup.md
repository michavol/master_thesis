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
tmux        #for being able to close the connection
quota -v    #for checking memory consumption
FileZilla   #for data transfer
nvidia-smi  #for Checking gpu load
btop        #for Process monitoring

nvidia-smi && (nvidia-smi |tr -s ' '|grep -Eo "| [0123456789]+ N/A N/A [0-9]{3,} .*"|awk -F' ' '{system("s=$(cat /proc/"$4"/cmdline| tr \"\\0\" \" \");u=$(ps -o uname= -p "$4");echo "$1"sep"$4"sep$u sep"$7"sep$s" ) }'|sed 's/sep/\t/g') # Check username of GPU processes

# File Zilla
Host: sftp://sftpmath.math.ethz.ch
User: michavol
Password: ...