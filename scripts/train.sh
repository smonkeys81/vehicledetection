# This script calls training functions of vehicle detection.

# arguments:
# 1. Config file.
# 2. Network model name.
# 3. Solverstate file to continue training (optional)
# $1 Model name

#!/usr/bin/env sh

# make upper case
input=${1^^}

if [[ $input == *"VGG"* ]]; then
  model=VGG16
elif [[ $input == *"RES"* ]]; then
  model=RES101
elif [[ $input == *"FPN"* ]]; then
  model=FPN
else
  echo "Model name is incorrect"
  return
fi

../tools/train_model  ../config/config.json \
                      ${model} \



# Make result directory only if the dir does not exist.
mkdir -p result
mkdir -p ../model/out/${model}

# For each result file, plot loss and move the to result directory.
for fname in loss_po_*.txt
do
  fbname=$(basename "$fname" .txt)
  fname_date="$fbname"_$(date +%m-%d-%Y).txt
  mv $fname $fname_date
  python plot_loss.py --file $fname_date
done

# Move all loss result files.
mv loss_po_* result/
