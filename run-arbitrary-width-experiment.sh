#!/bin/bash
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

echo "Experiment config: $1"
echo "Number of trials: $2"

eval $(parse_yaml $1)
echo "Device: $device"
echo "Epochs: $epochs"
echo "Log directory: $log_dir"
echo "Loss function: $loss_func"
echo "Initialization: $init"
echo "Activation: $activation"
echo "Layer type: $layer_type"
echo "width: $width"
echo "bias: $bias"
echo "dataset size: $dataset_size"

python train-simple.py --epochs $epochs \
--device $device \
--log-dir $log_dir \
--init $init \
--loss-func $loss_func \
--activation $activation \
--layer-type $layer_type \
--width $width \
--bias $bias \
--dataset-size $dataset_size \
--num-trials $2