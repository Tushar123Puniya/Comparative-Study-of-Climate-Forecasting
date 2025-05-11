#!/bin/bash

# Set the target screen name
TARGET_SCREEN_NAME="ram"

# Paths to scripts
SCRIPT_PATHS=(
    # "/home/anonymus/Weather Prediction/Code/Baselines/Resnet.py"
    # "/home/anonymus/Weather Prediction/Code/Baselines/CL_Unet.py"
    # "/home/anonymus/Weather Prediction/Code/Baselines/ViT.py"
    "/home/anonymus/Weather Prediction/Code/Main Model/Unet.py"
    # "/home/anonymus/Weather Prediction/Code/Main Model/Unet_MLP.py"
    # "/home/anonymus/Weather Prediction/Code/Main Model/Unet_sattn.py"
    # "/home/anonymus/Weather Prediction/Code/Main Model/LSTM.py"
    # "/home/anonymus/Weather Prediction/Code/Main Model/climatology.py"
    # "/home/anonymus/Weather Prediction/Code/Main Model/persistence.py"
    # "/home/anonymus/Weather Prediction/Code/Main Model/XGBOOST.py"
)

# Variables to predict
pred_vars=("z_500" "t2m" "t_850")  # example predicted variables

# Parameters
out_steps_list=(1 3 5 7)
gap_list=(1)

# Send each command to the screen session
for script_path in "${SCRIPT_PATHS[@]}"; do
    for out_steps in "${out_steps_list[@]}"; do
        for gap in "${gap_list[@]}"; do 
            for pred_var in "${pred_vars[@]}"; do
                command="python \"$script_path\" --out_steps $out_steps --gap $gap --pred_var $pred_var"
                echo "Sending to screen [$TARGET_SCREEN_NAME]: $command"
                screen -S "$TARGET_SCREEN_NAME" -X stuff "$command$(echo -ne '\r')"
            done
        done
    done
done
