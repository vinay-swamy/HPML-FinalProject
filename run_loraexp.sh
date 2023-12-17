#!/bin/bash 
/bin/rm lora_results.csv
touch lora_exp.csv
python lora_exp.py esm2_t6_8M_UR50D -1 120
python lora_exp.py esm2_t6_8M_UR50D 3 129 
python lora_exp_with_deepspeed.py esm2_t6_8M_UR50D 3 180 
python lora_exp.py esm2_t12_35M_UR50D -1 35  
python lora_exp.py esm2_t12_35M_UR50D 3 49 
python lora_exp_with_deepspeed.py esm2_t12_35M_UR50D 3 59
python lora_exp.py esm2_t30_150M_UR50D -1 3
python lora_exp.py esm2_t30_150M_UR50D 3 9
python lora_exp_with_deepspeed.py esm2_t30_150M_UR50D 3 12 

