#!/bin/bash

LOG_PREFIX="Hilbert_Curve_MSHilbNet"
### MSHILB model selection ###
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 21 23 28 29 30 31 32 33 34 35
do
	TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
	SUBJECT="1"
	for SUBJECT in 2 21 6 8 5 10 18 16 13 14
	do
		python3 run_experiment_hilbert.py --subject $SUBJECT --model "MSHILB" \
		--timestamp $TIMESTAMP --log "${LOG_PREFIX}_Model_$i" \
		--include_rest_gesture --img_height 8 --img_width 8 --img_depth 10 \
		--window_size 64 --window_step 1 \
		--augment_factor 1 \
		--augment_jitter 25 --augment_mwrp 0.2 \
		--hilbert_type "time" \
		--model_params "mshilb_params_$i.json" --dropout 0.3 \
		--epochs 30 --batch_size 1024 \
		--validation
		SUBJECT=$[$SUBJECT+1]
	done
done


BEST_CONFIG=1
## MSHILB Exps window 16 ###
TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
SUBJECT="1"
while [ $SUBJECT -lt 28 ]
do
	python3 run_experiment_hilbert.py --subject $SUBJECT --model "MSHILB" \
	--timestamp $TIMESTAMP --log "${LOG_PREFIX}_${BEST_CONFIG}" \
	--include_rest_gesture --img_height 4 --img_width 4 --img_depth 10 \
	--window_size 16 --window_step 1 \
	--augment_factor 1 \
	--augment_jitter 25 --augment_mwrp 0.2 \
	--hilbert_type "time" \
	--model_params "mshilb_params_${BEST_CONFIG}.json" --dropout 0.3 \
	--epochs 60 --batch_size 1024
	SUBJECT=$[$SUBJECT+1]
done

TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
SUBJECT="1"
while [ $SUBJECT -lt 28 ]
do
	python3 run_experiment_hilbert.py --subject $SUBJECT --model "MSHILB" \
	--timestamp $TIMESTAMP --log "${LOG_PREFIX}_${BEST_CONFIG}" \
	--include_rest_gesture --img_height 4 --img_width 4 --img_depth 16 \
	--window_size 16 --window_step 1 \
	--augment_factor 1 \
	--augment_jitter 25 --augment_mwrp 0.2 \
	--hilbert_type "electrodes" \
	--model_params "mshilb_params_${BEST_CONFIG}.json" --dropout 0.3 \
	--epochs 30 --batch_size 1024
	SUBJECT=$[$SUBJECT+1]
done

# MSHILB Exps window 32 ###
TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
SUBJECT="1"
while [ $SUBJECT -lt 28 ]
do
	python3 run_experiment_hilbert.py --subject $SUBJECT --model "MSHILB" \
	--timestamp $TIMESTAMP --log "${LOG_PREFIX}_${BEST_CONFIG}" \
	--include_rest_gesture --img_height 8 --img_width 8 --img_depth 10 \
	--window_size 32 --window_step 1 \
	--augment_factor 1 \
	--augment_jitter 25 --augment_mwrp 0.2 \
	--hilbert_type "time" \
	--model_params "mshilb_params_${BEST_CONFIG}.json" --dropout 0.3 \
	--epochs 60 --batch_size 1024
	SUBJECT=$[$SUBJECT+1]
done


## MSHILB Exps window 64 ###
TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
SUBJECT="1"
while [ $SUBJECT -lt 28 ]
do
	python3 run_experiment_hilbert.py --subject $SUBJECT --model "MSHILB" \
	--timestamp $TIMESTAMP --log "${LOG_PREFIX}_${BEST_CONFIG}" \
	--include_rest_gesture --img_height 4 --img_width 4 --img_depth 64 \
	--window_size 64 --window_step 1 \
	--augment_factor 1 \
	--augment_jitter 25 --augment_mwrp 0.2 \
	--hilbert_type "electrodes" \
	--model_params "mshilb_params_${BEST_CONFIG}.json" --dropout 0.3 \
	--epochs 30 --batch_size 1024
	SUBJECT=$[$SUBJECT+1]
done
