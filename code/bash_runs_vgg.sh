#!/bin/bash

LOG_PREFIX="Hilbert_Curve_VGGNet"
# ### VGG Exps window 16 ###
for i in {0..5..1}
do
	TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
	SUBJECT="1"
	for SUBJECT in 2 21 6 8 5 10 18 16 13 14
	do
		python3 run_experiment_hilbert.py --subject $SUBJECT --model "VGG" \
		--timestamp $TIMESTAMP --log "${LOG_PREFIX}_Model_$i" \
		--include_rest_gesture --img_height 16 --img_width 10 --img_depth 1 \
		--window_size 16 --window_step 1 \
		--augment_factor 1 \
		--augment_jitter 25 --augment_mwrp 0.2 \
		--hilbert_type "none" \
		--model_params "vgg_params_$i.json" --dropout 0.1 \
		--epochs 30 --batch_size 1024 \
		--validation
		SUBJECT=$[$SUBJECT+1]
	done
done

BEST_CONFIG=2
TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
SUBJECT="1"
while [ $SUBJECT -lt 28 ]
do
	python3 run_experiment_hilbert.py --subject $SUBJECT --model "VGG" \
	--timestamp $TIMESTAMP --log "${LOG_PREFIX}_${BEST_CONFIG}" \
	--include_rest_gesture --img_height 16 --img_width 10 --img_depth 1 \
	--window_size 16 --window_step 1 \
	--augment_factor 1 \
	--augment_jitter 25 --augment_mwrp 0.2 \
	--hilbert_type "none" \
	--model_params "vgg_params_${BEST_CONFIG}.json" --dropout 0.1 \
	--epochs 60 --batch_size 1024
	SUBJECT=$[$SUBJECT+1]
done

TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
SUBJECT="1"
while [ $SUBJECT -lt 28 ]
do
	python3 run_experiment_hilbert.py --subject $SUBJECT --model "VGG" \
	--timestamp $TIMESTAMP --log "${LOG_PREFIX}_${BEST_CONFIG}" \
	--include_rest_gesture --img_height 4 --img_width 4 --img_depth 10 \
	--window_size 16 --window_step 1 \
	--augment_factor 1 \
	--augment_jitter 25 --augment_mwrp 0.2 \
	--hilbert_type "time" \
	--model_params "vgg_params_${BEST_CONFIG}.json" --dropout 0.1 \
	--epochs 60 --batch_size 1024
	SUBJECT=$[$SUBJECT+1]
done

TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
SUBJECT="1"
while [ $SUBJECT -lt 28 ]
do
	python3 run_experiment_hilbert.py --subject $SUBJECT --model "VGG" \
	--timestamp $TIMESTAMP --log "${LOG_PREFIX}_${BEST_CONFIG}" \
	--include_rest_gesture --img_height 4 --img_width 4 --img_depth 16 \
	--window_size 16 --window_step 1 \
	--augment_factor 1 \
	--augment_jitter 25 --augment_mwrp 0.2 \
	--hilbert_type "electrodes" \
	--model_params "vgg_params_${BEST_CONFIG}.json" --dropout 0.1 \
	--epochs 60 --batch_size 1024
	SUBJECT=$[$SUBJECT+1]
done


TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
SUBJECT="1"
while [ $SUBJECT -lt 28 ]
do
	python3 run_experiment_hilbert.py --subject $SUBJECT --model "VGG" \
	--timestamp $TIMESTAMP --log "${LOG_PREFIX}_${BEST_CONFIG}" \
	--include_rest_gesture --img_height 32 --img_width 10 --img_depth 1 \
	--window_size 32 --window_step 1 \
	--augment_factor 1 \
	--augment_jitter 25 --augment_mwrp 0.2 \
	--hilbert_type "none" \
	--model_params "vgg_params_${BEST_CONFIG}.json" --dropout 0.1 \
	--epochs 60 --batch_size 1024
	SUBJECT=$[$SUBJECT+1]
done

TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
SUBJECT="1"
while [ $SUBJECT -lt 28 ]
do
	python3 run_experiment_hilbert.py --subject $SUBJECT --model "VGG" \
	--timestamp $TIMESTAMP --log "${LOG_PREFIX}_${BEST_CONFIG}" \
	--include_rest_gesture --img_height 8 --img_width 8 --img_depth 10 \
	--window_size 32 --window_step 1 \
	--augment_factor 1 \
	--augment_jitter 25 --augment_mwrp 0.2 \
	--hilbert_type "time" \
	--model_params "vgg_params_${BEST_CONFIG}.json" --dropout 0.1 \
	--epochs 60 --batch_size 1024
	SUBJECT=$[$SUBJECT+1]
done

TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
SUBJECT="1"
while [ $SUBJECT -lt 28 ]
do
	python3 run_experiment_hilbert.py --subject $SUBJECT --model "VGG" \
	--timestamp $TIMESTAMP --log "${LOG_PREFIX}_${BEST_CONFIG}" \
	--include_rest_gesture --img_height 4 --img_width 4 --img_depth 32 \
	--window_size 32 --window_step 1 \
	--augment_factor 1 \
	--augment_jitter 25 --augment_mwrp 0.2 \
	--hilbert_type "electrodes" \
	--model_params "vgg_params_${BEST_CONFIG}.json" --dropout 0.1 \
	--epochs 60 --batch_size 1024
	SUBJECT=$[$SUBJECT+1]
done


TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
SUBJECT="1"
while [ $SUBJECT -lt 28 ]
do
	python3 run_experiment_hilbert.py --subject $SUBJECT --model "VGG" \
	--timestamp $TIMESTAMP --log "${LOG_PREFIX}_${BEST_CONFIG}" \
	--include_rest_gesture --img_height 64 --img_width 10 --img_depth 1 \
	--window_size 64 --window_step 1 \
	--augment_factor 1 \
	--augment_jitter 25 --augment_mwrp 0.2 \
	--hilbert_type "none" \
	--model_params "vgg_params_${BEST_CONFIG}.json" --dropout 0.1 \
	--epochs 60 --batch_size 1024
	SUBJECT=$[$SUBJECT+1]
done

TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
SUBJECT="1"
while [ $SUBJECT -lt 28 ]
do
	python3 run_experiment_hilbert.py --subject $SUBJECT --model "VGG" \
	--timestamp $TIMESTAMP --log "${LOG_PREFIX}_${BEST_CONFIG}" \
	--include_rest_gesture --img_height 8 --img_width 8 --img_depth 10 \
	--window_size 64 --window_step 1 \
	--augment_factor 1 \
	--augment_jitter 25 --augment_mwrp 0.2 \
	--hilbert_type "time" \
	--model_params "vgg_params_${BEST_CONFIG}.json" --dropout 0.1 \
	--epochs 60 --batch_size 1024
	SUBJECT=$[$SUBJECT+1]
done

TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
SUBJECT="1"
while [ $SUBJECT -lt 28 ]
do
	python3 run_experiment_hilbert.py --subject $SUBJECT --model "VGG" \
	--timestamp $TIMESTAMP --log "${LOG_PREFIX}_${BEST_CONFIG}" \
	--include_rest_gesture --img_height 4 --img_width 4 --img_depth 64 \
	--window_size 64 --window_step 1 \
	--augment_factor 1 \
	--augment_jitter 25 --augment_mwrp 0.2 \
	--hilbert_type "electrodes" \
	--model_params "vgg_params_${BEST_CONFIG}.json" --dropout 0.1 \
	--epochs 60 --batch_size 1024
	SUBJECT=$[$SUBJECT+1]
done
