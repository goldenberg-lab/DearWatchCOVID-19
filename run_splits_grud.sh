#!/bin/bash

# first argument is the directory containing all subfolders with data, don't have a terminating / in the directory.

echo $1
echo $2
echo $3
echo $4
echo $5
echo $6

if [ "$1" == "" ]; then
    echo "Missing targeted directory, ex: /datasets/evidationdata/data_split"
    exit 1
elif [ ! -d "$1" ]; then
    echo "First argument is not a valid directory"
    exit 1
fi

targets=("ili" "covid" "symptoms__fever__fever")

if [ "$2" == "" ]; then
    echo "Missing target label"
    exit 1
elif [[ ! " ${targets[@]} " =~ " ${2} " ]]; then
    echo "Given target not in valid target array"
    exit 1
fi

woy_choices=("woy" "nowoy")

if [ "$3" == "" ]; then
    echo "Missing if using 'Week of Year' or not, must be 'woy' or 'nowoy'."
    exit 1
elif [[ ! " ${woy_choices[@]} " =~ " ${3} " ]]; then
    echo "Not a valid decision for if to use 'Week of Year', must be 'woy' or 'nowoy'."
    exit 1
elif [ "$3" == "woy" ]; then
    woy="--weekofyear"
    woy_suf="_WOY"
elif [ "$3" == "nowoy" ]; then
    woy=""
    woy_suf=""
fi

sampling_choices=("irregular" "regular")

if [ "$4" == "" ]; then
    echo "Missing fourth argument, sampling type. Choose from regular or irregular."
    exit 1
elif [[ ! " ${sampling_choices[@]} " =~ " ${4} " ]]; then
    echo "Not a valid choice for sampling type. Choose from irregular of regular."
    exit 1
elif [ "$4" == "irregular" ]; then
    samp=""
    samp_suf=""
elif [ "$4" == "regular" ]; then
    samp="--regularly_sampled"
    samp_suf="_reg"
fi

if [ "$5" == "" ]; then
    echo "Missing fifth argument, maximum missingness. must be float between 0 and 1."
    exit 1
# TODO: This condition still allows max_miss to be greater than 1 and less than 2, fix the regex.
elif [[ $5 =~ [01]\.?[0-9]* ]]; then
    miss="--max_miss ${5}"
    miss_suf="_mm${5}"
else
    echo "Fifth argument is not a valid float between 0 and 1."
    exit 1
fi

if [ -z "$6" ] ; then
    echo "running without resampling"
    extra_args=""
elif [ "$6"=="resample" ];then
    extra_args="--$6"
fi

data_dir=$1
target=$2

full_suf="${woy_suf}${samp_suf}${miss_suf}"

printf "Last_day, AUC, avg_precision_score \n" > "${data_dir}/all_grud_auc${full_suf}.csv"
train_dirs=()
directories=()
jobids=()


for D in $data_dir/*; do
    if [ -d "${D}" ]; then
    #&& grep -q 'ModuleNotFoundError' ${D}/out_grud_WOY_mm1.err; then
	train_dirs+=("${D}/out_grud${full_suf}")
	val_start=$(date -I -d "$(basename $D | tr '_' '-') - 7 day")
	train_end=$(date -I -d "$(basename $D | tr '_' '-') - 8 day")
	echo $(date -I -d "$(basename $D | tr '_' '-') - 1 day")
	echo $val_start
	job_file="${D}/grud_split${full_suf}.job"
    	echo "#!/bin/bash
#SBATCH --job-name=$(basename "${D}")
#SBATCH --output=${D}/out_grud${full_suf}.txt
#SBATCH --error=${D}/out_grud${full_suf}.err
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -p <your partition>
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

python run_model.py --target ${target} --modeltype grud --output_dir ${D}/out_grud${full_suf} --data_dir ${D} --batch_size 1 --num_dataloader_workers 7 --opt_level O0 --epochs 50 ${extra_args} --train_end_date ${train_end} --validation_start_date ${val_start} --validation_set_len 7 --reload --checkpt_dir /checkpoint/${USER}/\${SLURM_JOB_ID} ${woy} ${miss} ${samp}

# run_model.py without prospective validation set, used to get retrospective thresholds by copying the model from the prosp_val folder and just evaluating on the full validation set.
python run_model.py --target ${target} --modeltype grud --output_dir ${D}/out_grud${full_suf} --data_dir ${D} --batch_size 1 --num_dataloader_workers 7 --opt_level O0 --epochs 50 ${extra_args} --reload --checkpt_dir /checkpoint/${USER}/\${SLURM_JOB_ID} ${woy} ${miss} ${samp} --test

echo 'Next python run'

python eval_split_grud.py --target ${target} --home_dir ${D} --opt_level O0 ${woy} --out_suf ${full_suf} ${samp}
#python eval_split_grud.py --target ${target} --home_dir ${D} --opt_level O0 ${woy} --eval_train --out_suf ${full_suf}" > $job_file
	# Need ids to set up dependency for evaluate call
	jobid=$(sbatch --parsable $job_file)
	jobids+=":${jobid}"
    fi
done

#job_file="${data_dir}/evaluate_grud${full_suf}.job"

#echo "#!/bin/bash
#SBATCH --job-name=grud_aucs
#SBATCH --output=${data_dir}/out_grud${full_suf}.txt
#SBATCH --err=${data_dir}/out_grud${full_suf}.err
#SBATCH --mem=100G
#SBATCH --qos=nopreemption
#SBATCH -p cpu

#python evaluate.py --dirs "${train_dirs[@]/#}" --output_path ${data_dir}/test_results_grud${full_suf}.csv --file_pattern *_testset_results${full_suf}.csv

#python evaluate.py --dirs "${train_dirs[@]/#}" --output_path ${data_dir}/train_results_grud${full_suf}.csv --file_pattern *_trainset_results${full_suf}.csv" > $job_file

echo ${jobids[@]}

#sbatch --dependency=afterany${jobids[@]} $job_file

