#!/bin/bash

# first argument is the directory containing all subfolders with data, don't have a terminating / in the directory.

echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
echo $7
echo $8

if [ "$1" == "" ]; then
    echo "Missing targeted directory, ex: /datasets/evidationdata/covid_split"
    exit 1
elif [ ! -d "$1" ]; then
    echo "First argument is not a valid directory"
    exit 1
fi

targets=("ili" "covid" "symptoms__fever__fever" "flu_covid")

if  [ "$2" == "" ]; then
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

z_choices=("z" "noz")

if [ "$4" == "" ]; then
    echo "Missing if using 'zscores' or not, must be 'z' or 'noz'."
    exit 1
elif [[ ! " ${z_choices[@]} " =~ " ${4} " ]]; then
    echo "Not a valid decision for if to use 'zscores', must be 'z' or 'noz'."
    exit 1
elif [ "$4" == "z" ]; then
    z="--zscore"
    z_suf="_z"
elif [ "$4" == "noz" ]; then
    z=""
    z_suf=""
fi

subset_choices=("standard" "hr" "step" "sleep")

if [ "$5" == "" ]; then
    echo "Missing the fifth argument, feature subset. Choose from standard, hr, step or sleep."
    exit 1
elif [[ ! " ${subset_choices[@]} " =~ " ${5} " ]]; then
    echo "Not a valid choice for feature subset. Choose from standard, hr, step or sleep."
    exit 1
elif [ "$5" == "standard" ]; then
    sub=""
    sub_suf=""
elif [ "$5" == "hr" ]; then
    sub="--feat_regex heart_rate"
    sub_suf="_hr"
elif [ "$5" == "step" ]; then
    sub="--feat_regex steps"
    sub_suf="_st"
elif [ "$5" == "sleep" ]; then
    sub="--feat_regex sleep"
    sub_suf="_sl"
fi

sampling_choices=("irregular" "regular")

if [ "$6" == "" ]; then
    echo "Missing sixth argument, sampling type. Choose from regular or irregular."
    exit 1
elif [[ ! " ${sampling_choices[@]} " =~ " ${6} " ]]; then
    echo "Not a valid choice for sampling type. Choose from irregular of regular."
    exit 1
elif [ "$6" == "irregular" ]; then
    samp=""
    samp_suf=""
elif [ "$6" == "regular" ]; then
    samp="--regularly_sampled"
    samp_suf="_reg"
fi

imputation_choice=("noimp" "imp")

if [ "$7" == "" ]; then
    echo "Missing seventh argument, imputation flag. Choose from noimp or imp."
    exit 1
elif [[ ! " ${imputation_choice[@]} " =~ " ${7} " ]]; then
    echo "Not a valid choice for imputation flag. Choose from noimp or imp."
    exit 1
elif [ "$7" == "noimp" ]; then
    imp="--no_imputation"
    imp_suf="noimp"
elif [ "$7" == "imp" ]; then
    imp=""
    imp_suf=""
fi

if [ "$8" == "" ]; then
    echo "Missing eigth argument, maximum missingness. must be float between 0 and 1."
    exit 1
# TODO: This condition still allows max_miss to be greater than 1 and less than 2, fix the regex.
elif [[ $8 =~ [01]\.?[0-9]* ]]; then
    miss="--max_miss ${8}"
    miss_suf="_mm${8}"
else
    echo "Eigth argument is not a valid float between 0 and 1."
    exit 1
fi

if [ "$9" == "" ];then
    echo "Missing ninth argument, train start date. must be a valid date following ISO8601 or None."
    exit 1
elif [[ $9 =~ \d{4}-\d{2}-\d{2} ]]; then
    start_train="--train_start_date ${9}"
    start_train_suf="_tsd${9}"
elif [ "$9" == "None" ]; then
    start_train=""
    start_train_suf=""
else
    echo "Ninth argument is not a valid ISO8601 date or the string None."
    exit 1
fi

bounded_choice=("bound" "notbound" "fitbit")

if [ "${10}" == "" ]; then
    echo "Missing tenth argument, bounded flag. Choose from bound or nobound."
    exit 1
elif [[ ! " ${bounded_choice[@]} " =~ " ${10} " ]]; then
    echo "Not a valid choice for imputation flag. Choose from noimp or imp."
    exit 1
elif [ "${10}" == "bound" ]; then
    bound="--positive_labels 0 1 2 --mask_labels -6 -5 -4 -3 -2 -1 --bound_labels -21 2"
    bound_suf="_bounded"
elif [ "${10}" == "notbound" ]; then
    bound=""
    bound_suf=""
elif [ "${10}" == "fitbit" ]; then
    bound="--positive_labels 1 2 3 4 5 6 7 --mask_labels -7 -6 -5 -4 -3 -2 -1 0 --bound_labels -21 7"
    bound_suf="_fitbit"
fi

if [ -z "${11}" ] ; then
    echo "running without resampling"
    extra_args=""
elif [ "${11}"=="resample" ];then
    extra_args="--${11}"
fi

# Check for flag combinations which aren't implemented.
if [ "$imp" == "--no_imputation" ] && [ "$z" == "--zscore" ]; then
    echo "The combination of flags --no_imputation and --zscore has not been implemented."
    exit 1
fi

data_dir=$1
target=$2

full_suf="${woy_suf}${z_suf}${sub_suf}${samp_suf}${imp_suf}${miss_suf}${start_train_suf}${bound_suf}"
echo ${full_suf}

printf "Last_day, AUC, avg_precision_score \n" > "${data_dir}/all_xgb_auc${full_suf}.csv"
train_dirs=()
directories=()
jobids=()

for D in $data_dir/*; do
    if [ -d "${D}" ]; then
	train_dirs+=("${D}/out_xgb${full_suf}")
        echo $D
	val_start=$(date -I -d "$(basename $D | tr '_' '-') - 7 day")
	train_end=$(date -I -d "$(basename $D | tr '_' '-') - 8 day")
	test_start=$(date -I -d "$(basename $D | tr '_' '-')")
	test_end=$(date -I -d "$(basename $D | tr '_' '-') + 6 day")
	echo $(date -I -d "$(basename $D | tr '_' '-') - 1 day")
	echo $val_start
    	job_file="${D}/xgb_split${full_suf}.job"
    	echo "#!/bin/bash
#SBATCH --job-name=xgb_split
#SBATCH --output=${D}/out_xgb${full_suf}.txt
#SBATCH --error=${D}/out_xgb${full_suf}.err
#SBATCH --mem=100G
#SBATCH -c 8
#SBATCH -p <your cluster>
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

python run_model.py --target ${target} --modeltype xgboost --output_dir ${D}/out_xgb${full_suf} --data_dir ${D} ${z} ${woy} ${sub} ${samp} ${imp} ${miss} ${extra_args} ${start_train} --xgb_method gpu_hist ${bound} --train_end_date ${train_end} --validation_start_date ${val_start} --validation_set_len 7 --checkpt_dir /checkpoint/${USER}/\${SLURM_JOB_ID}

python eval_split_xgboost.py --target ${target} --home_dir ${D} ${woy} ${z} ${samp} ${sub} ${imp} --out_suf '${full_suf}'


python run_survey_model.py --target covid --modeltype gru --output_dir ${D}/out_gru_survey${woy_suf} --data_dir <path/to/data> --home_dir ${D} --batch_size 8 --num_dataloader_workers 7 --opt_level O1 --epochs 200 --reload ${woy} --feat_subset --override_splits

python eval_survey_gru_prospective.py --target ${target} --home_dir ${D} --data_dir ${data_dir} --opt_level O0 --train_end_date ${train_end} --validation_start_date ${val_start} --validation_set_len 7 --num_dataloader_workers 2 --test_start_date ${test_start} --test_end_date ${test_end} $extra_args ${woy} --feat_subset
" > $job_file

	# Need ids to set up dependency for evaluate call
	jobid=$(sbatch --parsable ${job_file})
	jobids+=":${jobid}"
    fi
done

#job_file="${data_dir}/evaluate_xgb${full_suf}.job"

#echo "#!/bin/bash
#SBATCH --job-name=train_aucs
#SBATCH --output=${data_dir}/out_xgb${full_suf}.txt
#SBATCH --err=${data_dir}/out_xgb${full_suf}.err
#SBATCH --mem=20G
#SBATCH -q nopreemption
#SBATCH -p cpu

#python evaluate.py --dirs "${train_dirs[@]/#}" --output_path ${data_dir}/train_results_xgb${full_suf}.csv --file_pattern *_testset_results.csv" > $job_file

echo ${jobids[@]}

#sbatch --dependency=afterany${jobids[@]} $job_file

