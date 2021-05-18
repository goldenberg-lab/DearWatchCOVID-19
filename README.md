# DearWatchCOVID-19
Code to reproduce the results from ["Dear Watch, Should I Get a COVID-19 Test? Designing deployable machine learning for wearables"](https://www.medrxiv.org/content/10.1101/2021.05.11.21257052v1)


## Requirements

Python 3, PyTorch, XGBoost, NumPy, Scikit-Learn, Pandas, and Jupyter Notebooks.
Packages are installed using [Anaconda3](https://www.anaconda.com/)

CPU machine with >100 GB RAM

GPU and multicore CPU is an asset.

## Data Access

Data can be obtained through access procedures described in [prior work](https://doi.org/10.1016/j.patter.2020.100188).
Qualified researchers can request access at https://www.synapse.org/#!Synapse:syn22891469/wiki/605716


## Usage

First, convert the weekly surveys into day-level labels merged with the wearable data and baselines. Run the cells in `dataloader.ipynb` to get an HDF file with the preprocessed data. Preprocessing options like regular or irregular sampling can be selected in this file.

To run a randomly split, time agnostic model using:
```
python run_model.py --data_dir /datasets/evidationdata --output_dir ./tmp_dir --batch_size 32

```

To run the temporal models, please follow these steps:
```
# create a directory with a split of prospective train participants

python split_df_v_time.py --data_dir </path/to/data>' --output_dir </path/to/output> --target ili --regularly_sampled --less_space


# now run the models. See the following scripts for slurm examples 
./run_splits_xgboost.sh </path/to/output> ili woy noz standard regular imp 1 2019-10-01 notbound #this launches scripts to train all weeks' xgboost models, gru survey models, and evaluates both. Please see run_model.py --help for argument descriptions.

./run_splits_grud.sh </path/to/output> ili woy regular 1 
```

Many metrics in the text are acquired through functions found in `get_stats.py`:
Please contact the authors for Notebooks which recreate plots.



If you use this code please cite
```
@article {Nestor2021DearWatch,
    author = {Nestor, Bret and Hunter, Jaryd and Kainkaryam, Raghu and Drysdale, Erik and Inglis, Jeffrey B and Shapiro, Allison and Nagaraj, Sujay and Ghassemi, Marzyeh and Foschini, Luca and Goldenberg, Anna},
    title = {Dear Watch, Should I Get a COVID-19 Test? Designing deployable machine learning for wearables},
    elocation-id = {2021.05.11.21257052},
    year = {2021},
    doi = {10.1101/2021.05.11.21257052},
    publisher = {Cold Spring Harbor Laboratory Press},
    URL = {https://www.medrxiv.org/content/early/2021/05/17/2021.05.11.21257052},
    eprint = {https://www.medrxiv.org/content/early/2021/05/17/2021.05.11.21257052.full.pdf},
    journal = {medRxiv}
}
```

