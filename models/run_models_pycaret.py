import pandas as pd 
import numpy as np 
from pathlib import Path
from pycaret.regression import *
import argparse

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_class", type=str, default="9chrome")
    args = parser.parse_args()
    data = np.load(Path().resolve().parents[0] / f"data/{args.data_class}_data.npy", 
    allow_pickle=True)[()]
    df = pd.DataFrame(data['X'].astype('float64'), columns=data['features'])
    df['CT_RT'] = data['y'].astype('float64')
    del data
    
    exp = setup(data = df, 
                target = 'CT_RT', 
                session_id=123,
                normalize = True, 
                transformation = True, 
                transform_target = True, 
                combine_rare_levels = True, 
                rare_level_threshold = 0.05,
                remove_multicollinearity = True, 
                multicollinearity_threshold = 0.95, 
                #train_size=0.8,
                log_experiment = True, 
                fold=5,
                experiment_name = args.data_class)
    
    compare_models()
    ct = create_model('catboost', fold = 5)
    tuned_ct = tune_model(ct)
    predict_model(tuned_ct)

if __name__ == '__main__':
    main()