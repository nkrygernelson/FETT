import numpy as np 
import pandas as pd
import os
import time
import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def show_dist(df1, df2, df1_name='Test', df2_name='Train'):
    """
    TODO: make it work with any property not just bg
    Displays and compares the binned distributions of two dataframes.

    The x-axis shows the average 'BG' value for each bin, making it more interpretable.
    """
    # Create a figure with two rows of subplots.
    # sharex is False because the mean BG per bin can differ between dataframes.
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    
    # --- Plot for the first DataFrame (df1) ---
    # Calculate the mean 'BG' for each category bin to use as x-axis labels
    mean_labels1 = df1.groupby('bg_category')['BG'].mean()
    
    sns.countplot(data=df1, x='bg_category', ax=axes[0], color='skyblue')
    axes[0].set_title(f'Distribution for {df1_name} Data')
    axes[0].set_xlabel('') # Hide x-label on top plot for a cleaner look
    axes[0].set_xticklabels([f'{label:.1f}' for label in mean_labels1]) # Set new labels

    # Calculate and display overall statistics
    mean1 = df1['BG'].mean()
    std1 = df1['BG'].std()
    stats_text1 = f'Overall Mean: {mean1:.2f}\nOverall Std Dev: {std1:.2f}'
    axes[0].text(0.95, 0.95, stats_text1, transform=axes[0].transAxes,
                 fontsize=12, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # --- Plot for the second DataFrame (df2) ---
    # Calculate the mean 'BG' for each category bin
    mean_labels2 = df2.groupby('bg_category')['BG'].mean()

    sns.countplot(data=df2, x='bg_category', ax=axes[1], color='salmon')
    axes[1].set_title(f'Distribution for {df2_name} Data')
    axes[1].set_xlabel('Average BG Value in Bin') # A more descriptive label
    axes[1].set_xticklabels([f'{label:.1f}' for label in mean_labels2]) # Set new labels
    
    # Calculate and display overall statistics
    mean2 = df2['BG'].mean()
    std2 = df2['BG'].std()
    stats_text2 = f'Overall Mean: {mean2:.2f}\nOverall Std Dev: {std2:.2f}'
    axes[1].text(0.95, 0.95, stats_text2, transform=axes[1].transAxes,
                 fontsize=12, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # --- Final Adjustments ---
    plt.tight_layout()
    plt.show()



def prop_binning(df, num_bins = 4, prop_name = "BG"):
    """
    Creates stratified bins for a column, grouping all outliers into the highest bin.
    Uses IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        prop_name (str): The name of the column to bin.
    Returns:
        pd.DataFrame: The DataFrame with the new 'bg_category' column.
    """
  
    Q1 = df[prop_name].quantile(0.25)
    Q3 = df[prop_name].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    is_outlier = df[prop_name] > outlier_threshold
    df[f'{prop_name}_category'] = pd.cut(
        df.loc[~is_outlier, prop_name],
        bins=num_bins,  
        labels=False,
        duplicates='drop'
    )
    # 4. Assign all outliers (which are currently NaN) to the highest bin
    highest_bin = df[f'{prop_name}_category'].max()
    df[f'{prop_name}_category'] = df[f'{prop_name}_category'].fillna(highest_bin)
    return df


    
def prepare_datasets(split_arr = [0.6,0.2,0.2], subsample_dict=None, fidelity_map = None,  num_bins=8, prop_name = "BG",dataset_name = "standard"):
    """
    """
    train_split = split_arr[0]
    test_split = split_arr[1]
    val_split = split_arr[2]
    if not subsample_dict:
        subsample_dict =  {"pbe":1, "scan":1, "gllb-sc":1, "hse":1, "expt":1}
    if not fidelity_map:
        fidelity_map = {"pbe":0, "scan":1, "gllb-sc":2, "hse":3, "expt":4}

    

    # Process each fidelity dataset
    #the order matters, I put expt last
    #MAKE sure to specify the right fidelity map
    data = {}
    trains = {}
    tests = {}
    vals = {}
    general_path = os.path.join("data","runs", dataset_name)
    if not os.path.exists(general_path):
        os.makedirs(general_path)

    if not os.path.exists(os.path.join(general_path, "test")):
        os.makedirs(os.path.join(general_path, "test"))
    if not os.path.exists(os.path.join(general_path, "val")):
        os.makedirs(os.path.join(general_path, "val"))
    if not os.path.exists(os.path.join(general_path, "train")):
        os.makedirs(os.path.join(general_path, "train"))

    for fidelity_name, fidelity_id in fidelity_map.items():
        if subsample_dict[fidelity_name] == 0:
            continue
        print(f"Processing {fidelity_name} dataset...")
        # Construct path for loading data
        # Assumes input data is in 'data/train/' relative to save_prefix if GOOGLE_DRIVE is True
        # Or locally if GOOGLE_DRIVE is False
        data_file_path = os.path.join(
             'data',"mp19", f'{fidelity_name}.csv')

        df = pd.read_csv(data_file_path)

        df = df.drop_duplicates()
        df.dropna()
        df["fidelity"] = fidelity_id
        data[fidelity_name] = df
    
    combined_train_df = pd.DataFrame()
    combined_val_df = pd.DataFrame()
    for fidelity_name in data.keys():
        data[fidelity_name] = prop_binning(data[fidelity_name], num_bins = num_bins,prop_name=prop_name)
        train,test = train_test_split(data[fidelity_name],
            train_size=1-test_split,
            stratify=data[fidelity_name][f'{prop_name}_category'],
            random_state=42 
        )
        train, val = train,test = train_test_split(train,
            train_size=train_split/(train_split+val_split),
            stratify=train[f'{prop_name}_category'],
            random_state=42 
        )
        trains[fidelity_name] = train
        tests[fidelity_name] = test
        vals[fidelity_name] = val

        combined_val_df = pd.concat([combined_val_df, val.drop(columns = f"{prop_name}_category")])
        combined_train_df = pd.concat([combined_train_df, train.drop(columns = f"{prop_name}_category")])
        tests[fidelity_name].to_csv(os.path.join(general_path, "test",f"{fidelity_name}.csv"), index=False)
        trains[fidelity_name].to_csv(os.path.join(general_path, "train",f"{fidelity_name}.csv"), index= False)
        vals[fidelity_name].to_csv(os.path.join(general_path, "val",f"{fidelity_name}.csv"), index= False)

    combined_train_df.to_csv(os.path.join(general_path, "combined_train.csv",), index=False)
    combined_val_df.to_csv(os.path.join(general_path, "combined_val.csv",), index=False)
  
    return combined_train_df, combined_val_df, trains, vals, tests


def prepare_only_new_on_expt(split_arr = [0.6,0.2,0.2], num_bins=8, ):
    """
    The test expt bandgaps only have bandgaps 
    that have not been seen at lower fidelities
    """
    train_split = split_arr[0]
    test_split = split_arr[1]
    val_split = split_arr[2]
    subsample_dict =  {"pbe":0.1, "scan":1, "gllb-sc":1, "hse":1, "expt":1}
    dataset_name = "only_new_on_expt"

    # Process each fidelity dataset
    #the order matters, I put expt last
    #MAKE sure to specify the right fidelity map
    fidelity_map = [["pbe",0], ["scan",1], ["gllb-sc",2], ["hse",3], ["expt",4]]
    data = {}
    trains = {}
    tests = {}
    vals = {}
    #fidelity_map = {"pbe":0, "scan":1, "gllb-sc":2, "hse":3, "expt":4}
    general_path = os.path.join("data","runs", dataset_name)
    if not os.path.exists(general_path):
        os.makedirs(general_path)

    if not os.path.exists(os.path.join(general_path, "test")):
        os.makedirs(os.path.join(general_path, "test"))
    if not os.path.exists(os.path.join(general_path, "val")):
        os.makedirs(os.path.join(general_path, "val"))
    if not os.path.exists(os.path.join(general_path, "train")):
        os.makedirs(os.path.join(general_path, "train"))

    for fidelity_name, fidelity_id in fidelity_map:
        print(f"Processing {fidelity_name} dataset...")
        # Construct path for loading data
        # Assumes input data is in 'data/train/' relative to save_prefix if GOOGLE_DRIVE is True
        # Or locally if GOOGLE_DRIVE is False
        data_file_path = os.path.join(
             'data',"mp19", f'{fidelity_name}.csv')

        df = pd.read_csv(data_file_path)

        df = df.drop_duplicates()
        df.dropna()
        df["fidelity"] = fidelity_id
        data[fidelity_name] = df
    
    data["expt"] = prop_binning(data["expt"], num_bins = num_bins)

    expt_train, expt_test = train_test_split(data["expt"],
                train_size=1-test_split,
                stratify=data["expt"][f"bg_category"],
                random_state=42 
            )
    expt_train, expt_val = train_test_split(expt_train,
                train_size=train_split/(train_split+val_split),
                stratify=expt_train[f"bg_category"],
                random_state=42 
            )
    
    
    combined_train_df = pd.DataFrame()
    combined_val_df = pd.DataFrame()
    for fidelity_name in data.keys():
        
        if fidelity_name != "expt":
            data[fidelity_name] = data[fidelity_name][~data[fidelity_name]["formula"].isin(expt_test["formula"])]
            data[fidelity_name] = prop_binning(data[fidelity_name], num_bins = num_bins)
            train,test = train_test_split(data[fidelity_name],
                train_size=1-test_split,
                stratify=data[fidelity_name]['bg_category'],
                random_state=42 
            )
            train, val = train,test = train_test_split(train,
                train_size=train_split/(train_split+val_split),
                stratify=train['bg_category'],
                random_state=42 
            )
            trains[fidelity_name] = train
            tests[fidelity_name] = test
            vals[fidelity_name] = val

            combined_val_df = pd.concat([combined_val_df, val.drop(columns = "bg_category")])
            combined_train_df = pd.concat([combined_train_df, train.drop(columns = "bg_category")])

        else:
            combined_train_df = pd.concat([combined_train_df, expt_train.drop(columns = "bg_category")])
            combined_val_df = pd.concat([combined_val_df, val.drop(columns = "bg_category")])
            trains["expt"] = expt_train
            tests["expt"] = expt_test
            vals["expt"] = expt_val
        tests[fidelity_name].to_csv(os.path.join(general_path, "test",f"{fidelity_name}.csv"), index=False)
        trains[fidelity_name].to_csv(os.path.join(general_path, "train",f"{fidelity_name}.csv"), index= False)
        vals[fidelity_name].to_csv(os.path.join(general_path, "val",f"{fidelity_name}.csv"), index= False)

    combined_train_df.to_csv(os.path.join(general_path, "combined_train.csv",), index=False)
    combined_val_df.to_csv(os.path.join(general_path, "combined_val.csv",), index=False)
  
    return combined_train_df, combined_val_df, trains, vals, tests

    

subsample_dict={"pbe":1,"hse":1,"scan":1,"gllb-sc":1, "expt":0}
fidelity_map = {"pbe":0, "scan":0, "gllb-sc":0, "hse":0, "expt":0}
prepare_datasets(prop_name="BG",fidelity_map=fidelity_map,subsample_dict=subsample_dict, dataset_name="FE_standard" )
