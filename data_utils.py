"""
Data loading and preprocessing utilities for FCG algorithm
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def readAndProcess(filepath, column_names, drop_names):
    """
    Read and preprocess data file
    """
    # Read the data
    df = pd.read_csv(filepath, names=column_names, skipinitialspace=True)
    
    # Drop specified columns
    df = df.drop(columns=drop_names)
    
    # Remove any rows with missing values
    df = df.replace('?', np.nan)
    df = df.dropna()
    
    # Encode categorical variables
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    
    return df


def loadAdult():
    """
    Load Adult dataset for training and testing
    """
    tr_path = 'ds/adult/adult.data'
    test_path = 'ds/adult/adult.test'
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'income']
    drop_names = ['education-num', 'fnlwgt', 'race', 'native-country']

    dftr = readAndProcess(tr_path, column_names, drop_names)
    dftst = readAndProcess(test_path, column_names, drop_names)
    return dftr, dftst


def convertAdultIncome(dfin):
    """
    Convert income labels to readable format
    """
    df = dfin.copy(deep=True)
    df.rename(columns={'income': 'income answer'}, inplace=True)
    df['income answer'] = df['income answer'].replace('<=50K', 'less than or equal to 50K')
    df['income answer'] = df['income answer'].replace('>50K', 'greater than 50K')
    return df


def create_subgroups(df, sensitive_feature='sex', label='income'):
    """
    Split data into 4 subgroups based on sensitive feature Z and label Y
    SG = {g1(Z=1,Y=0), g2(Z=1,Y=1), g3(Z=0,Y=0), g4(Z=0,Y=1)}
    
    Args:
        df: DataFrame with features and labels
        sensitive_feature: Name of sensitive attribute (default: 'sex')
        label: Name of target label (default: 'income')
    
    Returns:
        Dictionary of 4 subgroups
    """
    # Create binary versions if not already binary
    Z = df[sensitive_feature].values
    Y = df[label].values
    
    # Create 4 subgroups
    subgroups = {
        'g1': df[(df[sensitive_feature] == 1) & (df[label] == 0)].copy(),  # Z=1, Y=0
        'g2': df[(df[sensitive_feature] == 1) & (df[label] == 1)].copy(),  # Z=1, Y=1
        'g3': df[(df[sensitive_feature] == 0) & (df[label] == 0)].copy(),  # Z=0, Y=0
        'g4': df[(df[sensitive_feature] == 0) & (df[label] == 1)].copy(),  # Z=0, Y=1
    }
    
    return subgroups


def split_train_dev(df, dev_ratio=0.2, random_state=42):
    """
    Split training data into train and dev sets
    """
    from sklearn.model_selection import train_test_split
    
    train_df, dev_df = train_test_split(df, test_size=dev_ratio, random_state=random_state, stratify=df['income'])
    
    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True)


class Subgroup:
    """
    Class to represent a subgroup with associated samples and scores
    """
    def __init__(self, df, z_value, y_value, initial_score=0.05):
        self.df = df.reset_index(drop=True)
        self.z_value = z_value
        self.y_value = y_value
        self.indices = df.index.tolist()
        self.scores = np.ones(len(df)) * initial_score  # Initialize all scores to p
        self.evol_scores = []  # Store evolution scores
        
    def get_features(self, exclude_cols=['sex', 'income']):
        """Get feature matrix excluding sensitive and label columns"""
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        return self.df[feature_cols].values
    
    def get_samples(self):
        """Get all samples in the subgroup"""
        return self.df
    
    def update_scores(self, idx_list, evol_score):
        """Update scores for selected samples"""
        for idx in idx_list:
            # Average with existing score
            self.scores[idx] = (self.scores[idx] + evol_score) / 2
    
    def get_score(self, idx):
        """Get score for a specific sample"""
        return self.scores[idx]
    
    def __len__(self):
        return len(self.df)
