#!/usr/bin/env python
# coding: utf-8




def FeatureCategorizer(df):
    cat_cols = list(set(list(df.select_dtypes(["object"]).columns) + [col for col in df.select_dtypes(["float64", "int64"]).columns if df[col].nunique() == 2]))
    #<=3 unique value in numeric data types is selected based on the data 
    ord_cols = list(set(list(df.filter(regex="status").columns) + [col for col in df.select_dtypes(["float64", "int64"]).columns if (df[col].nunique() <= 4) & (df[col].nunique() > 2)]))
    num_cols = list(set(df.select_dtypes(["float64", "int64"]).columns).difference(set(list(ord_cols+cat_cols))))    
    return cat_cols, ord_cols, num_cols


def CustomImputer(df, cat_cols, ord_cols, num_cols):
    import pandas as pd
    num_imp = df[num_cols].fillna(-9999)
    ord_imp = df[ord_cols].fillna(0)
    cat_imp = df[cat_cols].fillna(-99)
    missing_indicators = df[cat_cols+num_cols].isnull().astype(int).add_suffix('_indicator')
    df_imputed = pd.concat([num_imp, ord_imp, cat_imp, missing_indicators], axis=1)
    return df_imputed


from sklearn.preprocessing import OneHotEncoder, LabelEncoder, FunctionTransformer, RobustScaler, StandardScaler, MinMaxScaler, KBinsDiscretizer
import category_encoders as ce 
from sklearn.base import BaseEstimator, TransformerMixin
#For keeping the data as it is
class AsIsReturner(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y=None):
        return X

    def transform(self, X, y=None):
        return X
    
#For selecting the best scaler 
class ScalerSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, scaler=StandardScaler()):
        super().__init__()
        self.scaler = scaler

    def fit(self, X, y=None):
        return self.scaler.fit(X)

    def transform(self, X, y=None):
        return self.scaler.transform(X)
    
#For selecting the best encoder    
class EncoderSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, encoder=ce.OneHotEncoder(use_cat_names=True)):
        super().__init__()
        self.encoder = encoder

    def fit(self, X, y):
        return self.encoder.fit(X, y)   

    def transform(self, X, y=None):
        return self.encoder.transform(X)    

#For selecting the best bining method
class BiningSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, bining=KBinsDiscretizer(n_bins=4, encode='onehot-dense')):
        super().__init__()
        self.bining = bining

    def fit(self, X, y=None):
        return self.bining.fit(X)

    def transform(self, X, y=None):
        return self.bining.transform(X)    
        


