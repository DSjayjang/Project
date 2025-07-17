# ISIS
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
"""
hyperparams
"""
r('options(warn=-1)')  

nfolds = FloatVector([10])[0]
nsis = FloatVector([100])[0]
seed = FloatVector([9])[0]

family: str = 'gaussian'
tune: str = 'cv'
varISIS: str = 'aggr'
q: float = 0.95
standardize: bool = False

class ISIS:
    """
    rpy2를 이용해 R SIS 패키지의 SIS 함수를 호출하여 
    변수 선택(Feature Screening)을 수행하는 클래스.
    """
    def __init__(self, df):
        pandas2ri.activate()

        self.df = df.copy()
        self.SIS = importr('SIS')
        self._folds = nfolds
        self._nsis = nsis
        self._seed = seed

        self._target = 'target'
        self.scaler = StandardScaler()

    def fit(self):
        X = self.df.drop(columns = ['target'])
        y = self.df['target']

        self.scaler.fit(X)
        X_scaling = self.scaler.transform(X)
        X_r = r['as.matrix'](X_scaling)
        y_r = FloatVector(y)

        self.model = self.SIS.SIS(
            X_r, y_r,
            family = family, tune = tune,
            nfolds = nfolds, nsis = nsis,
            varISIS = varISIS,
            seed = seed, q = q,
            standardize = standardize)
        
        ix_r = np.array(self.model.rx2('ix'))
        self.selected_indices = (ix_r -1).astype(int)

        self.selected_features = list(X.columns[self.selected_indices])

        return self.selected_features
    
    def transform(self):
        cols = self.selected_features + [self._target]

        return self.df[cols].copy()