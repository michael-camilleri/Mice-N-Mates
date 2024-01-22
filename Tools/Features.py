from mpctools.extensions import skext

import numpy as np
import pandas as pd

class ObservabilityFeatures:
    """
    Wrapper for Standardising Observability Features.

    Note that the class automatically Handles NaNs in the LFB
    """
    @staticmethod
    def fill_nans(lfb: pd.Series):
        """
        Fill LFB array NaNs
        """
        return pd.Series(
            [np.zeros(2048) if np.any(pd.isna(ll)) else ll for ll in lfb],
            index=lfb.index,
            name=lfb.name
        )

    def __init__(
            self,
            bti=25,
            max_area=60000,
            max_lfb=1.5,
            num_antennas=18,
            num_components=30,
            columns=('TIM.Dets', 'TIM.Area', 'RFID.Pos', 'LFB.Raw'),
            as_frame=False,
    ):
        self.__bti = bti
        self.__max_area = max_area
        self.__max_lfb = max_lfb
        self.__nant = num_antennas
        self.__pca = skdecomp.PCA(num_components, svd_solver='full')
        self.__lnorm = (None, None)
        self.__d, self.__a, self.__p, self.__l = columns
        self.__as_df = [
            self.__d,
            self.__a,
            *[f'{self.__p}.{i}' for i in range(num_antennas)],
            *[f'{self.__l}.{i}' for i in range(num_components)]
        ] if as_frame else None

    def _prepare_pca(self, X):
        return np.clip(np.stack(self.fill_nans(X[self.__l])), 0, self.__max_lfb)

    def fit(self, X):
        """
        Fits the PCA components, as well as the scaling required.
        """
        X = self._prepare_pca(X)
        # First Fit PCA
        self.__pca.fit(X)
        # Now Fit Scaling
        X = self.__pca.transform(X)
        self.__lnorm = (np.ptp(X), np.min(X))
        return self

    def transform(self, X):
        """
        Transform the Features

        :param X: Pandas DataFrame to transform
        :return: Numpy array of transformed features
        """
        # Prepare Placeholder
        X_tr = np.empty([len(X), 2 + self.__nant + self.__pca.n_components])
        # TIM stuff
        X_tr[:, 0] = X[self.__d].to_numpy() / self.__bti
        X_tr[:, 1] = X[self.__a].clip(upper=self.__max_area).to_numpy() / self.__max_area
        # RFID stuff
        o_hot = sk_onehot(categories=[np.arange(1, self.__nant + 1)], sparse=False)
        X_tr[:, 2:2+self.__nant] = o_hot.fit_transform(X[self.__p].to_numpy().reshape(-1, 1))
        # LFB stuff
        X_tr[:, 2+self.__nant:] = (self.__pca.transform(self._prepare_pca(X)) - self.__lnorm[1]) / self.__lnorm[0]
        if self.__as_df is not None:
            X_tr = pd.DataFrame(X_tr, index=X.index, columns=self.__as_df)
        return X_tr


class BehaviourCalibrator():
    """
    Implementation of Fusion which applies selective temperature scaling based on a flag.

    The model expects, for each sample, the following features (of [size]):
     * [7] - Label Logits
     * [1] - TIM Detections (0 or more)
    """
    def __init__(self, theta_init=1.0, lr=1e-4, max_iter=100):
        """
        Initialises the Model
        :param theta_init: Initial value for scaling parameter
        :param lr:  Learning rate
        :param max_iter: Maximum number of iterations
        """
        # Initialise Internal Model
        #   - Note that we do not fit the Prior calibrator, but just use the default theta
        self.__calib_lfb = skext.LogitCalibrator(np.arange(7), theta_init, lr, max_iter)
        self.__calib_prior = skext.LogitCalibrator(np.arange(7), 1.0, None, None)

    def fit(self, X, y):
        """
        Fits the model on the training Data

        :param X: The input Features (see description of class)
        :param y: The output labels (one of L behaviours, 0-indexed)
        """
        # Select the Samples which need calibration and fit on them
        non_prior = (X[:, 7] > 0)
        self.__calib_lfb.fit(X[non_prior, :7], y[non_prior])

        # Return self for chaining
        return self

    def predict_proba(self, X):
        """
        Predict Probabilities
        """
        non_prior = (X[:, 7] > 0); prior = ~non_prior
        probs = np.empty([X.shape[0], 7])
        probs[non_prior, :] = self.__calib_lfb.predict_proba(X[non_prior, :7])
        probs[prior, :] = self.__calib_prior.predict_proba(X[prior, :7])
        return probs

    def predict(self, X):
        """
        Predict Behaviour

        Note, this is really a dummy, since the ordering does not change after calibration.
        """
        return np.argmax(X[:, :7], axis=1)

    @property
    def classes_(self):
        return np.copy(self.__calib_lfb.classes_)

    @property
    def theta(self):
        return self.__calib_lfb.theta

    @property
    def calibrator(self):
        return copy.deepcopy(self.__calib_lfb)