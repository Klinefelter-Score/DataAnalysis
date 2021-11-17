"""Definition of Catboost classifier"""

import Dense


class KerasClassifierWrapper:
    def predict(self, x, **kwargs):
        """Returns the class predictions for the given test data.

        Arguments:
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.

        Returns:
            preds: array-like, shape `(n_samples,)`
                Class predictions.
        """
        import numpy as np
        raw_prediction = np.array(self.model.predict(x))
        if raw_prediction.shape[-1] == 1:
            classes = [1 if p >= 0.5 else 0 for p in raw_prediction]
        elif raw_prediction.shape[-1] == 2:
            classes = np.argmax(raw_prediction, axis=-1)
        else:
            # First two cases should be sufficient
            # shape [num_samples, binary or num_classes]
            raise NotImplementedError
        return classes

    def predict_proba(self, x, **kwargs):
        """Returns the class predictions for the given test data.

        Arguments:
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.

        Returns:
            preds: array-like, shape `(n_samples,)`
                Class predictions.
        """
        import numpy as np
        raw_prediction = np.array(self.model.predict(x))
        return raw_prediction


class KerasMLP:
    """Keras Classifier."""
    def __init__(self, *args, **kwargs):
        self.__weights_path = None
        self.__build_fn = kwargs.pop("build_fn", Dense.get_builder(6))
        self.__epochs = kwargs.pop("epochs", 200)
        super().__init__(*args, **kwargs)
        self._SKEstimator = KerasClassifierWrapper(build_fn=self.__build_fn, **kwargs)

    def fit(self, X, y, **kwargs):
        return self._SKEstimator.fit(X, y, epochs=self.__epochs, **kwargs)

    def predict(self, X, **kwargs):
        if self.__weights_path is not None:
            self._SKEstimator.model.l
        return self._SKEstimator.predict(X, **kwargs)
