from dataclasses import MISSING, dataclass, field


class FeatureConfig:
    pass


class ModelConfig:
    pass


class MLForecast:
    def __init__(
        self,
        model_config: ModelConfig,
        feature_config: FeatureConfig,
        target_transformer: object = None,
    ) -> None:
        """Convenient wrapper around scikit-learn style estimators

        Args:
            model_config (ModelConfig): Instance of the ModelConfig object defining the model
            feature_config (FeatureConfig): Instance of the FeatureConfig object defining the features
            missing_config (MissingValueConfig, optional): Instance of the MissingValueConfig object
                defining how to fill missing values. Defaults to None.
            target_transformer (object, optional): Instance of target transformers from src.transforms.
                Should support `fit`, `transform`, and `inverse_transform`. It should also
                return `pd.Series` with datetime index to work without an error. Defaults to None.
        """
        self.model_config = model_config
        self.feature_config = feature_config
        self.target_transformer = target_transformer
        self._model = model_config.model

    def fit(self):
        pass

    def predict(self):
        pass

    def feature_importance(self):
        pass
