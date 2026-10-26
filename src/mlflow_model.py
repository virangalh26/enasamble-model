"""MLflow pyfunc wrapper, so the registered model can be loaded with `mlflow.pyfunc.load_model`.

Input: a DataFrame of households with the raw feature columns of `src.features` (one row per household).
Output: P(poor), predicted consumption per person with an 80% interval, and the support flag.
"""
import mlflow.pyfunc
import pandas as pd


class PovertyTargeterModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.targeter = joblib.load(context.artifacts["targeter"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return self.targeter.predict(model_input)
