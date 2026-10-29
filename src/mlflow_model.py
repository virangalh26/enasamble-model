"""MLflow pyfunc wrapper, so the registered model can be loaded with `mlflow.pyfunc.load_model`.

Input: a DataFrame with columns h00..h23 (kWh imported per hour, one row per day).
       With an extra `hh` column the output is one row per household (mean P(solar) + verdict).
"""
import mlflow.pyfunc
import pandas as pd


class SolarDetectorModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.detector = joblib.load(context.artifacts["detector"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        if "hh" in model_input.columns:
            return self.detector.predict_household(model_input).reset_index()
        return pd.DataFrame({"p_solar": self.detector.predict_day_proba(model_input)})
