import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from gluonts.ext.prophet import ProphetPredictor
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError
from gluonts.dataset.common import ListDataset
from pathlib import Path


warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Forecaster:
    """A wrapper class for the Prophet Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "Prophet Forecaster"
    made_up_frequency = "S"  # by seconds
    made_up_start_dt = "2000-01-01 00:00:00"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        history_forecast_ratio: Optional[int] = None,
        use_exogenous: bool = True,
        random_state: int = 0,
        **kwargs,
    ):
        """Construct a new Prophet Forecaster

        Args:

            data_schema (ForecastingSchema): Schema of the data used for training.

            history_forecast_ratio (Optional[int]):
                Sets the history length depending on the forecast horizon.
                For example, if the forecast horizon is 20 and the history_forecast_ratio is 10,
                history length will be 20*10 = 200 samples.

            use_exogenous (bool): If true, uses covariates in training.

            random_state (int): Sets the underlying random seed at model initialization time.

            kwargs: Parameters to pass when instantiating the prophet model.
        """

        self.data_schema = data_schema
        self.random_state = random_state
        self.use_exogenous = use_exogenous and data_schema.future_covariates
        self.freq = self.map_frequency(data_schema.frequency)
        self._is_trained = False
        self.history_length = None
        self.gluonts_dataset = None

        if history_forecast_ratio:
            self.history_length = (
                self.data_schema.forecast_length * history_forecast_ratio
            )

        self.model = ProphetPredictor(
            prediction_length=data_schema.forecast_length, prophet_params=kwargs
        )

    def prepare_time_column(
        self, data: pd.DataFrame, is_train: bool = True
    ) -> pd.DataFrame:
        """
        Adds time column of type DATETIME to datasets that have time column dtype as INT.

        Args:
            data (pd.DataFrame): The input dataset.
            is_train (bool): Set to true for training dataset and false for testing dataset.

            Returns (pd.DataFrame): The dataset after processing time column.
        """
        # sort data
        time_col_dtype = self.data_schema.time_col_dtype
        id_col = self.data_schema.id_col
        time_col = self.data_schema.time_col

        data = data.sort_values(by=[id_col, time_col])

        if time_col_dtype == "INT":
            # Find the number of rows for each location (assuming all locations have
            # the same number of rows)
            series_val_counts = data[id_col].value_counts()
            series_len = series_val_counts.iloc[0]
            num_series = series_val_counts.shape[0]

            if is_train:
                # since GluonTS requires a date column, we will make up a timeline
                start_date = pd.Timestamp(self.made_up_start_dt)
                datetimes = pd.date_range(
                    start=start_date, periods=series_len, freq=self.made_up_frequency
                )
                self.last_timestamp = datetimes[-1]
                self.timedelta = datetimes[-1] - datetimes[-2]

            else:
                start_date = self.last_timestamp + self.timedelta
                datetimes = pd.date_range(
                    start=start_date, periods=series_len, freq=self.made_up_frequency
                )
            int_vals = sorted(data[time_col].unique().tolist())
            self.time_to_int_map = dict(zip(datetimes, int_vals))
            # Repeat the datetime range for each location
            data[time_col] = list(datetimes) * num_series
        else:
            data[time_col] = pd.to_datetime(data[time_col])
            data[time_col] = data[time_col].dt.tz_localize(None)

        return data

    def prepare_training_data(
        self, history: pd.DataFrame, test_data: pd.DataFrame
    ) -> ListDataset:
        """
        Applys the history_forecast_ratio parameter and puts the training data into the shape expected by GluonTS.

        Args:
            history (pd.DataFrame): The input dataset.
            test_data (pd.DataFrame): The testing dataframe.

        Returns (ListDataset): The processed dataset expected by GluonTS.
        """
        data_schema = self.data_schema
        # Make sure there is a date column
        history = self.prepare_time_column(data=history, is_train=True)

        # Manage each series in the training data separately
        all_covariates = []
        groups_by_ids = history.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
            for id_ in all_ids
        ]

        # Enforces the history_forecast_ratio parameter
        if self.history_length:
            new_length = []
            for series in all_series:
                series = series.iloc[-self.history_length :]
                new_length.append(series.copy())
            all_series = new_length

        cov_names = []

        if self.use_exogenous:
            cov_names = data_schema.future_covariates

        # Put future covariates into separate list
        all_covariates = []

        for id, series in zip(all_ids, all_series):
            series_covariates = []

            for covariate in cov_names:
                series_covariates.append(
                    pd.concat(
                        [
                            series[covariate],
                            test_data[test_data[self.data_schema.id_col] == id][
                                covariate
                            ],
                        ]
                    )
                )

            all_covariates.append(series_covariates)

        # If future covariates are available for training, create a dataset with future covariate features,
        # otherwise a dataset with only target series will be created.
        if cov_names and self.use_exogenous:
            list_dataset = [
                {
                    "start": series[data_schema.time_col].iloc[0],
                    "target": series[data_schema.target],
                    "feat_dynamic_real": series_covariates,
                }
                for series, series_covariates in zip(all_series, all_covariates)
            ]
        else:
            list_dataset = [
                {
                    "start": series[data_schema.time_col].iloc[0],
                    "target": series[data_schema.target],
                }
                for series in all_series
            ]

        gluonts_dataset = ListDataset(list_dataset, freq=self.freq)

        self.training_all_series = all_series
        self.training_future_covariates = all_covariates
        self.all_ids = all_ids

        return gluonts_dataset

    def map_frequency(self, frequency: str) -> str:
        """
        Maps the frequency in the data schema to the frequency expected by GluonTS.

        Args:
            frequency (str): The frequency from the schema.

        Returns (str): The mapped frequency.
        """
        frequency = frequency.lower()
        frequency = frequency.split("frequency.")[1]
        if frequency == "yearly":
            return "Y"
        if frequency == "quarterly":
            return "Q"
        if frequency == "monthly":
            return "M"
        if frequency == "weekly":
            return "W"
        if frequency == "daily":
            return "D"
        if frequency == "hourly":
            return "H"
        if frequency == "minutely":
            return "min"
        if frequency in ["secondly", "other"]:
            return "S"

    def fit(
        self, history: pd.DataFrame, test_data: pd.DataFrame, prediction_col_name: str
    ) -> None:
        """Fit the Forecaster to the training data.
        A separate Prophet model is fit to each series that is contained
        in the data.

        Args:
            history (pandas.DataFrame): The features of the training data.
            test_data (pandas.DataFrame): The testing dataframe.
            prediction_col_name (str): The name of the prediction column in the result dataframe.
        """
        np.random.seed(self.random_state)

        history = self.prepare_training_data(history=history, test_data=test_data)
        predictions = self.model.predict(history)
        values = []
        for forecast in predictions:
            mean = list(forecast.mean)
            values += mean

        predictions_df = test_data.copy()
        predictions_df[prediction_col_name] = values
        self.predictions = predictions_df
        self._is_trained = True

    def predict(self) -> pd.DataFrame:
        """Make the forecast of given length.

        Returns:
            pd.DataFrame: The predictions dataframe.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        return self.predictions

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        self.model.serialize(Path(model_dir_path))
        self.model = None
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        predictor = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        predictor.model = ProphetPredictor.deserialize(Path(model_dir_path))
        return predictor

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    test_data: pd.DataFrame,
    prediction_col_name: str,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """

    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(
        history=history, test_data=test_data, prediction_col_name=prediction_col_name
    )
    return model


def predict_with_model(model: Forecaster) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict()


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
