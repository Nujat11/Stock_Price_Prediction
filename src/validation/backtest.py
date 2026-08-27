import numpy as np
import pandas as pd
from typing import Dict, Any, List

class WalkForwardValidator:
    """
    Implements Rolling-Window Walk-Forward Cross-Validation for Time Series Data.
    Ensures zero data leakage during evaluation.
    """
    def __init__(self, train_window_size: int, test_window_size: int, step_size: int):
        """
        Args:
            train_window_size (int): Number of historical rows used for training.
            test_window_size (int): Number of consecutive rows used for testing.
            step_size (int): Number of rows to slide the window forward in each iteration.
        """
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.step_size = step_size

    def split(self, df: pd.DataFrame):
        """
        Yields train and test index splits sequentially over time.
        """
        n_samples = len(df)
        current_start = 0

        while current_start + self.train_window_size + self.test_window_size <= n_samples:
            train_end = current_start + self.train_window_size
            test_end = train_end + self.test_window_size

            train_indices = np.arange(current_start, train_end)
            test_indices = np.arange(train_end, test_end)

            yield train_indices, test_indices

            # Slide window forward
            current_start += self.step_size

    def evaluate_model(self, model: Any, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Dict[str, float]:
        """
        Executes the walk-forward validation and calculates performance metrics.
        """
        predictions = []
        actuals = []

        for train_idx, test_idx in self.split(df):
            X_train = df.iloc[train_idx][feature_cols]
            y_train = df.iloc[train_idx][target_col]

            X_test = df.iloc[test_idx][feature_cols]
            y_test = df.iloc[test_idx][target_col]

            # Fit model on historical fold window
            model.fit(X_train, y_train)

            # Predict test window
            preds = model.predict(X_test)

            predictions.extend(preds)
            actuals.extend(y_test.values)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Compute Metrics
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        
        # Calculate Directional Accuracy (% of correct up/down predictions)
        pred_direction = np.diff(predictions) > 0
        actual_direction = np.diff(actuals) > 0
        directional_acc = np.mean(pred_direction == actual_direction) * 100

        return {
            "RMSE": rmse,
            "MAE": mae,
            "Directional_Accuracy (%)": directional_acc
        }