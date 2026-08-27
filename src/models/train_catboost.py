import numpy as np
from catboost import CatBoostRegressor

class CatBoostDirectionalObjective(object):
    """
    Custom Directional Loss Objective for CatBoost Regressor.
    Calculates Gradient and Hessian for optimization.
    """
    def calc_ders_range(self, approxes, targets, weights):
        approxes = np.array(approxes)
        targets = np.array(targets)
        
        # Calculate standard residual error
        der1 = targets - approxes  # First derivative (Gradient)
        der2 = -np.ones_like(approxes)  # Second derivative (Hessian)

        # Apply additional penalty if price direction is incorrect
        if len(approxes) > 1:
            delta_target = targets[1:] - targets[:-1]
            delta_approx = approxes[1:] - approxes[:-1]
            
            # Identify incorrect direction masks
            mismatch_mask = (delta_target * delta_approx) < 0
            
            # Add penalty to gradient on directional mismatch
            der1[:-1][mismatch_mask] += np.sign(delta_target[mismatch_mask]) * 0.5
            
        if weights is not None:
            der1 *= weights
            der2 *= weights

        return list(zip(der1, der2))

# Usage inside CatBoost Training Code:
cb_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    loss_function=CatBoostDirectionalObjective(), # <-- Custom Objective Here
    eval_metric='RMSE'
)

# cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))