import torch
import torch.nn as nn

class DirectionalMSELoss(nn.Module):
    """
    Custom Loss Function for Time Series Forecasting.
    Combines Standard Mean Squared Error (MSE) with a penalty 
    for incorrect price movement direction (Up/Down).
    """
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha (float): Weight multiplier for the directional penalty term.
        """
        super(DirectionalMSELoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # 1. Compute baseline Mean Squared Error
        mse_loss = self.mse(y_pred, y_true)
        
        # 2. Calculate step-by-step price changes (Direction Delta)
        delta_true = y_true[1:] - y_true[:-1]
        delta_pred = y_pred[1:] - y_pred[:-1]
        
        # 3. Apply penalty when predictions move in opposite direction
        # Multiplying signs: negative result means direction mismatch
        direction_penalty = torch.relu(-delta_true * delta_pred)
        
        # 4. Total Loss calculation
        total_loss = mse_loss + self.alpha * torch.mean(direction_penalty)
        return total_loss