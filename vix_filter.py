
import numpy as np

def apply_vix_strategy(vix_safe_series, predictions):
    """
    Args:
        vix_safe_series: The 'VIX_Safe' column from your test_df (0s and 1s)
        predictions: The 0/1 output from your ML model
    """
    # 1. Convert to numpy arrays to ensure simple element-wise multiplication
    vix_array = np.array(vix_safe_series)
    pred_array = np.array(predictions)
    
    # 2. Logic Check: Ensure they are the same length
    if len(vix_array) != len(pred_array):
        raise ValueError(f"Length mismatch! VIX: {len(vix_array)}, Preds: {len(pred_array)}")
    
    # 3. Apply the 'AND' logic via multiplication
    final_signals = pred_array * vix_array
    
    return final_signals