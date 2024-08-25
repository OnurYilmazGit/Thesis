
import lightgbm as lgb

def get_model():
    return lgb.LGBMClassifier(device='gpu', random_state=42)
