from datetime import datetime

def should_retrain(config):
    if not config.get("auto_retrain", False):
        return False
    today = datetime.now().strftime("%A")
    return today.lower() == config.get("retrain_day", "Sunday").lower()
