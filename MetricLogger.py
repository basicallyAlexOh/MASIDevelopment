
import os
from pathlib import Path
from datetime import datetime

class MetricLogger():
    def __init__(self, config, config_id):
        self.config_id =config_id
        self.log_dir = os.path.join(config["log_dir"], config_id)
        self.datetime = datetime.now().strftime('%Y%m%d.%H%M')

    def log(self, metric_name, stats):
        epoch, metric = stats
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{self.config_id}_{metric_name}_{self.datetime}"
        logf = os.path.join(self.log_dir, filename)
        with open(logf, 'a+') as f:
            f.write(f"{epoch}, {metric}\n")