from pathlib import Path

root = Path(__file__).parent
path_to_logs = root.joinpath('logs')
path_to_lm_wts = root.joinpath('weights')
path_to_processor = root.joinpath('data_processor')