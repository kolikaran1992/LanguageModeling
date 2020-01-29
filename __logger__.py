from .__paths__ import path_to_logs
LOGGER_NAME = 'LM'
import logging
import datetime

logger_console = logging.getLogger(LOGGER_NAME+'_console')
logger_file = logging.getLogger(LOGGER_NAME+'_file')

formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s')

# define file handler and set formatter
file_handler = logging.FileHandler(path_to_logs.joinpath(datetime.datetime.now().strftime("%Y-%m-%d")).as_posix())
file_handler.setFormatter(logging.Formatter(''))

# define console handler and set formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# add handlers to logger
logger_file.addHandler(file_handler)
logger_console.addHandler(console_handler)

logger_file.setLevel(logging.INFO)
logger_console.setLevel(logging.INFO)