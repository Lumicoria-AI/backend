import logging
import os

# Create logs directory if it doesn't exist
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, 'app.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('lumicoria.agents')

def get_logger(name=None):
    """
    Returns a logger with the specified name, or the default agent logger.
    """
    if name:
        return logging.getLogger(f'lumicoria.agents.{name}')
    return logger