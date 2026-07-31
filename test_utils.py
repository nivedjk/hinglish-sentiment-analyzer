from src.utils.device import get_device
from src.utils.seed import set_seed
from src.utils.logger import get_logger

set_seed()

logger = get_logger()

device = get_device()

logger.info(f"Using device: {device}")