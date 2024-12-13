import logging
import sys
import mattermostdriver
from websocket_patched import PatchedWebsocket
from config import log_level

logger = logging.getLogger(__name__)
logger.setLevel(log_level)

mattermostdriver.websocket.Websocket = PatchedWebsocket

if 'mattermostdriver.driver' in sys.modules:
    del sys.modules['mattermostdriver.driver']

import mattermostdriver.driver  # noqa: E402

mattermostdriver.driver.Websocket = PatchedWebsocket

from mattermostdriver.driver import Driver  # noqa: E402

Driver = Driver
