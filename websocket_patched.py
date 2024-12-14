import logging
import json
import typing
import asyncio
import aiohttp
from config import log_level

logger = logging.getLogger(__name__)
logger.setLevel(log_level)


class PatchedWebsocket:
    """
    Simple class to override outdated mattermostdriver's websocket class, as its causing reconnection issues
    """
    heartbeat: int = 5
    keepalive_delay: float = 5

    def __init__(self, options: typing.Dict[str, typing.Any], token: str):
        self.options = options
        self._token = token
        self._alive = False

    async def connect(
            self,
            event_handler: typing.Callable[[str], typing.Awaitable[None]],
    ) -> None:
        logger.info("Connecting websocket")
        scheme = "wss://" if self.options.get("scheme", "https") == "https" else "ws://"
        url = f"{scheme}{self.options['url']}:{self.options['port']}{self.options['basepath']}/websocket"
        kw_args = {}
        if self.options["websocket_kw_args"] is not None:
            kw_args = self.options["websocket_kw_args"]
        proxy = self.options.get("proxy", None)
        self._alive = True
        while self._alive:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                            url,
                            heartbeat=self.heartbeat,
                            receive_timeout=self.options.get("timeout", 10),
                            verify_ssl=self.options["verify"],
                            proxy=proxy,
                            **kw_args,
                    ) as websocket:
                        await self._authenticate(websocket)
                        async for message in websocket:
                            await event_handler(message.data)
            except Exception as e:
                logger.exception(
                    f"Failed to establish websocket connection: {type(e)} thrown",
                )
                await asyncio.sleep(self.keepalive_delay)

    def disconnect(self) -> None:
        logger.info("Disconnecting websocket")
        self._alive = False

    async def _authenticate(self, websocket: aiohttp.client.ClientWebSocketResponse) -> None:
        logger.info("Authenticating websocket")
        json_data = json.dumps(
            {
                "seq": 1,
                "action": "authentication_challenge",
                "data": {"token": self._token},
            },
        )
        await websocket.send_str(json_data)
