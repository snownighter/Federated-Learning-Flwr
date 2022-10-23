from typing import Optional, Tuple

from flwr.server import Server
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.history import History
from flwr.server.app import ServerConfig
from flwr.server.strategy import FedAvg, Strategy
from flwr.common.logger import log
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from logging import INFO, WARN

from flwr.server import app
from flwr_server import CustServer

DEFAULT_SERVER_ADDRESS = "[::]:8080"

def re_start_server(  # pylint: disable=too-many-arguments
    *,
    server_address: str = DEFAULT_SERVER_ADDRESS,
    server: Optional[CustServer] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    certificates: Optional[Tuple[bytes, bytes, bytes]] = None,
) -> History:
    """Start a Flower server using the gRPC transport layer. """

    # Initialize server and server config
    initialized_server, initialized_config = _init_defaults(
        server=server,
        config=config,
        strategy=strategy,
        client_manager=client_manager,
    )
    log(
        INFO,
        "Starting Flower server, config: %s",
        initialized_config,
    )

    # Start gRPC server
    grpc_server = app.start_grpc_server(
        client_manager=initialized_server.client_manager(),
        server_address=server_address,
        max_message_length=grpc_max_message_length,
        certificates=certificates,
    )
    log(
        INFO,
        "Flower ECE: gRPC server running (%s rounds), SSL is %s",
        initialized_config.num_rounds,
        "enabled" if certificates is not None else "disabled",
    )

    # Start training
    hist = app._fl(
        server=initialized_server,
        config=initialized_config,
    )

    # Stop the gRPC server
    grpc_server.stop(grace=1)

    return hist

def _init_defaults(
    server: Optional[CustServer],
    config: Optional[ServerConfig],
    strategy: Optional[Strategy],
    client_manager: Optional[ClientManager],
) -> Tuple[Server, ServerConfig]:
    # Create server instance if none was given
    if server is None:
        if client_manager is None:
            client_manager = SimpleClientManager()
        if strategy is None:
            strategy = FedAvg()
        server = CustServer(client_manager=client_manager, strategy=strategy)
    elif strategy is not None:
        log(WARN, "Both server and strategy were provided, ignoring strategy")

    # Set default config values
    if config is None:
        config = ServerConfig()

    return server, config