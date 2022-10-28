from typing import Optional, Tuple

from flwr.server import Server
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.history import History
from flwr.server.app import ServerConfig
<<<<<<< HEAD
from flwr.server.strategy import Strategy
from flwr.common.logger import log
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from logging import INFO
=======
from flwr.server.strategy import FedAvg, Strategy
from flwr.common.logger import log
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from logging import INFO, WARN
>>>>>>> ca5288db1a53e9b996ccddc312fa365f48a26477

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
<<<<<<< HEAD
    """Start a Flower server using the gRPC transport layer.

    Arguments
    ---------
        server_address: Optional[str] (default: `"[::]:8080"`). The IPv6
            address of the server.
        server: Optional[flwr.server.Server] (default: None). An implementation
            of the abstract base class `flwr.server.Server`. If no instance is
            provided, then `start_server` will create one.
        config: ServerConfig (default: None).
            Currently supported values are `num_rounds` (int, default: 1) and
            `round_timeout` in seconds (float, default: None).
        strategy: Optional[flwr.server.Strategy] (default: None). An
            implementation of the abstract base class `flwr.server.Strategy`.
            If no strategy is provided, then `start_server` will use
            `flwr.server.strategy.FedAvg`.
        client_manager: Optional[flwr.server.ClientManager] (default: None)
            An implementation of the abstract base class `flwr.server.ClientManager`.
            If no implementation is provided, then `start_server` will use
            `flwr.server.client_manager.SimpleClientManager`.
        grpc_max_message_length: int (default: 536_870_912, this equals 512MB).
            The maximum length of gRPC messages that can be exchanged with the
            Flower clients. The default should be sufficient for most models.
            Users who train very large models might need to increase this
            value. Note that the Flower clients need to be started with the
            same value (see `flwr.client.start_client`), otherwise clients will
            not know about the increased limit and block larger messages.
        certificates : Tuple[bytes, bytes, bytes] (default: None)
            Tuple containing root certificate, server certificate, and private key to
            start a secure SSL-enabled server. The tuple is expected to have three bytes
            elements in the following order:

                * CA certificate.
                * server certificate.
                * server private key.

    Returns
    -------
        hist: flwr.server.history.History. Object containing metrics from training.

    Examples
    --------
    Starting an insecure server:

    >>> start_server()

    Starting a SSL-enabled server:

    >>> start_server(
    >>>     certificates=(
    >>>         Path("/crts/root.pem").read_bytes(),
    >>>         Path("/crts/localhost.crt").read_bytes(),
    >>>         Path("/crts/localhost.key").read_bytes()
    >>>     )
    >>> )
    """

    # Initialize server and server config
    initialized_server, initialized_config = app._init_defaults(
=======
    """Start a Flower server using the gRPC transport layer. """

    # Initialize server and server config
    initialized_server, initialized_config = _init_defaults(
>>>>>>> ca5288db1a53e9b996ccddc312fa365f48a26477
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
<<<<<<< HEAD
=======

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
>>>>>>> ca5288db1a53e9b996ccddc312fa365f48a26477
