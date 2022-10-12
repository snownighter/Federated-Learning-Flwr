import flwr as fl
from flwr.server.strategy import FedAvg

import utils as uts

# FedAvg is the default strategy used when you start the server without a custom strategy
strategy = FedAvg(
    min_fit_clients=3,
    min_evaluate_clients=4,
    # Minimum number of connected clients before sampling e.g. 10
    min_available_clients=4,
    # Fraction of clients which should participate in each round
    fraction_fit=0.75,
)

# import socket
# ip_address = socket.gethostbyname((socket.gethostname()))
# server_address=ip_address + ':8080'

# Start Flower server
fl.server.start_server(
    server_address=uts.sv,
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=uts.rs), #rounds
)
