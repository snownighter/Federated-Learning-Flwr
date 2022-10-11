import flwr as fl
from flwr.server.strategy import FedAvg

import utils as uts

# FedAvg is the default strategy used when you start the server without a custom strategy
strategy = FedAvg(
    min_fit_clients=2,
    min_evaluate_clients=3,
    # Minimum number of connected clients before sampling e.g. 10
    min_available_clients=3,
    # Fraction of clients which should participate in each round
    fraction_fit=1
)

print(uts.sv)

# Start Flower server
fl.server.start_server(
    server_address=uts.sv,
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=uts.rs), #rounds
)
