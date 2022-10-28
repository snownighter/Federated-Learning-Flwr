import flwr as fl
from flwr.server.strategy import FedAvg

import utils as uts
import flwr_cust as cuflwr

# FedAvg is the default strategy used when you start the server without a custom strategy
strategy = FedAvg(
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    fraction_fit=1,
)

# import socket
# ip_address = socket.gethostbyname((socket.gethostname()))
# server_address=ip_address + ':8080'

# Start Flower server
# fl.server.start_server(
#     server_address=uts.sv,
#     strategy=strategy,
#     config=fl.server.ServerConfig(num_rounds=uts.rs), #rounds
# )

cuflwr.re_start_server(
    server_address=uts.sv,
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=uts.rs), #rounds
)