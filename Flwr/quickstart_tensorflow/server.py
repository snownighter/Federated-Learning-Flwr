import flwr as fl
from flwr.server.strategy import FedAvg

import utils as uts
import flwr_cust as cuflwr
<<<<<<< HEAD

# FedAvg is the default strategy used when you start the server without a custom strategy
strategy = FedAvg(
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
=======
from flwr_server import CustServer
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.server import Server

# FedAvg is the default strategy used when you start the server without a custom strategy
strategy = FedAvg(
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
>>>>>>> ca5288db1a53e9b996ccddc312fa365f48a26477
    fraction_fit=1,
)

# import socket
# ip_address = socket.gethostbyname((socket.gethostname()))
# server_address=ip_address + ':8080'
client_manager=SimpleClientManager()
# Start Flower server
<<<<<<< HEAD
# fl.server.start_server(
#     server_address=uts.sv,
#     strategy=strategy,
#     config=fl.server.ServerConfig(num_rounds=uts.rs), #rounds
# )
=======
#fl.server.start_server(
#    server_address=uts.sv,
#    strategy=strategy,
#    config=fl.server.ServerConfig(num_rounds=uts.rs), #rounds
#)
>>>>>>> ca5288db1a53e9b996ccddc312fa365f48a26477

cuflwr.re_start_server(
    server_address=uts.sv,
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=uts.rs), #rounds
)