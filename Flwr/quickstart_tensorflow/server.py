import flwr as fl
from flwr.server.strategy import FedAvg

import utils as uts
import flwr_cust as cuflwr
from flwr_server import CustServer
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.server import Server

# FedAvg is the default strategy used when you start the server without a custom strategy
strategy = FedAvg(
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    fraction_fit=1,
)

# Start Flower server
fl.server.start_server(
    server_address=uts.sp.sv,
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=uts.sp.rs) #rounds
)


# cuflwr.re_start_server(
#     server_address=uts.sp.sv,
#     strategy=strategy,
#     config=fl.server.ServerConfig(num_rounds=uts.sp.rs) #rounds
# )
