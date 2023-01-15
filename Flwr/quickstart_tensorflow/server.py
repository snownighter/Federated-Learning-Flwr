import flandre as fl
from flandre.server.strategy import FedAvg

import utils as uts
import exFedAvg as eFedAvg

N = 2

# FedAvg is the default strategy used when you start the server without a custom strategy
strategy = FedAvg(
    min_fit_clients=N,
    min_evaluate_clients=N,
    min_available_clients=N,
    fraction_fit=1,
)

# Start Flower server
fl.server.start_server(
    server_address=uts.sp.sv,
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=uts.sp.rs) #rounds
)
