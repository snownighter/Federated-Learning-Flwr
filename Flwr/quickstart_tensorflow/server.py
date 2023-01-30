import flandre as fl
from flandre.server.strategy import FedAvg, FedAvgM, QFedAvg

import utils as uts
from FedFlan import FedFlan

N = 5
AVG = 'AvgM'

# FedAvg is the default strategy used when you start the server without a custom strategy
def FedAvg_mode(avg):
    global N
    if   avg == 'Flan':
        strategy = FedFlan(
            min_fit_clients=N,
            min_evaluate_clients=N,
            min_available_clients=N,
            fraction_fit=1,
        )
    elif avg == 'Avg':
        strategy = FedAvg(
            min_fit_clients=N,
            min_evaluate_clients=N,
            min_available_clients=N,
            fraction_fit=1,
        )
    elif avg == 'AvgM':
        strategy = FedAvgM(
            min_fit_clients=N,
            min_evaluate_clients=N,
            min_available_clients=N,
            fraction_fit=1,
            server_momentum=0.7
        )
    elif avg == 'QAvg':
        strategy = QFedAvg(
            min_fit_clients=N,
            min_evaluate_clients=N,
            min_available_clients=N,
            fraction_fit=1,
        )
    return strategy

# Start Flower server
fl.server.start_server(
    server_address=uts.sp.sv,
    strategy=FedAvg_mode(AVG),
    config=fl.server.ServerConfig(num_rounds=uts.sp.rs) #rounds
)
