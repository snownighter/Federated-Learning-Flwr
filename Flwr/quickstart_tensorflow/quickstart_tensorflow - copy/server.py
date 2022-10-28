import flwr as fl

import utils as uts

# Start Flower server
fl.server.start_server(
    server_address=uts.sv,
    config=fl.server.ServerConfig(num_rounds=uts.rs), #rounds
)
