import flwr as fl
import utils
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""
    test = pd.read_csv("../data/minmax-test.csv")
    y_target = test.iloc[:, 10]
    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.load_data(3)
    #_, (X_test, y_test) = utils.load_mnist()

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):

        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate

sv = "127.0.0.1:8080"

# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    #model = LogisticRegression()
    model = utils.create_model()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address=sv,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10),
    )
