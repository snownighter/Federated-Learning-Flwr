import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)

from flwr.server.server import FitResultsAndFailures
from flwr.server.server import Server
from flwr.server.server import fit_clients

# free memory
import reset_keras as reset

flwr_round = None

class CustServer(Server):
    #def __init__(self, *, client_manager: ClientManager, strategy: Optional[Strategy] = None) -> None:
    #    super().__init__(client_manager=client_manager, strategy=None)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        # record current round
        global flwr_round
        flwr_round = server_round
        reset.reset_keras()

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)