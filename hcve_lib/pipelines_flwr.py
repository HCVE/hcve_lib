import logging
from logging import INFO
from typing import Dict

import flwr
import toolz
from flwr.common import GetParametersIns, GetParametersRes, Status, Code, Parameters, FitIns, FitRes
from flwr.common.logger import FLOWER_LOGGER, console_handler
from pandas import DataFrame, Series

from hcve_lib.custom_types import Target
from hcve_lib.pipelines import evaluate_metrics_aggregation, FederatedSurvivalXGBoost


def start_xgb_server(num_client: int, hyperparameters: Dict = None):
    from flwr.server.strategy import FedXgbBagging

    FLOWER_LOGGER.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    hyperparameters = toolz.merge(
        {
            "num_rounds": 10,
        },
        hyperparameters or {},
    )

    strategy = FedXgbBagging(
        min_fit_clients=num_client,
        min_available_clients=num_client,
        min_evaluate_clients=num_client,
        fraction_fit=1,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    )

    return flwr.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=flwr.server.ServerConfig(num_rounds=hyperparameters["num_rounds"]),
    )


def start_xgb_client(
    client_id: int,
    X: DataFrame,
    y,
    X_validate: DataFrame = None,
    y_validate: Target = None,
    hyperparameters: Dict = None,
    sample_weights: Series = None,
):
    print(f"Starting client {client_id}")

    FLOWER_LOGGER.setLevel(INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    client = FederatedXGBoostFlowerClient(
        client_id,
        X,
        y,
        X_validate=X_validate,
        y_validate=y_validate,
        hyperparameters=hyperparameters,
        sample_weights=sample_weights,
    )

    flwr.client.start_client(
        server_address="127.0.0.1:8080",
        client=client,
    )

    return client.model
    # return client


class FederatedXGBoostFlowerClient(flwr.client.Client):
    model = None
    config: str = None

    def __init__(
        self,
        client_id,
        X_train,
        y_train,
        X_validate=None,
        y_validate=None,
        hyperparameters=None,
        sample_weights=None,
    ):
        self.hyperparameters = hyperparameters
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_validate = X_validate
        self.y_validate = y_validate
        self.sample_weights = sample_weights

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def fit(self, ins: FitIns) -> FitRes:
        if not self.model:
            self.model = FederatedSurvivalXGBoost(
                X_validate=self.X_validate,
                y_validate=self.y_validate,
                hyperparameters=self.hyperparameters,
                sample_weights=self.sample_weights,
            )
            self.model.fit(self.X_train, self.y_train)
            self.config = self.model.booster.save_config()
        else:
            global_model = None
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            self.model.booster.load_model(global_model)
            self.model.booster.load_config(self.config)
            self.model.local_boost()

        local_model = self.model.booster.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=len(self.X_train),
            metrics={},
        )

    # def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
    #
    #    self.model.predict(self.X_validate)
    #
    #
    #     return EvaluateRes(
    #         status=Status(
    #             code=Code.OK,
    #             message="OK",
    #         ),
    #         loss=0.0,
    #         num_examples=0,
    #         metrics={"AUC": 0},
    #     )
