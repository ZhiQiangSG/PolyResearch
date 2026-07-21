import unittest
from unittest.mock import patch

from pydantic import ValidationError

from polyresearch.configuration import Configuration
from polyresearch.runtime.model_utils import (
    QwenInvocationLoggingCallback,
    create_qwen_chat_model,
)


class ModelConfigurationTests(unittest.TestCase):
    def test_chat_model_config_includes_bounded_transport_controls(self) -> None:
        configuration = Configuration(
            model_timeout_seconds=90,
            model_max_retries=1,
            planner_model_max_tokens=6000,
        )

        model_config = configuration.chat_model_config(
            model="qwen3.7-max", max_tokens=configuration.planner_model_max_tokens, api_key="test-key"
        )

        self.assertEqual(model_config["timeout"], 90)
        self.assertEqual(model_config["max_retries"], 1)
        self.assertEqual(model_config["max_tokens"], 6000)

    def test_default_retries_use_only_the_structured_output_layer(self) -> None:
        configuration = Configuration()

        self.assertEqual(configuration.model_max_retries, 0)
        self.assertEqual(configuration.max_structured_output_retries, 2)

    def test_model_factory_attaches_invocation_logging_without_wrapping_chat_model(self) -> None:
        sentinel_model = object()
        with (
            patch("polyresearch.runtime.model_utils.get_qwen_api_key", return_value="test-key"),
            patch("polyresearch.runtime.model_utils.init_chat_model", return_value=sentinel_model) as factory,
        ):
            result = create_qwen_chat_model(
                Configuration(), "qwen3.7-max", 6000, {"configurable": {}}
            )

        self.assertIs(result, sentinel_model)
        callback = factory.call_args.kwargs["callbacks"][0]
        self.assertIsInstance(callback, QwenInvocationLoggingCallback)
        self.assertEqual(callback.max_tokens, 6000)

    def test_transport_controls_reject_invalid_values(self) -> None:
        with self.assertRaises(ValidationError):
            Configuration(model_timeout_seconds=0)
        with self.assertRaises(ValidationError):
            Configuration(model_max_retries=-1)
        with self.assertRaises(ValidationError):
            Configuration(planner_model_max_tokens=0)
