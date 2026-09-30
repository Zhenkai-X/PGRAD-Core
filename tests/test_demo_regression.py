# -*- coding: utf-8 -*-
"""Public repository checks that do not require private research assets."""

from pathlib import Path
import unittest

from core_method.config_loader import load_config
import pgrad_app.application
import pgrad_app.pipeline


class PublicRepositoryTests(unittest.TestCase):
    def test_public_modules_import(self) -> None:
        self.assertTrue(callable(pgrad_app.pipeline.process_cases))
        self.assertTrue(callable(pgrad_app.application.main))

    def test_config_loads(self) -> None:
        config = load_config()
        self.assertEqual(config["llm"]["temperature"], 0.1)
        self.assertEqual(config["prior_model"]["top_k_predictors"], 20)

    def test_public_directories_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        required = [
            root / "models",
            root / "data" / "reference",
            root / "data" / "evidence" / "correct",
            root / "data" / "evidence" / "error",
            root / "data" / "examples" / "single_inspiratory",
            root / "data" / "examples" / "paired_inspiratory_expiratory",
            root / "outputs",
            root / "tests" / "fixtures" / "single_inspiratory",
            root / "tests" / "fixtures" / "paired_inspiratory_expiratory",
        ]
        for path in required:
            self.assertTrue(path.is_dir(), path)


if __name__ == "__main__":
    unittest.main()
