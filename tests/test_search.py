"""Model search: CLI chooser, shared filter and the /v1/providers/{name}/models query."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from quanta.app import create_app
from quanta.config import example_config, validate_config
from quanta.picker import choose_model, filter_models

MODELS = [f"vendor{i}/model-{i}" for i in range(500)] + ["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-4o", "anthropic/claude-opus"]


def scripted(answers):
    queue = iter(answers)
    return lambda prompt: next(queue)


class FilterTests(unittest.TestCase):
    def test_case_insensitive_multi_term_and(self):
        self.assertEqual(filter_models(MODELS, "GPT 5"), ["openai/gpt-5", "openai/gpt-5-mini"])
        self.assertEqual(filter_models(MODELS, "mini openai"), ["openai/gpt-5-mini"])
        self.assertEqual(filter_models(MODELS, "zzz"), [])
        self.assertEqual(len(filter_models(MODELS, "")), len(MODELS))


class ChooseModelTests(unittest.TestCase):
    def pick(self, answers, models=MODELS):
        with patch("builtins.print"):
            return choose_model("opencode", models, reader=scripted(answers))

    def test_search_then_number(self):
        self.assertEqual(self.pick(["gpt 5", "2"]), "openai/gpt-5-mini")

    def test_single_match_is_selected_automatically(self):
        self.assertEqual(self.pick(["claude"]), "anthropic/claude-opus")

    def test_exact_id_and_initial_number(self):
        self.assertEqual(self.pick(["openai/gpt-4o"]), "openai/gpt-4o")
        self.assertEqual(self.pick(["1"]), "vendor0/model-0")

    def test_number_refers_to_latest_shown_list(self):
        self.assertEqual(self.pick(["openai", "3", ]), "openai/gpt-4o")

    def test_no_match_needs_confirmation_for_custom_id(self):
        self.assertEqual(self.pick(["my-own-model", "y"]), "my-own-model")
        self.assertEqual(self.pick(["my-own-model", "n", "claude"]), "anthropic/claude-opus")

    def test_blank_cancels(self):
        self.assertEqual(self.pick([""]), "")

    def test_long_list_is_paged_and_narrowing_is_hinted(self):
        with patch("builtins.print") as printed:
            choose_model("opencode", MODELS, reader=scripted([""]))
        text = "\n".join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn("more - type part of a name to narrow", text)
        self.assertNotIn("vendor100/", text)  # not dumped all 500

    def test_no_listing_falls_back_to_free_text(self):
        self.assertEqual(self.pick(["  some/model  "], models=[]), "some/model")

    def test_gives_up_after_many_invalid_rounds(self):
        with patch("builtins.print"):
            self.assertEqual(choose_model("x", MODELS, reader=lambda p: "zz" if "custom" not in p else "n"), "")


class ModelsEndpointTests(unittest.TestCase):
    def setUp(self):
        raw = example_config()
        raw["server"]["apiKey"] = "test-" + "k" * 24
        config = validate_config(raw, env={})
        self.client = TestClient(create_app(config, runner=lambda *a, **k: None), raise_server_exceptions=False)
        self.headers = {"Authorization": "Bearer " + config.server.api_key}

    def get(self, query=""):
        with patch("quanta.picker._provider_models", return_value=MODELS):
            return self.client.get(f"/v1/providers/opencode/models{query}", headers=self.headers)

    def test_search_and_limit(self):
        data = self.get("?q=gpt%205").json()
        self.assertEqual([m["id"] for m in data["data"]], ["openai/gpt-5", "openai/gpt-5-mini"])
        self.assertEqual((data["total"], data["matched"]), (len(MODELS), 2))
        limited = self.get("?limit=10").json()
        self.assertEqual((len(limited["data"]), limited["matched"]), (10, len(MODELS)))

    def test_unknown_provider_and_auth(self):
        self.assertEqual(self.client.get("/v1/providers/nope/models", headers=self.headers).status_code, 404)
        self.assertEqual(self.client.get("/v1/providers/opencode/models").status_code, 401)


if __name__ == "__main__":
    unittest.main()
