# Copyright 2026 Aayush Chawla
# SPDX-License-Identifier: Apache-2.0

"""Tests for the retired-model filter on GET /settings/available-models (issue #25).

LiteLLM's static `models_by_provider` list still offered `gemini-2.0-flash`
after Google shut it down, so a fresh setup could be pointed at a model whose
every call 404s. The filter reads the `deprecation_date` LiteLLM already
publishes rather than hardcoding a per-provider denylist.
"""

import sys
import types
from datetime import date, timedelta
from unittest.mock import patch

from laya.api.settings_api import _fetch_models_for_provider, _is_retired_model


def _years_ago(n: int) -> str:
    return str(date.today() - timedelta(days=365 * n))


def _years_ahead(n: int) -> str:
    return str(date.today() + timedelta(days=365 * n))


def _with_cost(cost: dict) -> object:
    """Patch litellm.model_cost with a stand-in cost table."""
    return patch("litellm.model_cost", cost)


class TestIsRetiredModel:
    def test_past_deprecation_date_is_retired(self):
        # The reported model: Google retired it, LiteLLM's table knows so.
        with _with_cost({"gemini/gemini-2.0-flash": {"deprecation_date": "2026-03-31"}}):
            assert _is_retired_model("gemini/gemini-2.0-flash") is True

    def test_no_deprecation_date_is_live(self):
        with _with_cost({"gemini/gemini-2.5-pro": {}}):
            assert _is_retired_model("gemini/gemini-2.5-pro") is False

    def test_future_deprecation_date_is_live(self):
        with _with_cost({"gemini/gemini-3-pro-preview": {"deprecation_date": _years_ahead(1)}}):
            assert _is_retired_model("gemini/gemini-3-pro-preview") is False

    def test_explicit_null_date_is_live(self):
        with _with_cost({"gemini/gemini-2.5-flash": {"deprecation_date": None}}):
            assert _is_retired_model("gemini/gemini-2.5-flash") is False

    def test_unknown_model_is_kept(self):
        # A missing cost entry is not evidence of retirement, so custom /
        # self-hosted / brand-new models must survive the filter.
        with _with_cost({}):
            assert _is_retired_model("lmstudio-local/qwen3-8b") is False

    def test_today_is_not_yet_retired(self):
        # Deprecation is a boundary: a model dated today still serves today.
        with _with_cost({"gemini/x": {"deprecation_date": str(date.today())}}):
            assert _is_retired_model("gemini/x") is False

    def test_bare_year_is_parsed(self):
        # A few upstream tables are year-granular; treat the year as EOY.
        with _with_cost({"vendor/old": {"deprecation_date": _years_ago(2)[:4]}}):
            assert _is_retired_model("vendor/old") is True

    def test_unparseable_date_is_kept(self):
        # Never hide a model because we failed to read its metadata.
        for raw in ("soon", "", "not-a-date", "2026-13-45"):
            with _with_cost({"vendor/mystery": {"deprecation_date": raw}}):
                assert _is_retired_model("vendor/mystery") is False

    def test_non_dict_entry_is_kept(self):
        with _with_cost({"vendor/weird": "nope"}):
            assert _is_retired_model("vendor/weird") is False

    def test_missing_cost_table_degrades_to_keep(self):
        # A LiteLLM that doesn't expose `model_cost` at all must not take the
        # whole model list down -- the filter has to fail open.
        with patch.dict(sys.modules, {"litellm": types.ModuleType("litellm")}):
            assert _is_retired_model("gemini/gemini-2.0-flash") is False

    def test_applies_to_any_provider(self):
        # Not Gemini-specific: the filter is driven by the date alone.
        with _with_cost(
            {
                "openai/gpt-3.5-turbo-0125": {"deprecation_date": _years_ago(1)},
                "openai/gpt-4o": {},
            }
        ):
            assert _is_retired_model("openai/gpt-3.5-turbo-0125") is True
            assert _is_retired_model("openai/gpt-4o") is False


class TestFetchModelsForProvider:
    """End-to-end over `_fetch_models_for_provider`, with LiteLLM stubbed."""

    def _run(self, models, *, dynamic_fails: bool):
        cost_map = {
            "gemini/gemini-2.0-flash": {"deprecation_date": "2026-03-31"},
            "gemini/gemini-2.5-flash": {},
            "gemini/gemini-2.5-pro": {},
            "gemini/gemini-2.0-flash-001": {"deprecation_date": "2026-03-31"},
        }
        with (
            patch("laya.api.settings_api.get_api_key", return_value="k"),
            _with_cost(cost_map),
            patch("litellm.models_by_provider", {"gemini": set(models)}),
            patch(
                "litellm.get_valid_models",
                side_effect=RuntimeError("no network") if dynamic_fails else (lambda **kw: list(models)),
            ),
        ):
            return [m["id"] for m in _fetch_models_for_provider("google")]

    def test_static_fallback_drops_retired_models(self):
        # The exact failure in #25: the static fallback still offers a model
        # that 404s, so setup happily points the Router at it.
        models = ["gemini/gemini-2.0-flash", "gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro"]
        assert self._run(models, dynamic_fails=True) == [
            "gemini/gemini-2.5-flash",
            "gemini/gemini-2.5-pro",
        ]

    def test_dynamic_fetch_drops_retired_models(self):
        # Applied to the dynamic path too, so a live listing that still
        # carries a dead entry is filtered as well.
        models = ["gemini/gemini-2.0-flash", "gemini/gemini-2.5-pro"]
        assert self._run(models, dynamic_fails=False) == ["gemini/gemini-2.5-pro"]

    def test_dated_variants_are_dropped_too(self):
        models = ["gemini/gemini-2.0-flash-001", "gemini/gemini-2.5-pro"]
        assert self._run(models, dynamic_fails=True) == ["gemini/gemini-2.5-pro"]

    def test_non_chat_filter_still_applies(self):
        # The pre-existing embed/TTS/image exclusions must survive.
        models = [
            "gemini/gemini-2.0-flash",
            "gemini/text-embedding-004",
            "gemini/gemini-2.5-pro",
        ]
        assert self._run(models, dynamic_fails=True) == ["gemini/gemini-2.5-pro"]

    def test_output_is_sorted(self):
        # Pre-existing contract: the endpoint returns a sorted list.
        models = ["gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash", "gemini/gemini-3-flash-preview"]
        got = self._run(models, dynamic_fails=True)
        assert got == sorted(got)
