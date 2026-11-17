import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from hardened_llm_gateway.pricing import PriceTable
from hardened_llm_gateway.providers import Usage

USAGE = Usage(prompt_tokens=1_000, completion_tokens=500, total_tokens=1_500)
DAY = datetime.date(2026, 11, 17)


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "prices.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_cost_is_tokens_times_price_per_million(tmp_path: Path) -> None:
    table = PriceTable.load(write(tmp_path, '[models."groq/m1"]\ninput = 0.10\noutput = 0.40\n'))
    # 1,000 x 0.10 + 500 x 0.40 = 300 USD per million tokens -> 0.0003 USD
    assert table.cost_usd("groq/m1", USAGE, DAY) == pytest.approx(0.0003)


def test_unknown_model_has_no_cost(tmp_path: Path) -> None:
    table = PriceTable.load(write(tmp_path, "[models]\n"))
    assert table.cost_usd("groq/unknown", USAGE, DAY) is None


def test_scheduled_price_changes_start_on_their_date(tmp_path: Path) -> None:
    table = PriceTable.load(
        write(
            tmp_path,
            '[models."gemini/m1"]\ninput = 1.0\noutput = 1.0\n'
            "changes = [{ from = 2026-12-01, input = 2.0, output = 2.0 },"
            " { from = 2027-01-01, input = 3.0, output = 3.0 }]\n",
        )
    )

    def cost_on(day: datetime.date) -> float | None:
        return table.cost_usd("gemini/m1", USAGE, day)

    assert cost_on(datetime.date(2026, 11, 30)) == pytest.approx(0.0015)
    assert cost_on(datetime.date(2026, 12, 1)) == pytest.approx(0.0030)
    assert cost_on(datetime.date(2027, 6, 1)) == pytest.approx(0.0045)


def test_negative_price_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        PriceTable.load(write(tmp_path, '[models."groq/m1"]\ninput = -1\noutput = 0\n'))


def test_shipped_price_table_loads_and_names_targets() -> None:
    table = PriceTable.load()
    assert "groq/openai/gpt-oss-20b" in table
    assert "gemini/gemini-2.5-flash-lite" in table
    # 1,000 x 0.075 + 500 x 0.30 = 225 USD per million tokens
    assert table.cost_usd("groq/openai/gpt-oss-20b", USAGE, DAY) == pytest.approx(0.000225)
