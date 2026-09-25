from __future__ import annotations

from pathlib import Path

import pytest

from skytrust.config import Settings, load_settings

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings()


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
