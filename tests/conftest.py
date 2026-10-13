import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from eegspeech.config import Config


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture(autouse=True)
def deterministic_threads():
    old=torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(11)
    yield
    torch.set_num_threads(old)
