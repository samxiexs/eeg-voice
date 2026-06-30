from __future__ import annotations

from src.karaone_v91.model import ChannelMoEFrontendV91 as ChannelMoEFrontendV10
from src.karaone_v91.model import KaraOneV91ClusteredChannelMoEFlow
from src.karaone_v91.model import KaraOneV91Config


class KaraOneV10Config(KaraOneV91Config):
    """v10 keeps the v9.1 Channel-MoE architecture and changes training/eval policy."""


class KaraOneV10ClusteredChannelMoEFlow(KaraOneV91ClusteredChannelMoEFlow):
    """KaraOne v10 model wrapper.

    The architecture remains state-dict compatible with v9.1.  v10's
    EEG-specific behavior is enforced through stronger train losses, selection
    rules, diagnostic synthesis, and reporting.
    """


__all__ = ["ChannelMoEFrontendV10", "KaraOneV10Config", "KaraOneV10ClusteredChannelMoEFlow"]

