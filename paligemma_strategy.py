"""
PaliGemmaStrategy — rover-side client for the PaliGemma cloud navigation server.

Identical to CloudOmniVLAStrategy in every way except the strategy name.
The WebSocket protocol, reconnect logic, and waypoint-to-drive conversion
are shared with cloud_omnivla.

The cloud server is paligemma_cloud_server.py.

Usage:
    python rover_agent.py --strategy paligemma \\
        --cloud-server ws://<cloud-ip>:8766 \\
        --goal "Follow the crop row" --interval 1.0 \\
        --rover atlas --atlas-port /dev/ttyACM0
"""

from cloud_omnivla_strategy import CloudOmniVLAStrategy


class PaliGemmaStrategy(CloudOmniVLAStrategy):
    @property
    def name(self) -> str:
        return "paligemma"
