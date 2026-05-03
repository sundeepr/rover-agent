"""
OmniVLAFullStrategy — rover-side strategy for full OmniVLA on a cloud GPU.

Thin subclass of CloudOmniVLAStrategy. The cloud server (omnivla_cloud_server.py)
runs the full OmniVLA model (VLA backbone + pose projector + action head).
All waypoint→drive conversion happens locally on the rover.

Usage:
    # On cloud GPU:
    python omnivla_cloud_server.py --model-path ./omnivla-original --host 0.0.0.0 --port 8765

    # On rover:
    python rover_agent.py --strategy omnivla_full \\
        --cloud-server ws://<cloud-ip>:8765 \\
        --goal "Follow the crop row" --interval 1.0 \\
        --rover atlas --atlas-port /dev/ttyACM0
"""

from cloud_omnivla_strategy import CloudOmniVLAStrategy


class OmniVLAFullStrategy(CloudOmniVLAStrategy):

    @property
    def name(self) -> str:
        return "omnivla_full"
