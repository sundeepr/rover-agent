"""
Abstract base class for rover navigation strategies, and AgentState —
the single shared-state object passed between the agent loop, strategy,
and web display.
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class AgentState:
    """
    All mutable state shared between the agent loop, the active strategy,
    and the web display.

    Each attribute is either protected by its own lock / event, or is
    written by exactly one thread (noted below).
    """

    # Mission counters — incremented by agent_loop under result_lock
    step: int = 0
    phase: int = 1

    # Latest LLM result + trajectory — written by strategy under result_lock
    latest_result: dict = field(default_factory=dict)
    trajectory: list = field(default_factory=list)  # list[dict]

    # LLM timing — written by strategy under result_lock
    llm_query_start: float = 0.0   # unix seconds; 0.0 = idle
    llm_response_s: float = 0.0    # elapsed time of last completed query

    # Live camera frame — written by agent_loop under raw_lock
    raw_frame: Optional[np.ndarray] = None
    raw_lock: threading.Lock = field(default_factory=threading.Lock)

    # Annotated LLM frame — written by strategy under llm_lock
    llm_frame: Optional[np.ndarray] = None
    llm_lock: threading.Lock = field(default_factory=threading.Lock)

    # Protects: latest_result, trajectory, step, phase, timing floats
    result_lock: threading.Lock = field(default_factory=threading.Lock)

    # Set while a query thread is running; checked by agent_loop before spawning
    query_in_flight: threading.Event = field(default_factory=threading.Event)

    # Set to pause the agent loop — no new queries are spawned while set.
    # The rover is stopped immediately when this is set via the web UI.
    paused: threading.Event = field(default_factory=threading.Event)


class NavigationStrategy(ABC):
    """
    Abstract navigation strategy. Subclasses encapsulate the entire
    decision-making pipeline for one step: acquiring context, querying a
    backend (LLM, local model, etc.), acting on the result, and updating
    shared AgentState.

    To add a new strategy:
      1. Create a new file (e.g. my_strategy.py) with a subclass.
      2. Implement run_query(), on_reset(), and name.
      3. Register it in rover_agent._build_strategy().
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and --strategy CLI arg."""
        ...

    @abstractmethod
    def run_query(
        self,
        state: AgentState,
        frame: np.ndarray,
        captures_dir: Path,
        rover_ctrl,
    ) -> None:
        """
        Execute one navigation query cycle in a fresh daemon thread.

        rover_ctrl is a RoombaController, AtlasController, or None (no hardware).
        All controllers share the same interface: navigate_to_waypoint(),
        uturn(), drive_raw(), stop().

        Contract:
        - MUST call state.query_in_flight.clear() in a finally block.
        - MUST write state.latest_result under state.result_lock.
        - MUST write state.llm_frame under state.llm_lock.
        - MUST append to state.trajectory under state.result_lock when a
          rank-1 waypoint exists.
        - MUST update state.phase and clear its internal buffer on
          phase1_complete, then call rover_ctrl.uturn() if available.
        - MUST reset state.llm_query_start to 0.0 (under result_lock)
          on both success and exception.
        """
        ...

    @abstractmethod
    def on_reset(self) -> None:
        """
        Hard reset — clear all internal buffers and accumulated state.
        Called when the agent is restarted externally.
        """
        ...
