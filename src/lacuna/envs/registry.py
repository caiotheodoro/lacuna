"""Every environment by name, for `scripts/run_episodes.py --env`.

`BACKOFFICE` points at the original back-office definitions, not copies, so
passing it is the same as passing no env at all (`tests/test_infra.py`).
"""

from __future__ import annotations

from lacuna.agents.llm import AGENT_SYSTEM
from lacuna.envs import bypass, tasks, tools
from lacuna.envs.base import Env
from lacuna.envs.infra_tools import INFRA
from lacuna.envs.kernel import Kernel
from lacuna.witnesses import from_call, observe

BACKOFFICE = Env(
    name="backoffice",
    tools=tools.TOOLS,
    action_tools=tools.ACTION_TOOLS,
    handlers=tools.handlers,
    normalise=from_call,
    observe=observe,
    make_episode=tasks.make_episode,
    make_kernel=Kernel,
    agent_system=AGENT_SYSTEM,
)

BACKOFFICE_BYPASS = Env(
    name="backoffice_bypass",
    tools=bypass.TOOLS,
    action_tools=tools.ACTION_TOOLS,
    handlers=bypass.handlers,
    normalise=from_call,
    observe=observe,
    make_episode=bypass.make_episode,
    make_kernel=Kernel,
    agent_system=AGENT_SYSTEM,
    script_tools=bypass.SCRIPT_TOOLS,
)

ENVS: dict[str, Env] = {e.name: e for e in (BACKOFFICE, INFRA, BACKOFFICE_BYPASS)}
