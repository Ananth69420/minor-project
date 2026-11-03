"""
Root delegator for server.py.
Allows launching the FastAPI live computational simulation server directly from root:
python server.py
"""

import os
import sys
import importlib.util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FLEET_RL_DIR = os.path.join(CURRENT_DIR, "fleet-rl")
SERVER_PATH = os.path.join(FLEET_RL_DIR, "server.py")

spec = importlib.util.spec_from_file_location("fleet_server", SERVER_PATH)
fleet_server = importlib.util.module_from_spec(spec)
sys.modules["fleet_server"] = fleet_server
spec.loader.exec_module(fleet_server)

app = fleet_server.app
main = fleet_server.main

if __name__ == "__main__":
    main()
