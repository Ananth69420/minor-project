"""Start the fleet simulator and open its dashboard."""

import os
import sys
import argparse
import webbrowser
import threading
import time
import uvicorn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FLEET_RL_DIR = os.path.join(CURRENT_DIR, "fleet-rl")
if FLEET_RL_DIR not in sys.path:
    sys.path.insert(0, FLEET_RL_DIR)

from server import app

def open_browser(url, delay=1.2):
    def _open():
        time.sleep(delay)
        print(f"🌐 Opening live simulation in browser: {url}")
        webbrowser.open(url)
    t = threading.Thread(target=_open, daemon=True)
    t.start()

def main():
    parser = argparse.ArgumentParser(description="Fleet Dispatch Lab server")
    parser.add_argument("--port", type=int, default=8000, help="HTTP/WebSocket server port (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    print("\n" + "=" * 80)
    print(" FLEET DISPATCH LAB")
    print(" Reinforcement Learning-Based Dispatch for Autonomous Fleet Management")
    print("=" * 80)
    print(f" Environment: V2 (16x16 grid, 8 vehicles, 300 steps)")
    print(f" Policy: tabular Q-learning with SMDP updates")
    print(f" Dynamic Passenger Generation & Manhattan Shortest-Path Dispatching")
    print("-" * 80)
    print(f" Web Dashboard:       {url}")
    print(f" REST API:            {url}/api/status")
    print(f" WebSocket Stream:    ws://localhost:{args.port}/ws")
    print("=" * 80 + "\n")

    if not args.no_browser:
        open_browser(url)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()

