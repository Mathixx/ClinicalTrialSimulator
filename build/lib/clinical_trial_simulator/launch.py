"""
Launch the full stack (API + frontend) in one command.
Sets PYTHONPATH and loads .env, then starts uvicorn.
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    os.chdir(_REPO_ROOT)
    path_prepend = os.pathsep.join([str(_REPO_ROOT / "src"), str(_REPO_ROOT / "servers")])
    os.environ["PYTHONPATH"] = path_prepend + os.pathsep + os.environ.get("PYTHONPATH", "")

    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ROOT / ".env")
    except ImportError:
        pass

    import uvicorn
    print("Clinical Trial Simulator — http://127.0.0.1:8000")
    print("  Frontend: http://127.0.0.1:8000  |  API docs: http://127.0.0.1:8000/docs")
    uvicorn.run(
        "clinical_trial_simulator.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
