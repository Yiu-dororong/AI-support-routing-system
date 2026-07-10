import os
import subprocess
import sys


if __name__ == "__main__":
    # Get path to app/main.py relative to this file's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(base_dir, "app", "main.py")

    # Run the streamlit application
    cmd = [sys.executable, "-m", "streamlit", "run", main_path] + sys.argv[1:]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        # Graceful exit without printing a giant Python traceback
        print("\n[Target] Streamlit application shutdown successfully.")
        sys.exit(0)
