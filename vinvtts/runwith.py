"""Run a third-party script with cpulimit applied:  python runwith.py path/to/script.py [args...]"""
import os, runpy, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpulimit  # noqa: E402,F401

script = sys.argv[1]; sys.argv = sys.argv[1:]; sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
