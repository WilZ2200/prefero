# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Compatibility entry point for older Streamlit launch commands."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("Home.py")), run_name="__main__")
