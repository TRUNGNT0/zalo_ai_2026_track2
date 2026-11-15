"""Compatibility module: aliases 'app' to 'datamate'."""
import sys
import datamate

# Alias datamate modules under app.* for backwards compatibility
sys.modules["app"] = datamate
from datamate import *  # noqa: F401, F403
