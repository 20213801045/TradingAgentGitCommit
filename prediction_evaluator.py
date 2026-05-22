"""Prediction accuracy evaluator — replaces audit-quality scoring with real-world accuracy.

Tracks every prediction, evaluates them against actual market data,
and provides a feedback loop for continuous improvement.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import yfinance as yf