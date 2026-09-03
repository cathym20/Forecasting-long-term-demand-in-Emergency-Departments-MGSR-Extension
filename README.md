# Forecasting-long-term-demand-in-Emergency-Departments-MGSR-Extension

# Multi-Granular Forecasting of Emergency Department Demand

Replication and post-2022 extension of the Multi-Granular Stacked Regression (MGSR)
model proposed by James, Wood and Denholm (2023).

Author: Cathy Mariam Vijay
MSc Data Science dissertation, University of Bristol, 2025.

## Overview

This project reproduces the MGSR model on its original 2018-2019 CCG level data, then extends it into the post-2022 period at Sub-ICB Location level using Hospital Episode Statistics emergency admissions as the target, QOF chronic disease prevalence indicators from OHID Fingertips as the population health signal, and tuned Gradient Boosting learners instead of the paper's Random Forests.

The full dissertation is included as `dissertation.pdf`.

## Requirements

Python 3.9 or later. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This reads `pyproject.toml`, resolves versions from `uv.lock`, and creates a
local `.venv/` with everything installed.

To launch Jupyter:

```bash
uv run jupyter lab
```

## Project layout
