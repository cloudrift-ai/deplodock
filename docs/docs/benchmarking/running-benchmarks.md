---
sidebar_position: 1
title: Running Benchmarks
description: Learn how to benchmark LLM models across GPU types and configurations.
---

# Running Benchmarks

Deplodock lets you run standardized benchmarks across different GPU types and model configurations,
so you can find the best performance-to-cost ratio before committing to a deployment.

## Metrics

Each benchmark captures:

- **Throughput** — tokens per second (TPS)
- **Time to first token (TTFT)**
- **Inter-token latency (ITL)**
- **GPU memory utilization**
- **Cost per 1M tokens** (based on CloudRift GPU pricing)

## Running a benchmark

Select a recipe or custom configuration, choose one or more GPU types, and launch.
Results are saved automatically and can be [shared as experiments](../experiments/overview).
