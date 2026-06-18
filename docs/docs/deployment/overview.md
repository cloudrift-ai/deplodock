---
sidebar_position: 1
title: Deployment Overview
description: Deploy optimized LLM models to GPU servers using vLLM or SGLang.
---

# Deployment Overview

Once you have a recipe or custom configuration ready, Deplodock handles provisioning the GPU server
and launching the inference backend.

## Backends

### vLLM
High-throughput inference engine with PagedAttention. Best for maximizing throughput
on single or multi-GPU setups.

### SGLang
Structured generation engine. Best for complex multi-turn workloads and structured outputs.

## Deployment flow

1. Select a recipe or custom configuration
2. Choose a GPU type (or let Deplodock recommend one based on your benchmark results)
3. Launch — Deplodock provisions the server and starts the inference endpoint
4. Get an OpenAI-compatible API endpoint to send requests to
