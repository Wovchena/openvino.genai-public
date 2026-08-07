---
sidebar_position: 3
---

# Continuous Batching

> **Note:** This page is a work in progress.

Text generation pipelines such as `LLMPipeline` and `VLMPipeline` instantiate continuous batching (Paged Attention) by default on CPU and GPU. The stateful backend is selected instead when:

- the device is NPU,
- `ATTENTION_BACKEND="SDPA"` is requested explicitly,
- the model architecture is known to require SDPA,
- the use case is implemented for the SDPA backend only,
- the build targets an architecture other than x86_64 or ARM64.

If continuous batching construction fails, the model is re-read and the stateful pipeline is constructed instead.
