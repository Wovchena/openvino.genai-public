# Design Decision: ContinuousBatching Integration for OmniPipeline

**Decision Date**: 2026-02-09  
**Status**: Approved  
**Decision**: Use separate adapter pattern (OmniContinuousBatchingAdapter) rather than integrating ContinuousBatching directly into OmniPipeline

## Problem Statement

Should ContinuousBatching features be:
1. **Integrated directly** into OmniPipeline (single class handles both modes), or
2. **Kept separate** as an adapter class (dual-class architecture)?

## Context

### Existing Patterns in OpenVINO GenAI

Both major pipelines use the **adapter pattern**:

**LLMPipeline Architecture:**
```
LLMPipeline (public API)
  └── LLMPipelineImplBase (interface)
       ├── StatefulLLMPipeline (regular mode)
       └── ContinuousBatchingAdapter (batch mode)
            └── wraps ContinuousBatchingPipeline
```

**VLMPipeline Architecture:**
```
VLMPipeline (public API)
  └── VLMPipelineBase (interface)
       ├── VLMPipelineImpl (regular mode)
       └── VLMContinuousBatchingAdapter (batch mode)
            └── wraps ContinuousBatchingPipeline
```

### ContinuousBatchingPipeline Characteristics

- **Complexity**: 50+ member variables including scheduler, model runner, sampler
- **Request Management**: Queue-based system with awaiting/active requests
- **KV Cache**: Sophisticated paging with eviction algorithms
- **Variants**: Speculative decoding (Eagle3, PromptLookup)
- **Metrics**: Complex aggregation for batch operations
- **~3000 LOC** in implementation

### Mode Selection Logic

Both pipelines use factory pattern with automatic selection:
```cpp
if (device == "NPU")
    → StatefulPipeline (NPU doesn't support CB)
else if (explicit SchedulerConfig in properties)
    → ContinuousBatchingAdapter
else if (Paged Attention backend detected && model supports)
    → Try ContinuousBatchingAdapter (with exception fallback)
else
    → StatefulPipeline (default)
```

## Options Considered

### Option 1: Direct Integration (Single Class)

**Approach**: OmniPipeline class contains both regular and batch processing logic internally.

**Pros:**
- ✓ Simpler public API (one class)
- ✓ Less indirection
- ✓ Potentially easier for users to understand
- ✓ Could share some code between modes

**Cons:**
- ✗ **Breaks from established patterns** (LLMPipeline, VLMPipeline)
- ✗ **Complex internal state management** (mode switching, dual code paths)
- ✗ **Tight coupling** between regular and batch logic
- ✗ **Harder to maintain** as CB evolves independently
- ✗ **Larger class** with mixed responsibilities
- ✗ **Testing complexity** (must test both modes in single class)
- ✗ **Cannot leverage existing ContinuousBatchingPipeline** directly

### Option 2: Separate Adapter Pattern (Current Design) ✅

**Approach**: OmniPipeline delegates to either OmniPipelineImpl or OmniContinuousBatchingAdapter.

**Pros:**
- ✓ **Consistent with existing patterns** (VLMPipeline, LLMPipeline)
- ✓ **Clean separation of concerns** (regular vs batch logic isolated)
- ✓ **Leverages existing ContinuousBatchingPipeline** without modification
- ✓ **Independent evolution** of regular and batch modes
- ✓ **Easier testing** (test implementations independently)
- ✓ **Smaller, focused classes** (Single Responsibility Principle)
- ✓ **Backward compatibility** preserved (regular mode unaffected by CB changes)
- ✓ **Type safety** through polymorphism
- ✓ **Factory pattern** enables smart backend selection

**Cons:**
- ✗ Extra adapter layer (minor indirection overhead)
- ✗ More classes in architecture
- ✗ Metrics aggregation needed in adapter

## Decision

**Choose Option 2: Separate Adapter Pattern**

### Rationale

1. **Consistency is Critical**: OpenVINO GenAI has established the adapter pattern as the standard for CB integration. Deviating would create confusion and maintenance burden.

2. **CB is Fundamentally Different**: ContinuousBatching represents a **different execution model**:
   - Regular: Single request, sequential processing
   - CB: Multiple requests, dynamic batching, request queuing
   
   These are better represented as separate implementations rather than conditional logic within a single class.

3. **Evolution Independence**: CB and regular modes evolve at different rates:
   - CB receives optimizations for batching, scheduling, cache management
   - Regular mode optimizes for latency, simplicity
   
   Separate classes allow independent evolution without risk of regression.

4. **Proven Pattern**: VLMPipeline successfully uses this pattern for multimodal models. OmniPipeline extends this to more modalities but should maintain the same architecture.

5. **Maintainability**: Future developers expect consistency. When they see VLMPipeline and LLMPipeline using adapters, they'll expect OmniPipeline to follow suit.

## Implementation

### Class Structure

```cpp
// Public API
class OmniPipeline {
protected:
    class OmniPipelineBase;  // Abstract interface
    class OmniPipelineImpl;  // Regular mode implementation
    class OmniContinuousBatchingAdapter;  // Batch mode implementation
    
private:
    std::unique_ptr<OmniPipelineBase> m_pimpl;  // PIMPL pattern
};

// Adapter wraps ContinuousBatchingPipeline
class OmniContinuousBatchingAdapter : public OmniPipelineBase {
private:
    ContinuousBatchingPipeline m_impl;  // Composition
};
```

### Mode Selection

Following VLMPipeline pattern (src/cpp/src/visual_language/pipeline.cpp:628-705):

```cpp
OmniPipeline::OmniPipeline(const std::filesystem::path& models_path,
                           const std::string& device,
                           const ov::AnyMap& properties) {
    auto [props, attention_backend] = utils::extract_attention_backend(properties);
    
    if (device == "NPU") {
        // NPU doesn't support ContinuousBatching
        m_pimpl = std::make_unique<OmniPipelineImpl>(models_path, device, props);
    } else {
        // Try ContinuousBatching if requested or available
        if (utils::explicitly_requires_paged_attention(properties)) {
            auto [plugin_props, scheduler_config] = 
                utils::extract_scheduler_config(props, 
                    utils::get_latency_oriented_scheduler_config());
            m_pimpl = std::make_unique<OmniContinuousBatchingAdapter>(
                models_path, scheduler_config, device, plugin_props);
        } else if (attention_backend == PA_BACKEND && !requires_sdpa(models_path)) {
            try {
                auto [plugin_props, scheduler_config] = 
                    utils::extract_scheduler_config(props,
                        utils::get_latency_oriented_scheduler_config());
                #if defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_ARM64)
                m_pimpl = std::make_unique<OmniContinuousBatchingAdapter>(
                    models_path, scheduler_config, device, plugin_props);
                #endif
            } catch (ov::Exception&) {
                // Fallback to regular mode
            }
        }
        
        if (m_pimpl == nullptr) {
            m_pimpl = std::make_unique<OmniPipelineImpl>(models_path, device, props);
        }
    }
}
```

## Consequences

### Positive

- **Consistency**: Aligns with VLMPipeline and LLMPipeline
- **Maintainability**: Clear separation makes code easier to understand and modify
- **Testability**: Each implementation can be tested independently
- **Flexibility**: Easy to add new execution modes (e.g., streaming-optimized)
- **Reliability**: Regular mode isolated from CB changes

### Negative

- **Adapter overhead**: Minor performance cost from virtual dispatch (negligible in practice)
- **More files**: Additional header files needed (but well-organized)
- **Learning curve**: Developers must understand dual-mode architecture

### Mitigations

- Document architecture clearly (this file, README.md, API_DESIGN_SUMMARY.md)
- Provide examples for both modes in samples
- Follow PIMPL pattern to hide implementation details from users
- Use factory pattern to make mode selection automatic and transparent

## References

### Existing Implementations
- `src/cpp/src/llm/pipeline.cpp` (lines 196-233)
- `src/cpp/src/llm/pipeline_continuous_batching_adapter.hpp`
- `src/cpp/src/visual_language/pipeline.cpp` (lines 628-705)
- `src/cpp/src/visual_language/continuous_batching_adapter.hpp`

### Design Principles
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- Single Responsibility Principle (SRP)
- Dependency Inversion Principle (DIP)
- Factory Pattern for object creation
- Adapter Pattern for interface compatibility

## Alternatives Rejected

### Hybrid Approach
**Idea**: Integrate simple batch processing directly, use adapter for advanced features.

**Rejected Because**: Creates confusion about when to use which approach. Better to have clear distinction.

### Template-Based Approach
**Idea**: Use templates to parameterize mode at compile time.

**Rejected Because**: 
- Loses runtime flexibility
- Increases compilation time
- Complicates shared library distribution
- Not compatible with existing CB infrastructure

### Plugin System
**Idea**: Make CB a loadable plugin.

**Rejected Because**:
- Adds complexity without clear benefit
- CB is core functionality, not optional extension
- Increases deployment complexity

## Approval

This design decision has been documented and implemented in the Omni API proposal.

**Approver**: OpenVINO GenAI Team  
**Implementation**: API design phase complete, ready for implementation
