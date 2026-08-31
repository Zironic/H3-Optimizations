# H3 Optimizations example workflows

- 01_h3_memory_optimization.json — stock MiniMax H3 structure + H3 Memory Optimization defaults.
- 02_h3_recommended_15pct.json — stock MiniMax H3 subgraph + Memory Optimization + simple Sparse Attention at 15% KV with the default early ramp. It starts at 50% KV and gradually returns to 15% while spending 12 additional percentage points per sampler step on average. Only Video attention budget is exposed as an H3 control on the main graph.
- 03_h3_advanced_15_68_ramp.json — flattened workflow + Memory Optimization + Advanced Sparse Attention at 15% middle, an 8-step Ramp starting at 68.33% Early KV, no late override, Kitchen INT8 backend. On a 20-step sampler this matches the simple node default cumulative video-attention budget while concentrating it more heavily in the earliest steps.

Source structure: Comfy-Org/workflow_templates/templates/video_minimax_h3_t2v.json (current stock MiniMax H3 template when generated).
