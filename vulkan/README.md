Vulkan backend.

The standalone int-dot GEMV shaders live in ../kernels/vulkan/; the fork
integration (q1_0/q2_0 MMVQ) is ../patches/0003. The race in
ggml_vk_graph_optimize and the barrier-cost analysis are in ../README.md
and ../results/vulkan-*.txt.
