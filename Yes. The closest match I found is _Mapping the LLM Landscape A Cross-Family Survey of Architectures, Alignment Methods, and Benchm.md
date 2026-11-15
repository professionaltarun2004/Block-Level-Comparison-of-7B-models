Yes. **The closest match I found is _Mapping the LLM Landscape: A Cross-Family Survey of Architectures, Alignment Methods, and Benchmark Performance_ (2026).** It surveys major LLM families and architectural advances, and reports a feature-based comparison of more than 50 architectures—overlapping with your cross-model architecture comparison, though it is broader and also covers alignment, benchmarks, and deployment topics. [^1]

**A useful related paper is _Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs_ (2025).** It compares how hidden size, the MLP-to-attention parameter allocation, and grouped-query attention relate to accuracy and inference cost. That is particularly relevant to your discussion of capacity allocation and resource trade-offs, but it is an empirical architecture-optimization study rather than a block-by-block comparison of LLaMA, Mistral, and Qwen. [^2]

So, there are related papers, but I did **not** find an exact match for your specific combination: a source-traced, functional-block comparison of those three roughly 7B model configurations, including tokenization, attention, FFN weights, and context policy. [^1][^2]

[^1]: Bhati et al., 2026. Mapping the LLM Landscape: A Cross-Family Survey of Architectures, Alignment Methods, and Benchmark Performance. Applied Informatics.

[^2]: Bian et al., 2025. Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs. arXiv.org.