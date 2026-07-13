# TensorRT-LLM Candidate

TensorRT-LLM is a high-upside but high-setup-cost option.

Use it only after the vLLM and SGLang baselines are measured because this contest has short iteration loops and a fixed 120-request trace. The expected work is:

1. Build or convert a Qwen3.5-2B engine.
2. Set `max_seq_len` from exact tokenizer analysis, likely near 65,536 for this trace.
3. Tune `max_batch_size`, `max_num_tokens`, and KV cache free memory fraction.
4. Verify OpenAI-compatible behavior and streaming TTFT before submission.

