# Current Findings — Reset 17/07/2026

## 1. Trạng Thái Hiện Tại

**Chưa có submission hợp lệ cho vòng mới. Không nộp root compose hiện tại.**

Repo vẫn là workspace của vòng Qwen3.5 đã bị reset:

| Artifact | Trạng thái |
| --- | --- |
| `docker-compose.yml` | Sai model (`Qwen3.5-2B`), dùng GHCR, chứa Qwen-only flags |
| `input/docker-compose-baseline.yml` | Baseline cũ |
| `input/trace-round1.jsonl` | Trace cũ 120 request, không phải trace mới 330 request |
| `src/qwen35_serving_bench/scoring.py` | Bounds cũ TTFT 100–1500, TPOT 20–45 |
| `configs/**` | Historical snapshots của leaderboard đã reset |
| `custom_runtime/**` | Thử nghiệm Qwen cũ, không phải candidate mới |

Best score vòng mới: **chưa có**. Mọi score 15–65, TTFT/TBT và SHA cũ không được dùng làm
baseline, projection hoặc promote gate.

## 2. Ground Truth Mới

| Thuộc tính | Vòng reset |
| --- | --- |
| Model | `LiquidAI/LFM2.5-1.2B-Instruct` |
| Framework | Chỉ vLLM |
| Scored workload | 330 request; official text mô tả 70 multi-turn conversations |
| Warm-up | 15 primer conversations, không tính điểm; quan hệ với tổng 70 cần trace xác nhận |
| Input/output | Khoảng 4k input token, tối đa 200 output token |
| Arrival | Poisson timeline deterministic |
| Hardware | H200 MIG 18 GB, 3 CPU, 8 GB RAM |
| Host | Ubuntu 24.04, driver 590.x, CUDA 13.x support |
| Online metric | ERS, TTFT 10–400 ms và TPOT 1–10 ms |
| Accuracy | Chỉ post-online trên tối đa 5 submissions tự chọn |
| Registry contract | Public Docker Hub image, immutable digest |

Public trace `trace_grading_public.jsonl` chưa có trong repo tại thời điểm reset. Không tối ưu
scheduler, batching hoặc context length dựa trên `input/trace-round1.jsonl` cũ.

## 3. Ý Nghĩa Của Scoring Mới

Bounds mới khắt khe hơn rất nhiều; TPOT trên 10 ms nhận 0 cho nửa điểm decode và TTFT trên
400 ms nhận 0 cho nửa điểm prefill/queue. Ví dụ nếu mọi request thành công:

| TTFT | TPOT | Request score xấp xỉ |
| ---: | ---: | ---: |
| 200 ms | 5 ms | 28.58/100 |
| 100 ms | 5 ms | 45.02/100 |
| 100 ms | 3 ms | 59.83/100 |
| 50 ms | 3 ms | 70.52/100 |
| 25 ms | 2 ms | 85.73/100 |
| 10 ms | 1 ms | 100/100 |

Vì hai component có trọng số ngang nhau, không thể bù một TPOT vượt ceiling chỉ bằng TTFT tốt
hoặc ngược lại. Failure/timeout/0-token vẫn phá trực tiếp ERS.

Primer 15 conversation làm warm state quan trọng: image/model load phải hoàn tất trước benchmark,
nhưng cache/kernel warm-up trong primer có thể ảnh hưởng request được chấm. Phải tách metric primer
và scored requests trong replay tooling mới.

## 4. Những Finding Cũ Bị Loại

Không mang sang vòng mới:

- “Budget 2048 là optimum”, fixed cohort 20, burst 6×20 và mọi kết luận từ trace 120 request.
- Common prefix 6,396 token, Qwen chat lengths 12k–27k và max-model-len 32k vì workload cũ.
- `--mamba-ssm-cache-dtype`, `--language-model-only`, Qwen GDN/Mamba/MTP findings.
- FP8 accuracy observations trên Qwen hoặc portal accuracy drop cũ.
- vLLM 0.24/0.25 performance projection cho Qwen3.5.
- Các score 47.52, 59.10, 65.06 và mọi promote threshold dựa trên chúng.
- GHCR package hiện tại; đề mới yêu cầu public Docker Hub.

Các file snapshot có thể còn trong git để audit lịch sử nhưng không được nhắc như candidate canonical.

## 5. Bài Học Quy Trình Còn Dùng Được

Chỉ giữ các lesson không phụ thuộc model/trace:

1. Pin image digest; tag và metadata không thay thế exact image verification.
2. Build `linux/amd64`, smoke-import đúng API module, kiểm tra public pull và Compose trước portal.
3. Giữ entrypoint chính thức; runner trước đây từng override backend entrypoint nên backend switch
   không được giả định chỉ từ Compose.
4. Thay một package/biến mỗi experiment; snapshot root trước khi sửa.
5. Không promote micro-change trong vùng noise. Tie-break chính thức đã coi 1–2 điểm là nhiễu.
6. Concurrency cap, custom scheduler và CUDA graph micro-tuning có thể làm latency xấu; chỉ mở khi
   trace/profiler mới chứng minh bottleneck tương ứng.
7. Package optional bị broken vẫn có thể crash import toàn server; smoke test phải đi qua exact
   `python3 -m vllm.entrypoints.openai.api_server` path.
8. Không nộp rollback chỉ để lấy lại điểm: leaderboard giữ best lịch sử. Mỗi lượt portal phải là
   candidate mới có hypothesis và expected metric signature rõ.

## 6. Migration Gate Trước Candidate Đầu Tiên

- [ ] Lưu/verify `trace_grading_public.jsonl` mới.
- [ ] Thay model và served name trong baseline/root Compose.
- [ ] Chuyển image sang public Docker Hub digest.
- [ ] Loại toàn bộ Qwen-only flags.
- [ ] Cập nhật scorer constants: TTFT `10/400`, TPOT `1/10`, gamma `2`, weight `0.5`.
- [ ] Cập nhật tests theo công thức mới.
- [ ] Cập nhật replay để phân biệt primer conversations và 330 scored requests.
- [ ] Phân tích arrival, turns, input/output token distributions từ trace mới.
- [ ] Boot BF16 baseline, healthcheck và one-request streaming smoke.
- [ ] Chạy local/proxy replay và lưu p50/p95 TTFT, mean/p50/p95 TPOT, failure count, ERS.

Cho tới khi checklist này hoàn tất, không có file nào được gọi là “ready-to-submit”.

## 7. Lộ Trình Tối Ưu Mới

### Phase A — Correct Baseline

1. Bắt đầu từ image/sample vLLM `v0.22.1` chính thức và BF16.
2. Giữ prefix caching như sample.
3. Dùng `--max-model-len=32768` ở baseline đầu để giảm risk; chỉ hạ sau khi trace/tokenizer
   chứng minh max total context và chat-template headroom.
4. Ghi baseline ERS, p95 TTFT, TPOT và failures trước mọi tuning.

### Phase B — Low-Risk Serving Knobs

Sau khi có trace thật, A/B từng biến:

- `max-num-batched-tokens` theo actual prefill/decode overlap.
- `gpu-memory-utilization` với memory headroom đo được.
- `max-model-len` 8k/16k chỉ khi total context được chứng minh an toàn.
- Prefix caching on/off để đo multi-turn reuse thật.
- CUDA graph/default batching chỉ khi profiler cho thấy launch/padding overhead.

Không hardcode concurrency cap trước khi biết số conversation đồng thời theo timeline mới.

### Phase C — Accuracy-Risk Candidates

- Online FP8 weight quantization.
- FP8/INT8 KV cache nếu LFM + vLLM version hỗ trợ.
- Speculative decoding chỉ khi proposer/model support và burst replay cho throughput tốt hơn.

Do GPQA chỉ chạy trên tối đa 5 bài sau vòng online, luôn giữ ít nhất một BF16 accuracy-safe
submission trong finalist pool. Quantized candidate ERS cao không tự động thay thế BF16.

### Phase D — Runtime/Kernel Ceiling

Chỉ mở sau profiler:

- vLLM/CUDA 13 version A/B bằng exact Docker Hub digest.
- FlashAttention/FlashInfer backend A/B.
- Custom Triton/CUDA fusion hoặc scheduler.
- Disaggregated prefill/decode nếu hợp lệ trên một MIG và có evidence rõ.

Custom engineering phải có expected gain lớn hơn vùng noise 1–2 điểm và rollback sạch.

## 8. Submission Và Finalist Policy

Mỗi submission phải ghi:

- Compose SHA-256 và image digest.
- Exact hypothesis và chỉ một package thay đổi.
- ERS, failed count, TTFT p50/p95, TPOT mean/p50/p95.
- Primer behavior và 330 scored request count.
- Expected accuracy risk: BF16 / weight quant / KV quant / speculative.
- Promote, hold-for-finalist hoặc reject decision.

Strong promote gate: ERS tăng ít nhất `0.02` (2 điểm trên thang 100) hoặc cải thiện tie-break rõ
ràng mà không tăng failures. Thay đổi nhỏ hơn được ghi nhận nhưng không trở thành base mặc định.

Duy trì tối đa năm loại finalist, không nhất thiết là năm ERS cao nhất tuyệt đối:

1. BF16 accuracy-safe.
2. Best overall ERS.
3. Best p95 TTFT.
4. Best generation speed/TPOT.
5. Best quantized candidate có accuracy risk chấp nhận được.

## 9. Candidate Tracker

| ID | Status | Package | Gate trước portal |
| --- | --- | --- | --- |
| `NEW-BASELINE-BF16` | `blocked-on-migration` | Official vLLM baseline + LFM2.5, no Qwen flags | Trace/scorer migration, Docker Hub digest, local boot |
| `CONTEXT-RIGHTSIZE` | `pending` | Baseline + one max-model-len change | Tokenizer-derived max context + headroom |
| `BATCH-TOKEN-A/B` | `pending` | Baseline + one batching budget | New trace concurrency/prefill analysis |
| `FP8-WEIGHT` | `pending` | Baseline + online FP8 | Clean ERS win; retain BF16 finalist |
| `KV-CACHE-QUANT` | `pending` | Baseline + one KV dtype | LFM/vLLM support and memory measurement |
| `RUNTIME-VERSION-A/B` | `pending` | Exact Docker Hub vLLM/CUDA version change | linux/amd64 import, boot and public pull proof |

Immediate next step: migrate code and baseline, not tune the legacy Qwen compose.
