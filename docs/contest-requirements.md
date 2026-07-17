# Yêu Cầu Chính Thức — Vòng Online Reset 17/07/2026

Tài liệu này chuẩn hóa hai nguồn chính thức được lưu dưới dạng raw textual snapshot tại
`docs/official-source-2026-07-17.txt`. Nếu có khác biệt, thông báo mới nhất của BTC và bản
nguyên văn là source of truth; file này phải được cập nhật ngay.

## 1. Nhiệm Vụ

Triển khai và tối ưu một LLM inference server bằng **vLLM** cho:

```text
LiquidAI/LFM2.5-1.2B-Instruct
```

Model được BTC mount tại `/model`. Không train, fine-tune, thay weights/tokenizer hoặc pull
model động khi serving.

## 2. Workload

| Item | Giá trị chính thức |
| --- | --- |
| Loại workload | Multi-turn conversation, causal giữa các lượt |
| Số hội thoại | 70 trong workload được mô tả; quan hệ với 15 primer cần xác nhận từ trace |
| Arrival | Poisson nhưng đã đóng băng thành timestamp deterministic |
| Primer | 15 hội thoại warm-up, không tính điểm |
| Request được chấm | 330 |
| Context input tối đa | Khoảng 4k token / 12k ký tự |
| Output tối đa | 200 token |
| Trace công khai | `trace_grading_public.jsonl`, chỉ arrival và token counts |
| Trace chấm | Prompt thật do BTC giữ và gửi lúc benchmark |

Không được thay arrival timestamp, phá tính nhân-quả multi-turn hoặc suy luận nội dung prompt
ẩn từ trace công khai.

Working assumption để lập tooling: 15 primer chạy trước 70 conversation/330 request được chấm.
Tuy nhiên câu chữ chính thức có thể được đọc là 15 primer nằm trong tổng 70; parser phải lấy
trace mới làm ground truth và docs sẽ cập nhật ngay khi file được cung cấp.

## 3. Hạ Tầng Đánh Giá

| Item | Contract |
| --- | --- |
| GPU | 1 NVIDIA H200 MIG, khoảng 18 GB VRAM |
| CPU | 3 cores |
| RAM | 8 GB |
| Host OS | Ubuntu 24.04 LTS |
| NVIDIA driver | 590.x |
| CUDA host support | CUDA 13.x |
| Framework | Chỉ vLLM |
| Endpoint | OpenAI-compatible `/v1/chat/completions` |
| Listen | `0.0.0.0:8000` |
| Served model name | `LFM2.5-1.2B-Instruct` |

## 4. Điểm ERS Online

Với `N` request được chấm:

```text
ERS = (1 / N) × Σ S_request,i
```

Request lỗi, timeout hoặc trả 0 token nhận `S_request = 0`. Request thành công:

```text
S_request = 0.5 × s_ttft + 0.5 × s_tpot

s_ttft = clamp((400 - TTFT_ms) / (400 - 10), 0, 1)²
s_tpot = clamp((10 - TPOT_mean_ms) / (10 - 1), 0, 1)²
```

| Tham số | Giá trị |
| --- | ---: |
| TTFT floor | 10 ms |
| TTFT ceiling | 400 ms |
| TPOT floor | 1 ms |
| TPOT ceiling | 10 ms |
| Gamma | 2 |
| TTFT weight | 0.5 |
| TPOT weight | 0.5 |

Leaderboard online chỉ dùng ERS của từng submission. Không chạy GPQA trên mỗi lượt online.

## 5. Accuracy Gate Sau Vòng Online

Sau vòng online, đội chọn thủ công tối đa **5 submissions đã nộp**. BTC hậu kiểm tính hợp
lệ và chạy GPQA Diamond full. Không được thay image/digest sau khi chọn.

```text
Δ = accuracy_baseline_BF16 - accuracy_submission

f(Δ) = 1.0                         nếu Δ <= 0.10
f(Δ) = 1.0 - (Δ - 0.10) / 0.06    nếu 0.10 < Δ < 0.16
f(Δ) = 0.0                         nếu Δ >= 0.16

Score = 100 × ERS × f(Δ)
```

Baseline accuracy tham chiếu mặc định được nêu là `0.4`. Điểm đội là Score tốt nhất trong
các submission còn hợp lệ sau hậu kiểm và GPQA.

Một chỗ trong bản “Đề bài & Quy định” ghi `ERC` ở công thức cuối; phần định nghĩa xung quanh
và bản “Tổng quan” đều dùng `ERS`. Canonical interpretation hiện tại là `ERS` cho tới khi BTC
xác nhận khác.

## 6. Không Gian Tối Ưu

Được phép, miễn trung thực và tương thích vLLM:

- Online weight quantization.
- KV cache quantization, Paged Attention, prefix/semantic caching, CPU/NVMe offload.
- Dynamic/continuous batching, speculative decoding, memory-aware scheduling.
- Custom CUDA/Triton kernels, FlashAttention/FlashInfer, CUDA Graphs, memory layout.
- Disaggregated prefill/decode nếu artifact vẫn đáp ứng contract và tài nguyên được cấp.

## 7. Artifact Nộp

- Đóng gói toàn bộ giải pháp thành một public Docker image trên **Docker Hub** cá nhân/tổ chức.
- Nộp một `docker-compose.yml` chứa chính xác image và lệnh chạy.
- Không dùng `build:`, secret hoặc volume local ngoài `/model` do BTC mount.
- Pin image bằng immutable digest; không tráo hoặc mutate image sau khi nộp.
- Baseline image được BTC cung cấp từ `vllm/vllm-openai:v0.22.1`; nguồn chính thức cũng trỏ
  tới Docker Hub image digest page `sha256-55c9bcee9fc66644b139fddae8a7a03e4c0c8a25ab5c64b0ce614554a8abf5d5`.
- Mẫu chính thức yêu cầu giữ nguyên entrypoint:

```yaml
entrypoint:
  - python3
  - -m
  - vllm.entrypoints.openai.api_server
```

Các argument bắt buộc phải giữ:

```text
--model=/model
--served-model-name=LFM2.5-1.2B-Instruct
--host=0.0.0.0
--port=8000
```

## 8. Anti-Cheating

Nghiêm cấm:

- Pre-bake/hardcode đáp án hoặc nhận diện hidden prompt.
- Dual-path giữa latency benchmark và accuracy evaluation.
- Gaming metrics, output rỗng hoặc cắt sinh trái phép.
- Gọi mạng ngoài trong inference.
- Sửa tokenizer/weights hoặc làm bẩn tài nguyên.
- Tráo image sau submission.

Giải pháp phải là serving system tổng quát có thể phục vụ người dùng thật.

## 9. Tie-Break, Re-grade Và Khiếu Nại

Khi chênh lệch nằm trong vùng nhiễu khoảng 1–2 điểm, thứ tự tie-break:

1. Accuracy drop thấp hơn.
2. p95 TTFT thấp hơn.
3. Tốc độ sinh văn bản cao hơn.
4. Submission hợp lệ sớm hơn.

BTC có thể chấm lại nhiều lần trên đúng image đã chốt và lấy median. Khiếu nại phải gửi trong
24 giờ kể từ email thông báo hoặc công bố kết quả.

## 10. Checklist Trước Submission

```bash
docker compose -f docker-compose.yml config -q
```

- [ ] Model và served name là `LFM2.5-1.2B-Instruct`.
- [ ] Image public trên Docker Hub và pin digest.
- [ ] Entrypoint giữ đúng module API server chính thức.
- [ ] `/model`, host và port đúng contract.
- [ ] Không có `build:`, secret, network runtime hoặc local volume.
- [ ] Context length cover max input cộng output và headroom tokenizer/chat template.
- [ ] Image là linux/amd64, boot được trên driver 590.x/CUDA 13.x.
- [ ] Endpoint healthcheck, streaming và output token đều hợp lệ.
- [ ] Candidate có snapshot, digest và rollback artifact.
- [ ] Nếu dùng quantization, giữ một accuracy-safe finalist để GPQA hậu kiểm.
