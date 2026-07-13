# Lịch Sử Submission — Phase 1

Tài liệu này chỉ lưu kết quả portal, config tương ứng và bài học đã rút ra. Metric portal là
ground truth; mọi giải thích không có runtime log phải được ghi rõ là suy luận.

Tham số chấm: TTFT `100–1500 ms`, TPOT `20–45 ms`, gamma 2, trọng số 0.5/0.5.

## Trước Khi Có Score — Boot Và Serving Failures

### Hai lượt timeout khi boot

Triệu chứng: container treo tới timeout, không có request thành công.

Finding được giữ lại:

- Model đã mount tại `/model`, nhưng Hugging Face/Transformers vẫn có thể thử gọi network.
- Môi trường BTC drop network packet nên call không fail nhanh mà treo lâu.
- Sau khi thêm `HF_HUB_OFFLINE=1` và `TRANSFORMERS_OFFLINE=1`, cùng dòng config cơ bản boot
  và serve được 120/120.

Bài học: giữ hai offline env trong mọi phiên bản. Không dùng `HF_HUB_ENABLE_HF_TRANSFER`.

### Aggressive scheduler — 118/120 transport errors

Config lịch sử có `max-num-batched-tokens=16384`, `max-num-seqs=32` và
`gpu-memory-utilization=0.95`.

Portal trả 118/120 transport errors. Không có runtime log nên chưa thể phân biệt OOM,
preemption storm hay engine crash. Không dùng lại tổ hợp này như một candidate hoàn chỉnh.

## Lượt 1 — Score 15.47

Kết quả:

| Metric | Value |
| --- | ---: |
| Score | **15.47** |
| ERC | 0.7 |
| Passed SLO | 84/120 |
| Failed | 0 |
| Accuracy drop | 0 |
| TTFT p50 | 709 ms |
| TTFT p95 | 10146 ms |
| TBT median | 59 ms |

Config đã biết:

```text
vllm/vllm-openai:v0.22.1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
gpu-memory-utilization=0.95
enable-prefix-caching
không FP8 weight
không explicit max-num-batched-tokens/max-num-seqs
```

Có mâu thuẫn lịch sử về `max-model-len`: snapshot
`configs/vllm/submission-score-15_47.compose.yml` dùng 262144, trong khi ghi chú sau đó nói
lượt portal dùng 32768. Không dùng lượt này để kết luận tác động riêng của context length.

Bài học:

- Offline boot path đã ổn định.
- TPOT không dễ trên H200 MIG: median 59 ms cao hơn ceiling 45 ms.
- TTFT tail hơn 10 giây làm nhiều request mất toàn bộ điểm TTFT.
- Accuracy drop 0 để lại toàn bộ budget 10 điểm phần trăm cho quantization experiment.

## Lượt 2 — Explicit Batched Tokens 8192 — Score 3.79

Thay đổi chính: thêm explicit `max-num-batched-tokens=8192` và explicit chunked-prefill
flag.

| Metric | Lượt 1 | Lượt 2 |
| --- | ---: | ---: |
| Score | 15.47 | **3.79** |
| ERC | 0.7 | 0.6 |
| Passed SLO | 84 | 72 |
| Failed | 0 | 0 |
| Accuracy drop | 0 | 0 |
| TTFT p50 | 709 | **1841 ms** |
| TTFT p95 | 10146 | 9538 ms |
| TBT median | 59 | 52 ms |

Bài học cũ "chunked prefill nói chung là thảm họa" đã bị loại bỏ. Audit source vLLM
`0.22.1` cho thấy Qwen3.5 đã bật chunked prefill mặc định và MIG 18 GB dùng implicit token
budget khoảng 2048. Thay đổi thực cần đánh giá là `2048 -> 8192`.

Suy luận hiện tại:

- Qwen3.5 dùng GDN/Mamba cache mode `align`.
- Budget 2048 tạo scheduler boundary tại 2048/4096/6144, nằm trong common prefix khoảng
  6396 token.
- Budget 8192 có thể vượt điểm phân nhánh trước boundary đầu tiên, làm mất phần lớn GDN
  prefix reuse và tăng TTFT.

Không quay lại 8192. Các budget trace-aware 1024/3072/6144 vẫn là experiment hợp lệ.

## Lượt 3 — FP8 Weight — Score 17.37

Nộp ngày 10/07/2026 lúc 09:55. Compose portal 796 bytes, SHA-256 bắt đầu bằng
`cef9be117338`.

Snapshot chính xác:

```text
configs/vllm/submission-score-17_37.compose.yml
```

Config:

```text
vllm/vllm-openai:v0.22.1
max-model-len=32768
gpu-memory-utilization=0.95
enable-prefix-caching
quantization=fp8
offline env
```

| Metric | Value | So với 15.47 |
| --- | ---: | ---: |
| Score | **17.37** | +12.28% |
| ERC | 0.708333 | +1.19% |
| Passed SLO | 85/120 | +1 |
| Failed | 0 | giữ nguyên |
| Accuracy drop | 0 | giữ nguyên |
| Penalty | 1 | giữ nguyên |
| TTFT p50 | 632 ms | -10.86% |
| TTFT p95 | 8424 ms | -16.97% |
| TBT median | 51 ms | -13.56% |
| Warmup count | 0 | — |

Bài học:

- FP8 weight boot sạch, serve đủ và không gây accuracy penalty.
- FP8 cải thiện cả TTFT và TBT nhưng không đạt kỳ vọng gần 2 lần.
- Median TBT vẫn cao hơn 45 ms, nên nửa điểm TPOT của request median vẫn bằng 0.
- TTFT tail còn 8.4 giây; queue/concurrency vẫn là bottleneck lớn.

## Lượt 4 — Language Model Only — Score 17.80

Candidate chỉ thêm đúng một flag vào bản 17.37:

```text
--language-model-only
```

Compose đã nộp có SHA-256:

```text
cda990684b92b308b37a7243682280e1c62f0685a6ec3331a489af3767547a6a
```

Snapshot chính xác:

```text
configs/vllm/submission-score-17_80.compose.yml
```

| Metric | 17.37 | 17.80 | Delta |
| --- | ---: | ---: | ---: |
| Score | 17.37 | **17.80** | +2.48% |
| ERC | 0.708333 | 0.716667 | +1.18% |
| Passed SLO | 85 | 86 | +1 request |
| Failed | 0 | 0 | giữ nguyên |
| Accuracy drop | 0 | 0 | giữ nguyên |
| Penalty | 1 | 1 | giữ nguyên |
| TTFT p50 | 632 | **614 ms** | -2.85% |
| TTFT p95 | 8424 | **8289 ms** | -1.60% |
| TBT median | 51 | **51 ms** | không đổi |
| Warmup count | 0 | 0 | không đổi |

Bài học:

- Bỏ vision encoder là một win nhỏ nhưng sạch và được promote thành base mới.
- Cải thiện score đến từ TTFT; TBT không đổi, nên `--language-model-only` không xử lý nút
  thắt decode.
- Thêm VRAM/cache headroom chỉ giúp thêm một request vượt SLO và giảm nhẹ TTFT tail.

## Lượt 5 — BF16 GDN/SSM State — Score 47.52

Candidate chỉ thêm đúng một flag vào bản 17.80:

```text
--mamba-ssm-cache-dtype=bfloat16
```

Compose đã nộp có SHA-256:

```text
06837f0546bf04b31e5e3cc5397db9a2e6155dc7d85cfa9f48e173c6855a248f
```

Snapshot chính xác:

```text
configs/vllm/submission-score-47_52.compose.yml
```

| Metric | 17.80 | 47.52 | Delta |
| --- | ---: | ---: | ---: |
| Score | 17.80 | **47.52** | +29.72 điểm / +166.97% |
| ERC | 0.716667 | **0.808333** | +0.091666 |
| Passed SLO | 86 | **97** | +11 request |
| Failed | 0 | 0 | giữ nguyên |
| Accuracy drop | 0 | 0 | giữ nguyên |
| Penalty | 1 | 1 | giữ nguyên |
| TTFT p50 | 614 | **1723 ms** | +1109 ms / xấu hơn 180.62% |
| TTFT p95 | 8289 | **2241 ms** | -6048 ms / tốt hơn 72.96% |
| TBT median | 51 | **22 ms** | -29 ms / tốt hơn 56.86% |
| Warmup count | 0 | 0 | giữ nguyên |

Bài học:

- Giả thuyết GDN/SSM state là bottleneck decode đã được portal xác nhận rất mạnh. BF16 state
  đưa median TBT từ ngoài SLO 51 ms xuống sát floor 20 ms mà không mất accuracy.
- Score tăng gần 30 điểm dù TTFT p50 xấu hơn. Lợi ích decode và việc drain queue nhanh hơn
  đã kéo TTFT p95 từ 8.3 giây xuống 2.2 giây.
- Bottleneck hiện chuyển sang prefill/TTFT: p50 1723 ms nằm ngoài ceiling 1500 ms, trong khi
  TBT 22 ms chỉ còn cách floor 2 ms.
- Không tiếp tục cap concurrency hoặc decode micro-tuning. Chỉ cần TBT tăng vài ms có thể
  mất nhiều điểm; lượt tiếp theo phải giảm TTFT nhưng giữ TBT quanh 22–23 ms.
- Bản này vượt promote gate và trở thành base mới.

## Lượt 6 — Batched Token Budget 3072 — Score 48.67

Candidate chỉ thêm đúng một flag vào bản 47.52:

```text
--max-num-batched-tokens=3072
```

Compose đã nộp có SHA-256:

```text
dce4a164c29c25ff791cd2e6d42f32352d4d73d52355906d60c144f0a1962316
```

Snapshot chính xác:

```text
configs/vllm/submission-score-48_67.compose.yml
```

| Metric | 47.52 | 48.67 | Delta |
| --- | ---: | ---: | ---: |
| Score | 47.52 | **48.67** | +1.15 điểm / +2.42% |
| ERC | 0.808333 | **0.700000** | -0.108333 |
| Passed SLO | 97 | **84** | -13 request |
| Failed | 0 | 0 | giữ nguyên |
| Accuracy drop | 0 | 0 | giữ nguyên |
| Penalty | 1 | 1 | giữ nguyên |
| TTFT p50 | 1723 | **1870 ms** | +147 ms / xấu hơn 8.53% |
| TTFT p95 | 2241 | **2743 ms** | +502 ms / xấu hơn 22.40% |
| TBT median | 22 | **21 ms** | -1 ms / tốt hơn 4.55% |
| Warmup count | 0 | 0 | giữ nguyên |

Bài học:

- Candidate vượt promote gate đúng 1.15 điểm, nhưng hypothesis giảm TTFT không thành công.
- Budget 3072 làm cả TTFT p50/p95 và passed SLO xấu hơn. Điểm tăng đến từ TBT 22 -> 21 ms,
  cho thấy batch lớn hơn cải thiện nhẹ decode efficiency nhưng làm prefill monopolization nặng hơn.
- Không tiếp tục 6144. Dữ liệu 2048 -> 3072 đã cho xu hướng TTFT xấu; nhảy tiếp 6144 không
  còn plausible upside đủ mạnh dù boundary 6144 nằm trong common prefix.
- Bản 48.67 vẫn được promote vì vượt policy +1 điểm, nhưng lever tiếp theo phải tăng cache
  capacity/admission thay vì tăng scheduler token budget.

## Lượt 7 — FP8 KV Cache — Score 28.18

Candidate thêm package hai flag cùng phục vụ một thay đổi KV quantization lên base 48.67:

```text
--kv-cache-dtype=fp8
--calculate-kv-scales
```

Compose đã nộp có SHA-256:

```text
08676e0bd197086fa356089bf99160ee898c21e9124fdfb75921a324a7074845
```

Snapshot chính xác:

```text
configs/vllm/submission-score-28_18.compose.yml
```

| Metric | 48.67 | 28.18 | Delta |
| --- | ---: | ---: | ---: |
| Score | 48.67 | **28.18** | -20.49 điểm / -42.10% |
| ERC | 0.700000 | 0.700000 | không đổi |
| Passed SLO | 84 | 84 | không đổi |
| Failed | 0 | 0 | giữ nguyên |
| Accuracy drop | 0 | **2** | +2 percentage points, penalty vẫn 1 |
| Penalty | 1 | 1 | giữ nguyên |
| TTFT p50 | 1870 | **1817 ms** | tốt hơn 53 ms / 2.83% |
| TTFT p95 | 2743 | **2105 ms** | tốt hơn 638 ms / 23.26% |
| TBT median | 21 | **30 ms** | xấu hơn 9 ms / 42.86% |
| Warmup count | 0 | 0 | giữ nguyên |

Bài học:

- FP8 KV có cải thiện cache/admission signature ở TTFT tail, đặc biệt p95, nhưng attention
  decode cost tăng quá mạnh và kéo score giảm hơn 20 điểm.
- Dynamic KV scales chỉ làm forward đầu tiên chạy ngoài CUDA graph; source vLLM tắt chế độ
  calculate sau forward đó. Median TBT 30 ms vì vậy phản ánh chủ yếu FP8 KV attention path,
  không phải overhead scale kéo dài.
- Không thử lại FP8 KV không-scale hoặc quantize subset layer. Mức TBT regression 9 ms quá
  lớn; các biến thể nhỏ không có plausible path quay lại best 48.67.
- Accuracy drop portal hiển thị 2 nhưng penalty vẫn 1, nên accuracy không phải nguyên nhân
  score giảm.
- Rollback hai flag FP8 KV. vLLM compose-level levers hiện không còn candidate đủ upside;
  chuyển sang backend A/B SGLang.

## Lượt 8 — SGLang Explicit Entrypoint — Boot Failure

Artifact backend A/B đầu tiên:

```text
configs/sglang/submission-backend-ab.compose.yml
SHA-256 02f8f6e032f2c127d54104302bc10a7bcd4daf6845e5809851a588fe1adbe5a4
```

Portal không tạo metric vì pod exit 1 trước khi ready. Log cuối:

```text
/usr/bin/python3: Error while finding module specification for
'vllm.entrypoints.openai.api_server' (ModuleNotFoundError: No module named 'vllm')
```

Finding:

- Artifact local và `docker compose config` đều chỉ định `python3 -m sglang.launch_server`.
- Digest image được Docker manifest xác nhận là SGLang v0.5.14 CUDA 12.9 runtime; image
  không phải vLLM image.
- Vì log gọi module vLLM, pod thực tế không chạy entrypoint trong artifact đã render. Khả
  năng hợp lý nhất là runner giữ/override explicit entrypoint từ baseline vLLM.
- Retry đúng một lần bằng command-only: bỏ Compose `entrypoint`, để NVIDIA entrypoint gốc
  của image thực thi toàn bộ `python3 -m sglang.launch_server ...` trong `command`.
- Không thay performance package ở retry, nên nếu boot thành công đây vẫn là cùng một phép
  A/B backend. Nếu log tiếp tục gọi vLLM, đóng nhánh SGLang trên runner hiện tại.

## Lượt 9 — SGLang Command-Only Retry — Boot Failure

Artifact retry:

```text
configs/sglang/submission-backend-ab-command-only.compose.yml
SHA-256 096560d7c47fa78247ca65b3bcaf7e888942f41001b6701176097350dda0f0bd
```

Retry bỏ hoàn toàn Compose `entrypoint` và đặt lệnh đầy đủ
`python3 -m sglang.launch_server ...` trong `command`. Portal vẫn trả đúng lỗi lượt trước:

```text
/usr/bin/python3: Error while finding module specification for
'vllm.entrypoints.openai.api_server' (ModuleNotFoundError: No module named 'vllm')
```

Kết luận:

- Runner không chỉ giữ explicit entrypoint; nó ép launch module vLLM bất kể cả `command`.
- Direct SGLang bằng upstream image không tương thích runner, nên đóng nhánh sau đúng hai
  boot attempts và không suy luận bất kỳ metric hiệu năng nào từ hai lượt này.
- Root rollback byte-identical về score 48.67.
- Muốn đổi backend sau này, public image phải tự cung cấp module path
  `vllm.entrypoints.openai.api_server` và tương thích các flag runner truyền vào. Đây là
  custom-image project, không còn là compose-only experiment.

## Lượt 10 — vLLM 0.24.0 Version Upgrade — Score 59.10

Candidate giữ nguyên toàn bộ package score 48.67 và chỉ đổi official image từ vLLM
`v0.22.1` sang vLLM `v0.24.0` CUDA 12.9 pinned digest:

```text
vllm/vllm-openai@sha256:3de11aaf1d2aa1c6245a93e9279cc10af6d0b9f5eb3b34704fbd099a8ac42c7d
```

Compose đã nộp và snapshot chính xác:

```text
configs/vllm/submission-score-59_10.compose.yml
SHA-256 b67e530b3ad40cc7a933f863574275ee21504162660e9f5080d94dbb9939c6c4
```

| Metric | 48.67 | 59.10 | Delta |
| --- | ---: | ---: | ---: |
| Score | 48.67 | **59.10** | +10.43 điểm / +21.43% |
| ERC | 0.700000 | **1.000000** | +0.30 |
| Passed SLO | 84 | **120** | +36 request |
| Failed | 0 | 0 | giữ nguyên |
| Accuracy drop | 0 | 0 | giữ nguyên |
| Penalty | 1 | 1 | giữ nguyên |
| TTFT p50 | 1870 | **521 ms** | tốt hơn 1349 ms / 72.14% |
| TTFT p95 | 2743 | **1147 ms** | tốt hơn 1596 ms / 58.18% |
| TBT median | 21 | **25 ms** | xấu hơn 4 ms / 19.05% |
| Warmup count | 0 | 0 | giữ nguyên |

Bài học:

- Version upgrade tạo bước nhảy runtime thật, không phải noise: score +10.43 và toàn bộ
  120 request qua SLO mà không có failure hay accuracy regression.
- Gain đến từ TTFT/cache/prefill: p50 giảm 72%, p95 giảm 58%. Đây là bằng chứng v0.24 đã
  thay đổi performance ceiling đúng workload hybrid Qwen3.5.
- TBT tăng 21 -> 25 ms, nên không được xem package hiện tại là tối ưu hoàn tất. Decode trở
  thành bottleneck mới dù tổng score tăng mạnh.
- Promote v0.24.0 thành base 59.10. Candidate tiếp theo phải giữ image và retune scheduler,
  bắt đầu bằng budget 2048 để giảm prefill interference; không đổi thêm image hoặc cache dtype.

## Lượt 11 — vLLM 0.24.0 Budget 2048 — Score 65.06

Candidate giữ nguyên base 59.10 và chỉ đổi:

```text
--max-num-batched-tokens=3072
->
--max-num-batched-tokens=2048
```

Compose đã nộp và snapshot chính xác:

```text
configs/vllm/submission-score-65_06.compose.yml
SHA-256 e8acfc79438e87922f8fbbaf8b295b65cfc980ea4d523e8a3f85e315620c143b
```

| Metric | 59.10 | 65.06 | Delta |
| --- | ---: | ---: | ---: |
| Score | 59.10 | **65.06** | +5.96 điểm / +10.08% |
| ERC | 1.000000 | 1.000000 | giữ nguyên |
| Passed SLO | 120 | 120 | giữ nguyên |
| Failed | 0 | 0 | giữ nguyên |
| Accuracy drop | 0 | 0 | giữ nguyên |
| Penalty | 1 | 1 | giữ nguyên |
| TTFT p50 | 521 | **266 ms** | tốt hơn 255 ms / 48.94% |
| TTFT p95 | 1147 | **1256 ms** | xấu hơn 109 ms / 9.50% |
| TBT median | 25 | **26 ms** | xấu hơn 1 ms / 4.00% |
| Warmup count | 0 | 0 | giữ nguyên |

Bài học:

- Budget thấp hơn tăng fairness rất mạnh cho phần lớn workload: TTFT p50 gần giảm một nửa.
- Chi phí là nhiều scheduler step hơn, thể hiện ở TBT +1 ms và p95 +109 ms. Tuy vậy net
  score vẫn +5.96, nên median/fairness gain áp đảo tail/decode regression ở điểm hiện tại.
- Promote 2048 thành base 65.06. Không quay lại 3072 và chưa cap concurrency trước khi đo
  frontier budget 1024.
- Candidate 1024 là phép đo quyết định: nếu tiếp tục thắng thì giữ và chuyển sang concurrency;
  nếu thua thì rollback 2048 và đóng hướng giảm budget, không sweep mù nhiều giá trị giữa.

## Lượt 12 — vLLM 0.24.0 Budget 1024 — Score 64.34

Candidate giữ nguyên base 65.06 và chỉ đổi:

```text
--max-num-batched-tokens=2048
->
--max-num-batched-tokens=1024
```

Compose đã nộp và snapshot chính xác:

```text
configs/vllm/submission-score-64_34.compose.yml
SHA-256 8aaf6a64f0669894a92203be9cb9329ac1dc33c423d20aff29f398fe8f163ed4
```

| Metric | 65.06 | 64.34 | Delta |
| --- | ---: | ---: | ---: |
| Score | 65.06 | **64.34** | -0.72 điểm / -1.11% |
| ERC | 1.000000 | 1.000000 | giữ nguyên |
| Passed SLO | 120 | 120 | giữ nguyên |
| Failed | 0 | 0 | giữ nguyên |
| Accuracy drop | 0 | **3** | +3 percentage points, penalty vẫn 1 |
| Penalty | 1 | 1 | giữ nguyên |
| TTFT p50 | 266 | **277 ms** | xấu hơn 11 ms / 4.14% |
| TTFT p95 | 1256 | **1242 ms** | tốt hơn 14 ms / 1.11% |
| TBT median | 26 | **27 ms** | xấu hơn 1 ms / 3.85% |
| Warmup count | 0 | 0 | giữ nguyên |

Bài học:

- 1024 không tiếp tục xu hướng 3072 -> 2048: TTFT p50 và TBT cùng xấu hơn, chỉ p95 tốt hơn
  14 ms. Scheduler overhead đã vượt fairness benefit.
- Accuracy drop portal là 3 điểm phần trăm nhưng penalty vẫn 1, không phải nguyên nhân score
  giảm. Latency regression đủ giải thích kết quả.
- Rollback 2048 và xác nhận đây là budget winner trong ba điểm 3072/2048/1024.
- Không thử budget thấp hơn hoặc 1536. Chuyển đúng một lần sang cap active sequences 20 để
  nhắm TBT; nếu cap không thắng thì đóng compose scheduler tuning.

## Lượt 13 — vLLM 0.24.0 Max Sequences 20 — Score 54.56

Candidate rollback budget winner 2048 và thêm đúng một flag:

```text
--max-num-seqs=20
```

Compose đã nộp và snapshot chính xác:

```text
configs/vllm/submission-score-54_56.compose.yml
SHA-256 aa8c5f7dadc7fb673149d51bbb15247ff4845ab70a1e1b477b11f8dde5330827
```

| Metric | 65.06 | 54.56 | Delta |
| --- | ---: | ---: | ---: |
| Score | 65.06 | **54.56** | -10.50 điểm / -16.14% |
| ERC | 1.000000 | **0.166667** | -0.833333 |
| Passed SLO | 120 | **20** | -100 request |
| Failed | 0 | 0 | giữ nguyên |
| Accuracy drop | 0 | 0 | giữ nguyên |
| Penalty | 1 | 1 | giữ nguyên |
| TTFT p50 | 266 | **3499 ms** | xấu hơn 3233 ms / 1215.41% |
| TTFT p95 | 1256 | **3981 ms** | xấu hơn 2725 ms / 216.96% |
| TBT median | 26 | **18 ms** | tốt hơn 8 ms / 30.77% |
| Warmup count | 0 | 0 | giữ nguyên |

Bài học:

- Concurrency là lever decode mạnh nhất đã thấy: cap20 kéo TBT từ 26 xuống 18 ms, vượt
  floor 20 ms. Hypothesis giảm decode contention là đúng.
- Fixed cap bằng đúng một burst gây head-of-line queue: ERC còn 1/6 và chỉ 20/120 request
  qua SLO. Score mất 10.50 dù decode cực nhanh.
- Với 200 output token, completion đại diện là `3499 + 199×18 = 7081 ms`, lớn hơn burst
  period 5000 ms. Minimum capacity theo workload là hai burst, tức 40 sequence.
- Cho phép đúng một candidate cap40 vì giá trị này được suy ra từ service time và burst width,
  không phải sweep. Nếu cap40 thua hoặc ERC <0.8, đóng fixed-cap tuning hoàn toàn.
- Đường dài tới 100 chuyển sang adaptive scheduler và generic in-flight prefix coalescing;
  fixed compose cap không thể đồng thời đạt TTFT 100 ms và TBT 20 ms.

## Lượt 14 — vLLM 0.24.0 Max Sequences 40 — Score 63.89

Candidate giữ base 65.06 và thêm:

```text
--max-num-seqs=40
```

Compose đã nộp và snapshot chính xác:

```text
configs/vllm/submission-score-63_89.compose.yml
SHA-256 71a2c1ae74186e88dcc3337ac212d4a6321631c48159e335d3e05e8a27e16317
```

| Metric | 65.06 | 63.89 | Delta |
| --- | ---: | ---: | ---: |
| Score | 65.06 | **63.89** | -1.17 điểm / -1.80% |
| ERC | 1.000000 | 1.000000 | giữ nguyên |
| Passed SLO | 120 | 120 | giữ nguyên |
| Failed | 0 | 0 | giữ nguyên |
| Accuracy drop | 0 | 0 | giữ nguyên |
| Penalty | 1 | 1 | giữ nguyên |
| TTFT p50 | 266 | **292 ms** | xấu hơn 26 ms / 9.77% |
| TTFT p95 | 1256 | **1280 ms** | xấu hơn 24 ms / 1.91% |
| TBT median | 26 | 26 ms | không đổi |
| Warmup count | 0 | 0 | giữ nguyên |

Bài học:

- Cap40 loại hoàn toàn queue catastrophe của cap20: ERC và passed SLO trở lại 1/120.
- Tuy nhiên TBT quay lại đúng 26 ms và cả TTFT p50/p95 đều xấu hơn best. Fixed capacity hai
  burst không giữ được decode benefit 18 ms của cap20.
- Rollback best 65.06. Đóng toàn bộ fixed-cap branch; không thử 32/48/60/80.
- Ba endpoint đủ để kết luận fixed cap không thể đạt cả queue và decode: default tốt queue,
  cap20 tốt decode, cap40 giống default nhưng chậm hơn.
- Phase kế tiếp là custom adaptive scheduler: admit/prefill burst mới sớm để giữ TTFT, nhưng
  điều tiết running decode theo queue age/service time thay vì hard cap tĩnh.

## Lượt 15 — Custom Adaptive Scheduler Cohort 20 — Chờ Kết Quả

Candidate giữ nguyên toàn bộ best 65.06 và chỉ đổi ba điểm liên kết thành một package runtime:

```text
image ghcr.io/tanphong-sudo/qwen35-adaptive@sha256:8a18315745a39d54085e1d99bfbb7e5ae55e5b6fb320132c7261abfa4dfc18db
QWEN35_DECODE_WINDOW=20
--scheduler-cls=qwen35_adaptive.scheduler.CompletionCohortAsyncScheduler
```

Artifact chờ nộp:

```text
configs/vllm/submission-adaptive-cohort20.compose.yml
SHA-256 54e27c29e647bc5b826da907d45572b90bda9faac90c1c544ce218a21e99e29e
```

Gate đã qua trước portal:

- 29 unit tests pass và `docker compose config -q` pass.
- GitHub Actions run `29231204106` build đúng `linux/amd64` từ base vLLM v0.24 pinned.
- Scheduler import smoke test pass trong exact image.
- Exact `schedule(throttle_prefills=False)` signature và các field request dùng bởi policy đã
  đối chiếu với source vLLM `v0.24.0`.
- Anonymous GHCR manifest GET trả HTTP 200, đúng digest; image config là Linux amd64.

Hypothesis: burst mới vẫn được prefill và nhận first token sớm, trong khi tối đa 20 mature
decoder gần hoàn thành được chạy liên tục. Mục tiêu là giữ ERC 1 và TTFT gần base nhưng kéo
TBT từ 26 ms về 20–23 ms. Đây là một phép đo scheduler duy nhất; không thêm flag khác trước
khi portal trả metric.

## Candidate Cascade Attention — Hủy Trước Khi Nộp

Candidate từng dự kiến thêm đúng một flag vào bản 17.80:

```text
--no-disable-cascade-attn
```

Snapshot nghiên cứu:

```text
configs/vllm/submission-cascade-attn.compose.yml
```

Candidate bị hủy trước khi nộp. Qwen3.5 chỉ có 6/24 full-attention layer, trong khi TBT
không đổi sau language-model-only cho thấy bottleneck chính nằm ở decode path/GDN. Expected
gain của cascade không đủ lớn để tiêu một lượt portal.

Rollback:

```bash
cp configs/vllm/submission-score-48_67.compose.yml docker-compose.yml
```

## Nguyên Tắc Giữ Lại

- Một biến mỗi lượt portal.
- Snapshot mọi phiên bản đã score trước khi sửa root compose.
- `failed_count > 0` hoặc `penalty < 1` thì rollback ngay.
- Promote khi final score tăng ít nhất 1 điểm; median riêng lẻ không đủ quyết định.
- Giữ offline env và prefix caching trong mọi candidate.
- Không xem local replay là official score.
