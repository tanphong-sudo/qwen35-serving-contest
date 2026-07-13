# Finding Kỹ Thuật Hiện Tại

Tài liệu này lưu mô hình kỹ thuật hiện tại và thứ tự experiment. Ground truth metric nằm
trong `submission-results-log.md`.

## 1. Ground Truth Của Base 65.06

Best confirmed hiện tại là vLLM `v0.24.0` với scheduler token budget 2048. Snapshot rollback:

```text
configs/vllm/submission-score-65_06.compose.yml
SHA-256 e8acfc79438e87922f8fbbaf8b295b65cfc980ea4d523e8a3f85e315620c143b
```

Config chính:

```text
vllm/vllm-openai@sha256:3de11aaf1d2aa1c6245a93e9279cc10af6d0b9f5eb3b34704fbd099a8ac42c7d
--max-model-len=32768
--max-num-batched-tokens=2048
--gpu-memory-utilization=0.95
--enable-prefix-caching
--quantization=fp8
--mamba-ssm-cache-dtype=bfloat16
--language-model-only
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Kết quả portal:

| Metric | Value |
| --- | ---: |
| `final_score` | **65.06** |
| `ers` hiển thị | **65.06** |
| ERS chuẩn hóa | **0.6506** |
| `erc` | **1.000000** |
| `passed_slo` | **120/120** |
| `failed_count` | 0 |
| `accuracy_drop` | 0 |
| `penalty` | 1 |
| `ttft_p50_ms` | 266 |
| `ttft_p95_ms` | 1256 |
| `tbt_median_ms` | 26 |
| `warmup_count` | 0 |

So với base 59.10:

| Metric | 59.10 | 65.06 | Delta |
| --- | ---: | ---: | ---: |
| Score | 59.10 | **65.06** | +5.96 điểm / +10.08% |
| ERC | 1.000000 | 1.000000 | giữ nguyên |
| Passed SLO | 120 | 120 | giữ nguyên |
| TTFT p50 | 521 | **266 ms** | tốt hơn 255 ms / 48.94% |
| TTFT p95 | 1147 | **1256 ms** | xấu hơn 109 ms / 9.50% |
| TBT median | 25 | **26 ms** | xấu hơn 1 ms / 4.00% |

Budget 2048 là fairness/median win lớn: p50 gần giảm một nửa và score +5.96, dù scheduler
overhead làm TBT và p95 xấu nhẹ. Điều này chứng minh score hiện còn nhạy mạnh với TTFT của
phần lớn request hơn là một điểm TBT. Bước kế tiếp giảm budget xuống 1024 để đo frontier
dứt khoát; không dùng 1536 trước khi biết hướng 1024 vì phép đo giữa điểm không xác định
được còn upside lớn hay đã vượt optimum.

## 2. Hình Dạng Workload Thật

Trace gồm 120 request, tổ chức thành 20 conversation và 6 turn:

```text
t=0s:   20 request, 2 messages
t=5s:   20 request, 4 messages
t=10s:  20 request, 6 messages
t=15s:  20 request, 8 messages
t=20s:  20 request, 10 messages
t=25s:  20 request, 12 messages
```

Trong mỗi burst, request đến cách nhau 25 ms và burst kéo dài 475 ms. Mỗi turn sau giữ
nguyên toàn bộ conversation prefix của turn trước. Toàn bộ 120 request còn có chung một
system prefix dài khoảng 6,396 token.

Ở base 17.80, TBT median 51 ms làm một generation 200 token mất xấp xỉ 10 giây, khiến nhiều
turn chồng lên nhau và tạo TTFT tail 8.3 giây. Base 65.06 có TBT 26 ms, tương đương khoảng
5.2 giây cho 200 token, hơi dài hơn khoảng cách giữa hai burst. Dù vậy budget 2048 kéo TTFT
p50 xuống 266 ms; p95 1256 ms cho thấy một nhóm tail vẫn chịu overlap giữa decode turn trước
và prefill turn kế tiếp.

## 3. Default Thật Của vLLM 0.22.1

Các config cũ gọi bản không có `--enable-chunked-prefill` là "không chunked". Cách gọi đó
sai với vLLM V1:

- Qwen3.5 hỗ trợ chunked prefill nên vLLM bật mặc định.
- MIG 18 GB nằm dưới nhánh GPU 70 GiB, nên OpenAI API server mặc định dùng
  `max_num_batched_tokens=2048`.
- Default `max_num_seqs=256`, nhưng số sequence được admit thực tế còn bị giới hạn bởi
  cache capacity và `scheduler_reserve_full_isl`.
- Async scheduling cũng tự bật khi config không có tính năng xung đột.

Nguồn implementation:

- `vllm/engine/arg_utils.py` tag `v0.22.1`
- `vllm/config/scheduler.py` tag `v0.22.1`
- <https://docs.vllm.ai/en/v0.22.0/configuration/engine_args/>

Hệ quả: các base vLLM không đặt explicit budget đã chunk prompt 13k–27k thành khoảng 7–14
scheduler step với budget mặc định 2048. Không cần thêm `--enable-chunked-prefill` vào
compose.

## 4. Qwen3.5 Không Phải Transformer Attention Thuần

Qwen3.5-2B có 24 decoder layer nhưng chỉ 6 layer full attention; 18 layer còn lại dùng
Gated DeltaNet/linear attention. Model config còn đặt SSM state dtype mặc định là
`float32`.

Nguồn:

- <https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json>
- `vllm/model_executor/models/qwen3_5.py` tag `v0.22.1`

Điều này thay đổi thứ tự tối ưu:

1. FP8 KV chỉ tác động trực tiếp tới 6 full-attention layer, không phải toàn bộ 24 layer.
2. GDN/Mamba state có thể là nguồn memory bandwidth đáng kể trong decode.
3. Portal đã xác nhận `--mamba-ssm-cache-dtype=bfloat16` là đòn lớn: TBT 51 -> 22 ms,
   score 17.80 -> 47.52 và accuracy drop vẫn bằng 0.

## 5. Diễn Giải Lại Thất Bại 8192

Lượt 3.79 đã thêm explicit chunked prefill và đặt `max-num-batched-tokens=8192`. Thay đổi
thật so với default không phải "bật chunked", mà chủ yếu là tăng token budget từ khoảng
2048 lên 8192.

Qwen3.5 dùng Mamba prefix cache mode `align`. Audit sâu hơn source vLLM `0.22.1` sửa lại
kết luận boundary cũ:

- `_mamba_block_aligned_split` căn theo physical cache block, mặc định 16 token, không căn
  trực tiếp theo `max_num_batched_tokens`.
- Common prefix 6396 token có last cache-aligned position 6384.
- Nếu một scheduled chunk định đi qua 6384, scheduler tự cắt chunk tại 6384. Vì vậy budget
  8192 không thực sự buộc Mamba state vượt vào divergent user tokens trong cùng chunk.
- Kết quả 2048 -> 3072 xác nhận trade-off thật: budget lớn hơn làm TBT 22 -> 21 ms nhưng
  TTFT p50/p95 xấu 8.5%/22.4%. Đây là prefill monopolization đổi lấy batch/decode efficiency.

Kết luận hiện tại: không quay lại 6144/8192. Lập luận “8192 phá prefix vì crossing 6396”
đã bị loại khỏi finding canonical.

## 6. Trạng Thái Repo Và Chính Sách Nộp

Root `docker-compose.yml` đã rollback về best confirmed 65.06 và trùng byte với:

```text
configs/vllm/submission-score-65_06.compose.yml
SHA-256 e8acfc79438e87922f8fbbaf8b295b65cfc980ea4d523e8a3f85e315620c143b
```

Không có compose candidate đang chờ nộp. Cap40 đã loại queue catastrophe của cap20 nhưng
không giảm TBT và score vẫn thấp hơn best. Fixed-cap branch đã đóng; work tiếp theo là custom
adaptive scheduler, chỉ được đặt vào root sau khi image public, launch contract và local
policy tests đều qua gate.

Best rollback vẫn là:

```text
configs/vllm/submission-score-65_06.compose.yml
SHA-256 e8acfc79438e87922f8fbbaf8b295b65cfc980ea4d523e8a3f85e315620c143b
```

Hai artifact SGLang liên tiếp đều bị runner chạy cố định
`python3 -m vllm.entrypoints.openai.api_server`, bất kể artifact dùng explicit SGLang
`entrypoint` hay command-only. Vì image SGLang không chứa module vLLM nên cả hai lượt đều
exit 1 trước ready và không có metric. Direct backend switch bằng compose đã đóng.

Hai snapshot boot-fail được giữ để không lặp lại:

```text
configs/sglang/submission-backend-ab.compose.yml
SHA-256 02f8f6e032f2c127d54104302bc10a7bcd4daf6845e5809851a588fe1adbe5a4
configs/sglang/submission-backend-ab-command-only.compose.yml
SHA-256 096560d7c47fa78247ca65b3bcaf7e888942f41001b6701176097350dda0f0bd
```

Base 47.52 vẫn được giữ để tách tác động của budget 3072:

```text
configs/vllm/submission-score-47_52.compose.yml
SHA-256 06837f0546bf04b31e5e3cc5397db9a2e6155dc7d85cfa9f48e173c6855a248f
```

Một candidate chỉ được chuyển sang root khi đã qua review theo tracker bên dưới. Chính sách
mới nhằm tránh tiêu lượt portal cho micro-tuning:

- Không nộp candidate có expected upside dưới khoảng **1 điểm**.
- Ưu tiên candidate có plausible upside **3 điểm trở lên** hoặc phép A/B backend có thể
  nâng performance ceiling.
- Chỉ bundle các thay đổi cùng phục vụ một giả thuyết kỹ thuật và không thể đánh giá hợp lý
  khi tách rời.
- `failed_count > 0` hoặc `penalty < 1` là hard rollback. Portal hiển thị accuracy drop theo
  điểm phần trăm; chỉ dùng `penalty` và free gate 10 điểm phần trăm để quyết định, không so
  trực tiếp số hiển thị với `0.10`.
- Một kết quả chỉ được promote thành base khi score tăng ít nhất **1.00 điểm**, không có
  failure và penalty vẫn bằng 1. Win nhỏ hơn vẫn được ghi vào history nhưng không làm root
  phình thêm flag.
- Mọi bản đã score phải có snapshot và hash trước khi sửa root.

## 7. Submission Tracker Đã Review

Trạng thái dùng trong bảng:

- `completed`: portal đã trả metric.
- `boot-failed`: container chưa ready nên không có metric.
- `local-prototype`: code và unit tests đã sẵn sàng nhưng image public/smoke test chưa qua.
- `ready`: artifact đã được đặt vào root, validate xong và đang chờ upload/kết quả portal.
- `planned`: candidate tiếp theo đủ upside để chuẩn bị, nhưng chưa được đặt vào root.
- `conditional`: chỉ được chuẩn bị khi trigger ở cột quyết định xảy ra.
- `canceled`: không nộp vì risk/reward không đạt chính sách.

### 7.1 Ground Truth Và Candidate Đã Hủy

| ID | Status | Config/package | Hypothesis | Metric signature | Promote gate | Quyết định tiếp theo |
| --- | --- | --- | --- | --- | --- | --- |
| `BASE-65.06` | `completed` | vLLM v0.24.0, budget 2048; snapshot `submission-score-65_06.compose.yml`; SHA-256 `e8acfc79438e87922f8fbbaf8b295b65cfc980ea4d523e8a3f85e315620c143b` | Best confirmed và rollback chuẩn | Score 65.06; TTFT 266/1256 ms; TBT 26 ms; ERC 1; 120 SLO; 0 failed | Current best | Candidate mới phải vượt 66.06, không failure và penalty 1 |
| `BASE-59.10` | `completed` | vLLM v0.24.0, budget 3072 | Historical rollback trước budget retune | Score 59.10; TTFT 521/1147 ms; TBT 25 ms | Đã bị thay thế | Dùng để kết luận 2048 đổi 1 ms TBT và 109 ms p95 lấy 255 ms p50, net +5.96 |
| `BASE-48.67` | `completed` | vLLM v0.22.1, budget 3072 | Historical rollback trước version upgrade | Score 48.67; TTFT 1870/2743 ms; TBT 21 ms; 0 failed | Đã bị thay thế | Dùng để tách tác động image v0.24.0: TTFT win rất lớn, TBT regression 4 ms |
| `BASE-47.52` | `completed` | FP8 weight, prefix cache, BF16 GDN state, language-model-only, implicit budget 2048 | Historical rollback | Score 47.52; TTFT 1723/2241 ms; TBT 22 ms | Đã bị thay thế về score | Dùng để kết luận 3072 tăng decode efficiency nhưng làm TTFT xấu |
| `BASE-17.80` | `completed` | FP8 weight + language-model-only, chưa BF16 GDN state | Historical rollback | Score 17.80; TTFT 614/8289 ms; TBT 51 ms | Đã bị thay thế | Chỉ dùng để đối chiếu tác động riêng của BF16 GDN state |
| `EXP-CASCADE` | `canceled` | Thêm `--no-disable-cascade-attn` | Chỉ tối ưu 6 full-attention layer | Expected gain nhỏ, khó kéo TBT qua 45 ms | Không đạt submission policy | Không nộp lại nếu chưa có bằng chứng bottleneck chuyển sang full attention |
| `SGLANG-ENTRYPOINT` | `boot-failed` | Snapshot `configs/sglang/submission-backend-ab.compose.yml`; SHA-256 `02f8f6e032f2c127d54104302bc10a7bcd4daf6845e5809851a588fe1adbe5a4`; explicit `entrypoint: python3 -m sglang.launch_server` | Đo backend ceiling | Pod chạy `/usr/bin/python3 -m vllm.entrypoints.openai.api_server` trong image SGLang và exit 1 vì không có module vLLM | Không có metric | Không thay performance flags; retry một lần bằng command-only để loại runner entrypoint mismatch |
| `SGLANG-COMMAND` | `boot-failed` | Snapshot `configs/sglang/submission-backend-ab-command-only.compose.yml`; SHA-256 `096560d7c47fa78247ca65b3bcaf7e888942f41001b6701176097350dda0f0bd`; không override entrypoint | Kiểm tra runner chỉ giữ explicit entrypoint hay ép toàn bộ launch path | Pod vẫn chạy đúng module vLLM cũ và exit 1 | Không có metric | Xác nhận runner ép launch module; đóng direct SGLang compose branch và rollback 48.67 |

### 7.2 Phase A — Phá Nút Thắt Decode, Mục Tiêu 20–30

| ID | Status | Exact config delta/package | Hypothesis | Expected metric signature | Promote gate | Rollback/branch decision |
| --- | --- | --- | --- | --- | --- | --- |
| `V-GDN-BF16` | `completed` | Base 17.80 + `--mamba-ssm-cache-dtype=bfloat16`; snapshot `submission-score-47_52.compose.yml`; SHA-256 `06837f0546bf04b31e5e3cc5397db9a2e6155dc7d85cfa9f48e173c6855a248f` | Giảm footprint/bandwidth của SSM state trên 18/24 layer | Actual: score 47.52; TBT 22 ms; TTFT 1723/2241 ms; 0 failed; accuracy drop 0 | Vượt gate rất xa | Promote thành base. Decode phase hoàn tất; bottleneck chuyển sang TTFT/prefill |
| `V-UPGRADE-0.24` | `completed` | Base 48.67, chỉ đổi image sang `vllm/vllm-openai@sha256:3de11aaf1d2aa1c6245a93e9279cc10af6d0b9f5eb3b34704fbd099a8ac42c7d`; snapshot `submission-score-59_10.compose.yml`; compose SHA-256 `b67e530b3ad40cc7a933f863574275ee21504162660e9f5080d94dbb9939c6c4` | Tối ưu Qwen3.5 GDN/hybrid cache/mixed scheduling của v0.24.0 nâng runtime ceiling | Actual: score 59.10; TTFT 521/1147 ms; TBT 25 ms; ERC 1; 120 SLO; accuracy drop 0 | Vượt gate +10.43 | Promote thành best. Giữ version; retune budget để lấy lại TBT |
| `SGLANG-DIRECT` | `canceled` | Hai package upstream SGLang ở trên | Runner ép module path `vllm.entrypoints.openai.api_server` | Hai boot failure giống hệt nhau | Không thể đo metric | Không nộp thêm image SGLang upstream; backend switch chỉ quay lại qua compatibility image có module path vLLM |

`V-GDN-BF16` đã hoàn thành Phase A và vượt thẳng mốc 30–50. Version v0.24 làm TBT quay lại
25 ms, nên decode tuning chỉ được mở lại dưới dạng scheduler retune có kiểm soát.

Hai boot failure không cung cấp bằng chứng hiệu năng SGLang; chúng chỉ chứng minh contract
runner thực tế hẹp hơn contract compose giả định. Không tiêu lượt thứ ba cho upstream image.
Version upgrade đã thắng lớn và xác nhận runner-compatible path có thể nâng ceiling. Không
đổi image tiếp cho tới khi retune scheduler v0.24 xong.

### 7.3 Phase B — Concurrency Shaping

| ID | Status | Exact config delta/package | Hypothesis | Expected metric signature | Promote gate | Rollback/branch decision |
| --- | --- | --- | --- | --- | --- | --- |
| `WIN-SEQS20` | `completed` | Best 65.06 + `--max-num-seqs=20`; snapshot `configs/vllm/submission-score-54_56.compose.yml`; SHA-256 `aa8c5f7dadc7fb673149d51bbb15247ff4845ab70a1e1b477b11f8dde5330827` | Một burst active có thể giảm decode contention | Actual: score 54.56; TTFT 3499/3981 ms; TBT 18 ms; ERC 0.166667; 20 SLO | Thất bại -10.50 | Decode hypothesis đúng nhưng admission cap quá thấp: chỉ một burst qua SLO. Không thử 12/8 |
| `WIN-SEQS40` | `completed` | Best 65.06 + `--max-num-seqs=40`; snapshot `configs/vllm/submission-score-63_89.compose.yml`; SHA-256 `71a2c1ae74186e88dcc3337ac212d4a6321631c48159e335d3e05e8a27e16317` | Hai-burst capacity có thể giữ decode gain cap20 mà bỏ queue | Actual: score 63.89; TTFT 292/1280 ms; TBT 26 ms; ERC 1; 120 SLO | Thất bại -1.17 | Cap40 loại queue nhưng không giữ bất kỳ decode gain nào. Rollback 65.06 và đóng toàn bộ fixed-cap tuning |
| `WIN-SEQS12` | `canceled` | Hạ cap xuống 12 | Đổi throughput lấy TPOT | Quá thấp so với burst 20, dễ tạo queue TTFT | Không đạt submission policy | Không thử 12 hoặc 8 |
| `WIN-GDN-PACKAGE` | `completed` | vLLM giữ `--mamba-ssm-cache-dtype=bfloat16` qua mọi promoted base | BF16 SSM state là điều kiện để tránh decode regression lớn | Base 65.06 vẫn dùng BF16 state, accuracy drop 0 | Đã chứng minh | Không sweep float16/float32; giữ BF16 cố định |

Hai endpoint đã hoàn tất: cap20 cho decode 18 ms nhưng queue thảm họa; cap40 cho queue sạch
nhưng TBT quay lại 26 ms. Fixed cap không tạo được điểm Pareto tốt hơn best, nên branch đóng;
không thử 32/48/60/80. Evidence này trở thành input cho adaptive scheduler design.

### 7.4 Phase C — Prefill/Prefix Và Cache, Mục Tiêu 50–70

| ID | Status | Exact config delta/package | Hypothesis | Expected metric signature | Promote gate | Rollback/branch decision |
| --- | --- | --- | --- | --- | --- | --- |
| `V-PREFIX-3072` | `completed` | Base 47.52 + `--max-num-batched-tokens=3072`; snapshot `submission-score-48_67.compose.yml`; SHA-256 `dce4a164c29c25ff791cd2e6d42f32352d4d73d52355906d60c144f0a1962316` | Giảm scheduler steps và giữ boundary 6144 | Actual: score 48.67; TBT 21 ms; TTFT 1870/2743 ms; passed SLO 84 | Vượt gate +1.15 | Promote theo score, nhưng hypothesis TTFT thất bại; không tăng budget tiếp |
| `V024-BUDGET-2048` | `completed` | Base 59.10, đổi `--max-num-batched-tokens=3072` -> `2048`; snapshot `configs/vllm/submission-score-65_06.compose.yml`; SHA-256 `e8acfc79438e87922f8fbbaf8b295b65cfc980ea4d523e8a3f85e315620c143b` | Chunk prefill nhỏ hơn tăng fairness và giảm TTFT median | Actual: score 65.06; TTFT 266/1256 ms; TBT 26 ms; 120 SLO; 0 failed | Vượt gate +5.96 | Promote. Xu hướng score vẫn tăng khi giảm budget; đo frontier 1024 trước concurrency |
| `V024-BUDGET-1024` | `completed` | Best 65.06, đổi `--max-num-batched-tokens=2048` -> `1024`; snapshot `configs/vllm/submission-score-64_34.compose.yml`; SHA-256 `8aaf6a64f0669894a92203be9cb9329ac1dc33c423d20aff29f398fe8f163ed4` | Fair scheduling mạnh hơn có thể kéo TTFT gần floor | Actual: score 64.34; TTFT 277/1242 ms; TBT 27 ms; accuracy drop portal 3; penalty 1 | Thất bại -0.72 | Rollback 2048. Cả p50 và TBT xấu hơn nên 1024 đã vượt optimum; không thử budget thấp hơn hoặc 1536 |
| `V-PREFIX-6144` | `canceled` | Tăng budget 3072 -> 6144 | Ban đầu nhằm cache gần trọn common prefix trong một step | 3072 đã làm TTFT p50/p95 xấu 8.5%/22.4% | Không còn plausible upside >=3 | Không nộp; dữ liệu cho thấy prefill monopolization tăng theo budget |
| `V-KV-FP8` | `completed` | Base 48.67 + `--kv-cache-dtype=fp8 --calculate-kv-scales`; snapshot `configs/vllm/submission-score-28_18.compose.yml`; SHA-256 `08676e0bd197086fa356089bf99160ee898c21e9124fdfb75921a324a7074845` | FP8 KV có thể tăng admission/cache capacity và giảm TTFT | Actual: score 28.18; TTFT 1817/2105 ms; TBT 30 ms; failed 0; accuracy drop portal 2; penalty 1 | Thất bại rất xa | Hủy toàn bộ FP8 KV branch: TTFT tail tốt hơn nhưng TBT +9 ms xóa 20.49 điểm; không thử no-scale hoặc partial-layer |

Độ nhạy dưới đây dùng công thức official trên một cặp latency đại diện, không phải dự báo
score portal từ percentile:

| TTFT đại diện | TPOT/TBT đại diện | Request score đại diện |
| ---: | ---: | ---: |
| 266 ms | 26 ms | 67.73 |
| 350 ms | 25 ms | 65.74 |
| 400 ms | 24 ms | 66.15 |
| 400 ms | 23 ms | 69.59 |
| 500 ms | 23 ms | 64.23 |
| 600 ms | 23 ms | 59.38 |

Kết luận: budget optimum là 2048. Cap20 chứng minh TBT có thể xuống dưới floor nhưng một-burst
capacity tạo queue thảm họa. Cap40 là candidate cuối của compose scheduler vì được suy ra từ
completion time 7.08 giây và burst period 5 giây. Nếu cap40 không thắng, fixed-cap branch đóng;
không nội suy thêm cap vì fixed admission không thể đồng thời tối ưu queue và decode.

### 7.5 Phase D — Kernel/Runtime Ceiling, Mục Tiêu 70–90

| ID | Status | Exact package | Hypothesis | Expected metric signature | Promote gate | Rollback/branch decision |
| --- | --- | --- | --- | --- | --- | --- |
| `CUSTOM-ADAPTIVE-SCHED` | `local-prototype` | `custom_runtime/qwen35_adaptive/`: subclass vLLM `AsyncScheduler`; first-token priority + completion cohort; Dockerfile pinned exact v0.24 base; workflow `.github/workflows/build-adaptive-image.yml`; digest-only compose renderer | Default có TTFT 266/TBT 26; cap20 có TTFT 3499/TBT 18; cap40 có TTFT 292/TBT 26. Policy cho mọi prefill/first-token chạy, sau đó ưu tiên request ít output token còn lại nhất để cohort cũ hoàn thành trước | Local pure-policy/renderer suite 29 tests pass; mục tiêu portal TBT 20–23 ms, TTFT p50 <250–400 ms, p95 <1000–1500 ms, ERC 1 | Chưa được nộp: exact image phải build amd64, smoke-import scheduler trong container, public anonymous pull và projected score >=70 | Root giữ best 65.06. Local Docker daemon tắt và disk chỉ còn 14 GiB nên smoke-build chuyển sang manual GHCR workflow; cần explicit approval trước remote push |
| `CUSTOM-INFLIGHT-PREFIX` | `conditional` | Trên adaptive scheduler winner, generic coalescing cho request đồng thời có shared token prefix: một prefix leader materialize KV/GDN state rồi followers reuse trước khi diverge | 20 request mỗi burst có common 6396-token prefix; prefix cache thông thường không loại toàn bộ duplicate work khi requests đến gần đồng thời | TTFT p50 <=100–180 ms; p95 <300–600 ms; TBT <=21–23 ms | Chỉ nộp khi implementation generic theo token prefix và score projection >=80 | Đây là lever chính cho 80–95; cấm hardcode prompt/hash/turn. Nếu accuracy hoặc failure gate hỏng: rollback adaptive winner |
| `MTP-1` | `conditional` | Winning v0.24 scheduler package + đúng một speculative token; không đổi scheduler cùng lượt | TBT 26 ms còn upside decode đáng kể, nhưng trước hết phải chốt budget/concurrency | Chỉ thử sau budget/concurrency retune nếu TBT vẫn >=24 ms | Plausible upside >=3 trước khi nộp; actual +1 để promote | Nếu throughput hoặc TTFT xấu: hủy toàn bộ MTP branch, không sweep số token |
| `TRTLLM-CHECK` | `conditional` | Chỉ tạo candidate khi upstream TensorRT-LLM có Qwen3.5 dense hybrid/GDN support và OpenAI serving path tương thích | Engine compile/capture sâu hơn có thể nâng ceiling vượt vLLM/SGLang | Local boot và output contract sạch trước portal | Không nộp chỉ để kiểm tra support | Nếu thiếu GDN, prefix cache hoặc image public phù hợp CUDA 12.x: đóng nhánh |

### 7.6 Phase E — North Star 90–100

Score 100 không phải mục tiêu có thể hứa bằng compose tuning. Theo công thức, nó đòi gần
như mọi request có TTFT <=100 ms và TPOT <=20 ms. Với 20 prompt dài đến cùng lúc và các
burst chồng nhau, mốc này cần thay đổi performance ceiling chứ không phải thêm flag.

| Score band | Metric signature cần hướng tới | Loại công việc cần thiết |
| --- | --- | --- |
| 20–30 | Đã vượt | BF16 GDN state |
| 30–50 | Đã vượt ở v0.22.1 | BF16 GDN state và initial batching |
| 50–70 | **Hiện tại 65.06:** default TBT 26/TTFT 266; cap20 TBT 18/TTFT 3499 | Cap40 là fixed-cap gate cuối; sau đó dừng compose scheduler |
| 70–80 | TBT 20–23 ms; TTFT p50 <250–400 ms, p95 <1000–1500 ms | Custom adaptive admission/scheduler trên vLLM v0.24 |
| 80–95 | TBT 20–22 ms; TTFT p50 100–180 ms, p95 <300–600 ms | Generic in-flight shared-prefix coalescing và branch-state reuse |
| 95–100 | Hầu hết request sát TTFT 100 ms và TPOT 20 ms | Prefix reuse gần hoàn hảo, adaptive queueing và kernel/runtime tuning; không thể đạt chỉ bằng compose |

Go/no-go cho engineering lớn:

- Cap40 là submission compose cuối đã được duyệt trước custom engineering.
- Không hardcode trace hay precompute output; tối ưu phải generic theo prefix/cache/scheduler.
- Mỗi custom-runtime submission phải có expected upside tối thiểu 3 điểm vì cost build,
  pull và boot cao hơn compose-only.
- Thứ tự bắt buộc: cap40 -> adaptive scheduler -> in-flight prefix -> MTP/kernel finishing.
- Mốc 100 là north star, không hứa bằng flag; mỗi phase phải vượt gate metric mới mở phase sau.

## 8. Gate Trước Và Sau Mỗi Submission

### Trước khi nộp

```text
1. Candidate có ID trong tracker và status planned.
2. Expected upside >=1 điểm; ưu tiên >=3 điểm.
3. Chỉ chứa delta/package đã review trong đúng row.
4. docker compose config pass.
5. Image/digest public, CUDA compatible, offline-safe.
6. Root cũ đã có snapshot + SHA-256.
7. Có sẵn lệnh rollback về best base.
```

### Sau khi portal trả metric

```text
failed_count > 0 hoặc penalty < 1
  -> rollback ngay; không promote

accuracy drop portal > 10 điểm phần trăm
  -> penalty phải được đối chiếu; không so số portal trực tiếp với 0.10

score >= best + 1.00, failed_count = 0, penalty = 1
  -> snapshot, ghi history, promote thành base mới

best < score < best + 1.00
  -> ghi history nhưng không promote; tránh tích lũy micro-flag

custom adaptive: TBT <=23, TTFT p50 <=400 ms, p95 <1500 ms và ERC =1
  -> đủ metric signature để cân nhắc portal submission

TBT giảm nhưng TTFT tăng
  -> trade-off expected; chỉ promote nếu final score vẫn >= best + 1

failed/accuracy gate hỏng hoặc projected/actual score <65.06
  -> rollback ngay về base 65.06

score không đổi trong noise
  -> revert; không giữ flag chưa chứng minh giá trị
```

## 9. Những Nhánh Không Tự Động Thử Lại

- Không thêm `--enable-chunked-prefill`: vLLM đã bật mặc định.
- Không thêm `--async-scheduling`: vLLM đã tự bật trong config hiện tại.
- Không quay lại batched-token budget 8192.
- Không dùng concurrent partial prefill trên image đã báo unsupported.
- Không dùng `--enforce-eager`: mất CUDA graphs.
- Không đổi `stream-interval`: có thể làm sai latency streaming.
- Không tăng `max-model-len`: 32768 đã cover max total 27598.
- Không sweep `max-num-seqs=12/8`; cap 20 chỉ được thử một lần nếu budget 2048 chưa thắng.
- Không coi FP8 KV là đòn decode chính vì chỉ 6/24 layer dùng full attention.
- Không nộp MTP trước khi giải quyết TTFT/prefill.

## 10. Evidence Đã Kiểm Tra Cho Roadmap

- vLLM `v0.22.1` có `--mamba-ssm-cache-dtype`; Qwen3.5 mặc định đọc
  `mamba_ssm_dtype` từ model config và cảnh báo khi override.
- Official vLLM `v0.24.0` release có các thay đổi trực tiếp liên quan workload: fused
  Qwen3.5 GDN QK-RMSNorm/RoPE/gate, cải thiện hybrid cache admission/prefix retention và
  sửa mixed prefill+decode cho Qwen3.5. Đây là lý do version A/B có upside lớn hơn micro-flag.
- Image `v0.24.0` được pin ở amd64 CUDA 12.9 manifest digest
  `3de11aaf1d2aa1c6245a93e9279cc10af6d0b9f5eb3b34704fbd099a8ac42c7d`, build commit
  `ee0da84ab9e04ac7610e28580af62c365e898389`; compressed size khoảng 11.31 GiB so với
  8.59 GiB của v0.22.1. Boot risk tăng nhưng thấp hơn image SGLang 16.5 GiB đã pull được.
- SGLang `v0.5.14` có built-in Qwen3.5, `--mamba-ssm-dtype`, `--language-only`, LPM,
  chunked prefill, max-running-requests và online FP8 quantization.
- Digest SGLang CUDA 12.9 trong tracker đã được Docker manifest xác nhận ngày 10/07/2026.
- SGLang CUDA 12.9 runtime khoảng 16.5 GiB compressed; vLLM `v0.22.1` khoảng 8.6 GiB.
  Đây là deployment risk thật, không phải performance metric.
- Source SGLang Qwen3.5 vẫn construct vision model trong conditional-generation class;
  vì vậy `--language-only` chưa được tính là memory win cho tới khi runtime measurement
  chứng minh ngược lại.

Nguồn implementation đã review:

- <https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/model_executor/models/config.py>
- <https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/config/cache.py>
- <https://github.com/vllm-project/vllm/releases/tag/v0.24.0>
- <https://docs.vllm.ai/en/v0.24.0/configuration/engine_args/>
- <https://github.com/sgl-project/sglang/blob/v0.5.14/python/sglang/srt/server_args.py>
- <https://github.com/sgl-project/sglang/blob/v0.5.14/python/sglang/srt/models/qwen3_5.py>
- <https://github.com/sgl-project/sglang/blob/v0.5.14/python/sglang/srt/models/qwen3_vl.py>
