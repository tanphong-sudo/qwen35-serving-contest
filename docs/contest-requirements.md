# Yêu Cầu Chính Thức — Phase 1

Tài liệu này chỉ lưu luật thi và contract chính thức. Không đặt giả thuyết tối ưu hay
kết luận benchmark ở đây. Khi luật thay đổi, cập nhật file này trước.

## 1. Mục Tiêu

Serve model cố định do BTC cung cấp trên hạ tầng cố định, replay production trace và tối
đa hóa Effective Request Score trong khi vượt accuracy gate GPQA Diamond.

Không train, fine-tune hoặc thay checkpoint/tokenizer.

## 2. Công Thức Chấm

```text
Score = 100 × ERS × f(Δ)

ERS = (1/N) Σ S_request,i

S_request = 0
  nếu request lỗi, timeout hoặc trả về 0 token

S_request = 0.5 × s_ttft + 0.5 × s_tpot
  nếu request thành công

s_ttft = clamp((1500 - TTFT_ms) / (1500 - 100), 0, 1)²
s_tpot = clamp((45 - TPOT_mean_ms) / (45 - 20), 0, 1)²
```

Tham số Phase 1:

| Tham số | Giá trị |
| --- | ---: |
| TTFT floor | 100 ms |
| TTFT ceiling | 1500 ms |
| TPOT floor | 20 ms |
| TPOT ceiling | 45 ms |
| Gamma | 2 |
| TTFT weight | 0.5 |
| TPOT weight | 0.5 |

TPOT dùng giá trị trung bình trên toàn stream của từng request.

## 3. Accuracy Gate

Baseline accuracy Phase 1 là `0.4` trên 100 câu GPQA Diamond, greedy/temperature 0.

```text
Δ = baseline_accuracy - team_accuracy

f(Δ) = 1.0                         nếu Δ <= 0.10
f(Δ) = 1.0 - (Δ - 0.10) / 0.06    nếu 0.10 < Δ < 0.16
f(Δ) = 0.0                         nếu Δ >= 0.16
```

Accuracy drop tối đa 10 điểm phần trăm không bị phạt. Từ 10 đến 16 điểm phần trăm bị phạt
tuyến tính; từ 16 điểm phần trăm trở lên score bằng 0.

## 4. Hạ Tầng

| Item | Contract |
| --- | --- |
| GPU | 1 NVIDIA H200 MIG, khoảng 18 GB VRAM |
| CPU | 3 cores |
| RAM | 8 GB |
| OS/CUDA | Ubuntu 22.04, CUDA 12.x |
| Model working assumption | `Qwen/Qwen3.5-2B` |
| Model mount | `/model` |
| Precision gốc | BF16 |
| Timeout toàn lượt | 600 giây |

BTC cố định model hash cho round. Config phải hoạt động với model được mount sẵn, không
được pull model lúc serving.

## 5. Artifact Nộp

- Một `docker-compose.yml`.
- Image phải public và pull được từ môi trường BTC.
- Không dùng `build:` hoặc đường dẫn local.
- Service phải listen trên `0.0.0.0:8000`.
- Endpoint bắt buộc: `/v1/chat/completions` tương thích OpenAI.
- Served model name phải chấp nhận `Qwen3.5-2B`.
- Request body từ trace phải được xử lý nguyên trạng.
- Không phụ thuộc secret hoặc volume local ngoài `/model` do BTC mount.

## 6. Anti-Cheating

Các hành vi bị cấm:

- Hardcode đáp án hoặc nhận diện probe/hidden subset.
- Pre-compute response cho request trong trace.
- Gọi API hoặc network external trong quá trình inference.
- Pull model/tokenizer động lúc runtime.
- Thay đổi tokenizer hoặc tokenizer files.
- Thay checkpoint/model hash do BTC cung cấp.
- Thay arrival timestamp hoặc concurrency của trace.
- Dùng account khác để thu thập hidden trace.

`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` và `VLLM_NO_USAGE_STATS=1` phù hợp với contract
offline và nên được giữ trong mọi submission.

## 7. Điều Kiện Một Submission Hợp Lệ

- Container boot và endpoint ready trước timeout.
- Serve đủ request, không OOM hoặc transport error.
- Không trả output rỗng.
- Context length đủ cho input cộng output budget.
- Không có network call khi serving.
- Accuracy penalty được portal báo rõ.
- Docker Compose parse hợp lệ và chỉ tham chiếu public artifact.

## 8. Checklist Trước Khi Nộp

```bash
docker compose -f docker-compose.yml config -q
```

- [ ] Image/tag hoặc digest public.
- [ ] Không có `build:`.
- [ ] `/model` được dùng làm model path.
- [ ] Port `8000` được expose.
- [ ] `Qwen3.5-2B` là served model name.
- [ ] Offline env được giữ.
- [ ] Không có volume/secret local.
- [ ] `max-model-len` cover trace.
- [ ] Mỗi thay đổi tối ưu có rollback snapshot.
