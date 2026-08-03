# Batched Observable Training and Fixed-Batch Scaling

日期：2026-08-03

## 摘要

本次實驗回答兩個問題：

1. 目前的 Torch einsum 實作能將批次化 observable training 擴展到多大？
2. 固定 batch size 時，可重用的 cuTensorNet contraction 如何隨 qubit 數擴展？

主要結論：

- 16q、depth 4、batch 512 可完成 100 個 training steps。
  `complex64` 約 4,086 samples/s、峰值 21.6 GB；`complex128` 約
  2,876 samples/s、峰值 43.1 GB。
- 24q、depth 1 的 `complex64` 可執行到 batch 128，約
  294 samples/s、峰值 86.5 GB。`complex128` 最大測試成功值為 B96，約
  169 samples/s、峰值 130.0 GB；B112 OOM。
- 24q、depth 4 的 `complex64` 最大測試成功值為 B10，約
  3.86 samples/s、峰值 111.7 GB；B12 OOM。`complex128` 只有 B8 成功，
  B9 OOM。
- 24q/depth4 的容量斷崖主要來自 contraction path 切換。從 B8 增加到 B9 時，
  allocated memory 從約 0.07 GB 跳到 101 GB，吞吐量也同時下降約 54 倍。
- Fixed-batch depth-1 可完成 48q/B256，但吞吐量不會隨 qubit 數平滑變化。
  40q 每個 batch/method 組態的 7 個 replicas 都落入低效 path；42q 的 resident
  結果大多恢復高速，updated-input 仍偶爾出現嚴重離群值。
- 同一 shape 的獨立規劃結果可相差約 100 倍。由於 resident 與 updated-input
  分別規劃 path，兩者的差異不能解讀為單純的 tensor-update overhead。
- 增加 optimizer samples 並執行 autotuning 不保證能提升效能。是否值得使用，
  必須同時考量規劃結果的分布、workspace、初始化成本與預期重用次數。

## Key Takeaways

這批資料最重要的訊息，不是「qubit 越多就一定越慢」，而是目前的 path selection
存在明顯的效能斷崖（cliff）與執行間變異：

1. **Training cliff 可以重現。** 24q/depth4 的 B8→B9，以及 depth1 的 B8→B16
   轉折，在三個資料 seed 中都一致。兩者都伴隨 intermediate memory 大幅增加，
   因此不是單次計時雜訊。
2. **同一條 path 的執行時間穩定，但重新規劃的結果不穩定。** 已選定 path 後，
   重複計時通常很集中；真正的大幅變異出現在重新建立 network 並規劃 path 時。
3. **Resident 與 updated-input 尚未構成配對比較。** 兩種方法各自建立 cuTensorNet
   network，也各自呼叫 planner。Updated-input 偶爾比較快，只代表它取得較好的
   path，不代表 host-to-device update 沒有成本。
4. **單筆 benchmark 不足以代表部署效能。** 應報告多個獨立 planner replicas
   的中位數、範圍或分位數，並保存實際 path 與 workspace 資訊。

## Environment

| 項目 | 值 |
|---|---|
| GPU | NVIDIA H200, 143,771 MiB |
| Driver | 580.65.06 |
| CUDA module | 12.6 |
| Python | 3.11.15 |
| PennyLane | 0.42.3 |
| NumPy | 2.4.4 |
| opt_einsum | 3.4.0 |
| Torch | 2.8.0 |
| CuPy | 14.1.1 |
| cuQuantum Python | 26.6.0 |
| cuTensorNet | 2.13.0 |
| Original sweep base | `d10ea9c515c72d3be8f41bf50aebbc1da6efd2c6` |
| Planner-rerun base | `a2eb67487e8e9f77ea0bb67285b3f9d327aff3d9` |

執行 benchmark 時，worktree 中仍有尚未 commit 的變更。每個結果目錄都保存
`git_commit.txt`、`git_status.txt`，以及受 Git 追蹤檔案的 `worktree.diff`。

新增的 benchmark 與 Slurm scripts 當時尚未被 Git 追蹤，因此不包含在該 diff
中。目前應以 workspace 內的 scripts 作為重現實驗的來源；完成這批變更的 commit
後，才能由單一 revision 完整還原。

## Methodology

### Circuit

使用 `pennylane_einsum.qae_circuit.qae_circuit`：

- 每個 qubit 先施加 Hadamard gate，再施加三個 encoding rotation gates。
- 每層對每個 qubit 施加三個共享權重的 rotation gates。
- 每層最後接上一條 linear CNOT chain。
- 批次輸入的 shape 為 `(B, n_qubits, 3)`。
- VQC 權重的 shape 為 `(layers, n_qubits, 3)`；同一 batch 共用一組權重。

### Observable Training

學生與教師都使用作用在單一 qubit 上的 Pauli-direction observable：

```text
H(theta, phi) = cos(theta) Z
              + sin(theta) cos(phi) X
              + sin(theta) sin(phi) Y
```

設定：

| 項目 | 值 |
|---|---|
| Teacher angles | `(theta, phi) = (2.0, 1.0)` |
| Student initialization | `(0.1, 0.1)` |
| Optimizer | Adam |
| Learning rate | 0.05 |
| Loss | batch mean squared error |
| Measured steps | 100 |
| Untimed warmups | 5 forward/backward passes |
| Seeds | 2026, 2027, 2028 |

Circuit tensors 只建立一次，並在開始訓練前移到 GPU。
`opt_einsum.contract_expression` 也只編譯一次。每個 training step 唯一重新建立的
部分，是可微分的 2x2 observable matrix。計時拆分為：

- forward
- backward
- optimizer update
- total step

所有 GPU 計時都在邊界前後執行 `torch.cuda.synchronize()`。

#### Correctness

- `n_qubits <= 16`：抽樣 3 筆輸入，以 PennyLane `default.qubit` 為參考。
- `n_qubits > 16`：為避免建立指數大小的 statevector，抽樣 3 筆輸入，改以
  非批次 Torch direct-expectation contraction 為參考。
- 額外的 unit test 會比較 batched observable gradient 與逐樣本計算的 gradient。
- `complex64` 的 reference error 約為 `1e-8` 至 `1.5e-6`。
- `complex128` 的 reference error 約為 `1e-16` 至 `1.8e-15`。

### Fixed-Batch Contraction

每個組態都會建立 direct `Z(0)` expectation network，流程如下：

1. 轉換 circuit。
2. 將 tensors 從 host 複製到 device。
3. 建立 cuTensorNet `Network`。
4. 規劃 contraction path。
5. 視設定執行 autotuning。
6. 執行 warmup contractions。
7. 重複執行並計時 contractions。

實驗比較兩種執行方式：

- `batch-reuse`（以下稱 resident）：inputs 保持不變，tensors 常駐 GPU。
- `batch-update`（以下稱 updated-input）：每次從預先建立的 host tensors 更新
  encoding gate buffers，並在該方法自己的 network 內重用 plan。

兩種方法的 topology 與 tensor shapes 相同，但會分別建立 network，也會分別規劃
path。除非明確保存並套用同一條 path，兩者只能視為兩個獨立的規劃結果；它們的
latency 差異不能直接當成 update overhead。

主要的 fixed-batch sweep 使用 3 次 warmup 與 10 次計時；較慢的 depth-4 sweep
則使用 1 次 warmup 與 3 次計時。每次計時都會在前後同步目前的 CuPy stream。
不同計時設定的資料不會用來計算 tuning ratio。Depth-4 regime 圖僅對具有相同
shape 的多次結果取中位數，用途是定性呈現 path regime，而不是進行精確的設定
比較。

#### Correctness

- `n_qubits <= 16`：PennyLane `default.qubit`。
- `n_qubits > 16`：unbatched one-shot cuQuantum direct expectation。
- 所有成功組態的最大絕對誤差都小於 `3.1e-15`。

`cupy_pool_used_bytes` 是組態結束前的 CuPy memory-pool 使用量，不等同整個
process 的峰值記憶體，但可協助辨識 path 產生的大型 intermediate 或 workspace。

## Training Results

### 8- and 16-Qubit Training

表格列出三個 seed 的吞吐量中位數。圖中的線同樣使用中位數，error bars 則表示
觀測到的最小值與最大值；每個 seed 的 loss 與 timing 都保留在原始 CSV 中。

| Qubits | Depth | Batch | Dtype | Samples/s | Peak allocated |
|---:|---:|---:|---|---:|---:|
| 8 | 1 | 128 | complex64 | 17,309 | 0.15 GB |
| 8 | 1 | 512 | complex64 | 66,654 | 0.40 GB |
| 8 | 1 | 512 | complex128 | 57,769 | 0.74 GB |
| 8 | 4 | 512 | complex64 | 38,746 | 0.40 GB |
| 8 | 4 | 512 | complex128 | 36,695 | 0.74 GB |
| 16 | 1 | 128 | complex64 | 9,946 | 0.41 GB |
| 16 | 1 | 512 | complex64 | 34,298 | 1.41 GB |
| 16 | 1 | 512 | complex128 | 25,916 | 2.76 GB |
| 16 | 4 | 128 | complex64 | 2,657 | 5.45 GB |
| 16 | 4 | 128 | complex128 | 1,913 | 10.84 GB |
| 16 | 4 | 512 | complex64 | 4,086 | 21.56 GB |
| 16 | 4 | 512 | complex128 | 2,876 | 43.05 GB |

16q/depth4/B512 的三個 seed 都順利完成。執行 100 個 steps 後，final loss 約為
`5.1e-7` 至 `6.1e-5`。

![Observable training scaling](assets/observable_training_scaling.png)

### 24-Qubit Training

#### Depth 1

| Batch | complex64 samples/s | complex64 peak | complex128 samples/s | complex128 peak |
|---:|---:|---:|---:|---:|
| 8 | 503-513 | 0.07 GB | 521-532 | 0.07 GB |
| 16 | 212 | 11.34 GB | 107 | 22.62 GB |
| 32 | 251-252 | 22.08 GB | 139 | 44.09 GB |
| 64 | 282 | 43.55 GB | 163 | 87.04 GB |
| 80 | 282 | 54.29 GB | 160 | 108.52 GB |
| 96 | 289 | 65.03 GB | 169 | 129.99 GB |
| 112 | 291 | 75.77 GB | OOM | OOM |
| 128 | 294 | 86.50 GB | OOM | OOM |
| 512 | OOM | OOM | OOM | OOM |

#### Depth 4

| Batch | complex64 | complex128 |
|---:|---|---|
| 8 | 206 samples/s | 214 samples/s |
| 9 | 3.79 samples/s | OOM |
| 10 | 3.86 samples/s | OOM |
| 12 | OOM | OOM |
| 14 | OOM | OOM |
| 16 | OOM | OOM |
| 32 | OOM | OOM |
| 64 | OOM | OOM |

24q/depth4 在 B8 使用 intermediate 與 workspace 都較小的 path。Batch size
增加到 B9/B10 後，`complex64` 雖然仍可執行，吞吐量卻立即下降約 54 倍，
記憶體也逼近 H200 上限；B12 則發生 OOM。這種變化無法用線性的 batch-memory
scaling 解釋。

![Observable training memory boundary](assets/observable_training_memory_boundary.png)

### Convergence

所有成功組態的 loss 都有下降。多數實驗在 100 個 steps 後低於 `1e-4`，少數
seed-組態組合的 loss 停在約 `1e-3` 至 `3e-3`。因此，本實驗證明這套流程可以
訓練並量測吞吐量，但不主張所有隨機資料都能在固定 100 個 steps 內完全收斂。

## Fixed-Batch Results

### Depth-1 Qubit Scaling

40q 與 48q 各彙整 7 個獨立 planner replicas，42q 則彙整 6 個。表格格式為
median `[min-max]`；其餘 qubit 數只有單次結果。圖表使用相同的彙整方式。

#### Resident Throughput

| Qubits | B32 | B128 | B256 |
|---:|---:|---:|---:|
| 32 | 17,648 | 54,524 | 96,342 |
| 36 | 15,571 | 49,811 | 87,952 |
| 38 | 14,251 | 46,224 | 82,380 |
| 40 | 97 [87-101] | 218 [217-222] | 289 [288-297] |
| 42 | 13,073 [12,891-13,464] | 42,251 [41,762-43,782] | 76,794 [51,824-77,444] |
| 44 | 12,340 | 39,254 | 72,169 |
| 46 | 11,864 | 38,546 | 68,223 |
| 48 | 11,328 [350-11,699] | 37,443 [35,944-38,714] | 67,520 [65,873-69,899] |

#### Updated-Input Throughput

| Qubits | B32 | B128 | B256 |
|---:|---:|---:|---:|
| 32 | 11,079 | 35,086 | 59,621 |
| 36 | 9,928 | 31,328 | 52,934 |
| 38 | 9,272 | 29,311 | 49,740 |
| 40 | 88 [86-101] | 217 [217-222] | 288 [288-296] |
| 42 | 8,351 [5,231-8,744] | 24,118 [2,092-28,198] | 46,191 [437-46,809] |
| 44 | 7,941 | 25,854 | 34,821 |
| 46 | 7,570 | 24,797 | 41,246 |
| 48 | 7,454 [237-7,688] | 23,907 [388-24,424] | 41,228 [35,841-42,329] |

40q 的 7 個 resident 與 updated-input replicas 都只有約 86-297 samples/s，確認
低效 path 可以穩定重現。42q 的結果則不完全穩定：B128 updated-input 最低只有
2,092 samples/s；B256 updated-input 有兩次落在 437-466 samples/s，其餘四次
則為 46,114-46,809 samples/s。這些局部崩落主要來自 contraction-path selection，
而不是 tensor 尺寸平滑增加所造成的自然退化。

五次新的重跑沒有再次出現舊資料中極慢的 48q updated-input 結果：B32 為
6,861-7,688 samples/s，B128 為 22,472-24,424 samples/s。新的長尾反而出現在
48q/B32 resident，其範圍為 350-11,699 samples/s，相差 33.4 倍；42q/B256
updated-input 更達到 466-46,809 samples/s，相差 100.5 倍。

圖與表彙整所有新舊的獨立規劃結果，並報告中位數與 min-max。這裡的範圍反映
planner outcomes 的差異，而不是同一條 path 內重複計時的雜訊。

![Fixed-batch depth-1 scaling](assets/fixed_batch_depth1_scaling.png)

### Depth-4 Scaling

| Qubits | Batch | Resident samples/s | Update samples/s | 狀態 |
|---:|---:|---:|---:|---|
| 16 | 32 | 260 | 258 | 完成 |
| 16 | 128 | 2,888 | 2,845 | 完成 |
| 16 | 256 | 1,914 | 1,901 | 完成 |
| 24 | 32 | 497 | 148 | 完成 |
| 24 | 128 | 1,162 | 824 | 完成 |
| 24 | 256 | 186 | 509 | 完成 |
| 32 | 32 | 11.2 | 5.9 | 完成 |
| 32 | 128 | 8.6 | 9.9 | 完成 |
| 32 | 256 | 25.7 | - | updated-input path 超過 18 分鐘 |
| 36 | 32 | 2.5 | 4.6 | 完成 |
| 40 | 32 | 0.018 | 0.476 | long-tail 測試完成 |

被停止的 sweep 都保留了先前完成的資料列。停止時若最後一個組態尚未產生計時
資料，便將它視為 timeout，也就是受截尾的觀測值（censored observation）；
該組態不算成功，也不估算吞吐量。表格中的 `-` 表示未取得可用數值，不代表零。

#### Long-Tail Fixed Batches

| Qubits | Batch | Resident samples/s | Update samples/s |
|---:|---:|---:|---:|
| 32 | 8 | 9.08 | 31.52 |
| 32 | 16 | 2.07 | 9.62 |
| 32 | 32 | 14.58 | 11.98 |
| 34 | 8 | 6.17 | 66.39 |
| 34 | 16 | 13.50 | 5.93 |
| 34 | 32 | 16.29 | 4.65 |
| 36 | 8 | 4.06 | 4.26 |
| 36 | 16 | 3.17 | 4.10 |
| 36 | 32 | 2.24 | 5.98 |
| 38 | 8 | 1.99 | 1.48 |
| 38 | 16 | 0.316 | 0.296 |
| 38 | 32 | 0.0347 | 0.251 |
| 40 | 8 | 0.547 | 0.0683 |
| 40 | 16 | 0.0237 | 0.407 |
| 40 | 32 | 0.0184 | 0.476 |

最慢的 40q/B32 resident contraction 中位數為 1,740.6 秒；另一個獨立規劃的
updated-input network 則為 67.2 秒。兩者的 estimated FLOPs 分別約為
`7.33e14` 與 `3.03e13`，相差約 24 倍；觀察到的 latency 也相差約 26 倍。
因此，主要原因是所選 path 不同，而不是 input update 本身。

![Fixed-batch depth-4 regimes](assets/fixed_batch_depth4_regimes.png)

Heatmap 會對相同 `(method, qubits, batch)` 的所有成功規劃結果取中位數。
只有 32q/B32 與 36q/B32 各有兩次結果；其中 32q/B32 使用了不同的
warmup/repeat 設定。因此，這張圖只用來呈現數量級與 path regime，精確數值仍以
前面的表格為準。

### Optimizer Sampling and Autotuning

預設設定（default）與調校設定（tuned）都使用 1 次 warmup 與 3 次計時，並各自
執行 3 個在獨立 process 中建立的 planner replicas。Tuned 使用 100 個 pathfinder
samples 與 3 次 autotune iterations。表格格式為 median `[min-max]`，ratio 是
兩組中位數的比值。

| Qubits | Batch | Method | Default samples/s | Tuned samples/s | Ratio |
|---:|---:|---|---:|---:|---:|
| 24 | 32 | resident | 2,043 [266-3,116] | 533 [343-3,975] | 0.26x |
| 24 | 32 | updated-input | 1,162 [213-2,426] | 1,407 [1,242-4,764] | 1.21x |
| 24 | 128 | resident | 659 [468-772] | 726 [329-2,529] | 1.10x |
| 24 | 128 | updated-input | 510 [243-2,443] | 1,555 [479-1,807] | 3.05x |
| 32 | 32 | resident | 14.7 [7.23-23.5] | 7.64 [1.26-16.4] | 0.52x |
| 32 | 32 | updated-input | 7.64 [2.37-135] | 19.6 [6.10-307] | 2.56x |
| 32 | 128 | resident | 4.24 [2.15-4.65] | 8.13 [1.80-19.1] | 1.92x |
| 32 | 128 | updated-input | 10.1 [3.02-61.7] | 9.21 [6.85-47.2] | 0.92x |

8 個組態中，有 5 個在 tuning 後提升中位數，另外 3 個反而退化，而且兩組的
min-max 高度重疊。相較於舊的單次測量，重跑後有 4/8 個組態連改善或退化的方向
都反轉。換句話說，tuning 改變的是一個帶有長尾的 path-outcome 分布，而不是提供
穩定、固定的加速倍率。

Default planning 花費 12.3-47.1 秒；tuned planning 增加到 51.2-224.8 秒，
另外還需要 0.19-53.8 秒執行 autotuning。即使 tuned 的 execution median 較快，
仍必須根據增加的初始化成本、預期重用次數與最差 workspace 計算 break-even。
單次 throughput ratio 不足以決定部署設定。

![Path tuning speedup](assets/path_tuning_speedup.png)

柱高是 tuned 與 default 吞吐量中位數的比值。黑點和灰色叉號分別代表每個 tuned
與 default replica 的吞吐量，再除以 default median。這種畫法能呈現兩組規劃
結果高度重疊的情況，也避免把彼此獨立的 paths 人為配對成 ratio。

## Reproduction Notes

Benchmark 與繪圖程式位於：

- `scripts/benchmark_observable_training.py`
- `scripts/benchmark_qae_batching_gpu.py`
- `scripts/plot_completion_experiments.py`
- `scripts/slurm_pilot.sh`

報告不逐條列出執行命令。每次執行的 artifact directory 應保存完整命令、Git
revision 與 worktree diff、Python 與套件版本、CUDA/GPU metadata，以及逐步寫入的
CSV。

若要量測 planner variance，每個 replica 都必須使用獨立 process 與獨立輸出檔。
同一條 path 的 execution repeats 只能用來衡量執行時間分布，不能視為獨立的
planner replicas。

本次本機 correctness suite 的結果為 105 tests passed。GPU benchmark 必須透過
scheduler 執行，不可直接在 login node 上執行；完整參數以 artifact 中的
`command.txt` 為準。

## Limitations

- Training 只更新一個共享的 single-qubit observable，不會更新 circuit gates。
- Training 使用 Torch/opt_einsum，inference 則使用 cuTensorNet。兩者採用不同的
  path optimizer，因此不能直接比較 execution latency。
- 超過 16q 後，正確性參考不再是完整的 PennyLane statevector，而是非批次的
  direct-expectation contraction。
- Fixed-batch 的記憶體欄位是 CuPy memory-pool snapshot，不是完整的 GPU peak。
- Path planning 具有隨機性。三到七個 replicas 足以揭露部分長尾，但仍不足以
  精確估計低機率的低效 path 發生率。
- Resident 與 updated-input 尚未共用同一條 path，因此目前的資料無法分離純粹的
  host-to-device update overhead。
- Fixed-batch benchmark 使用預先產生的 host tensors。計時不包含新資料生成、
  gate matrix 建立，也不包含完整 input pipeline 的成本。
- 報告保留所有 OOM 與 timeout，沒有只挑選成功組態。

## Recommendations

1. 下一版 benchmark 應在同一個 contraction network 與同一條 path 上，交錯量測
   resident 與 updated-input，才能估計實際的 update overhead。
2. 部署時，應為每個 `(topology, batch, dtype)` 保存多個候選 paths，而不是只保留
   單次 planner 結果。
3. 候選 paths 應依 execution time、workspace、長尾風險與 setup break-even
   綜合排序。部署前也應排除 40q 這類持續低效的 regime。
4. Observable training 應優先使用 `complex64`；16q/depth4/B512 已有穩定結果。
5. 24q/depth1 建議保守限制在 B64，以保留足夠的記憶體餘裕。實驗成功到 B128，
   但 B64 是部署建議，不是實測上限。
6. 24q/depth4 目前只建議使用 B8。若要支援更大的 batch，需要改善 Torch 的
   contraction path，而不是單純增加 GPU memory。
