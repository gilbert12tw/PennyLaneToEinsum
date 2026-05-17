# PennyLane Einsum Project Review

Review date: 2026-05-15

## 總結

`pennylane-einsum` 目前有一個方向正確的 MVP：把 PennyLane circuit 中可提供
dense matrix 的 qubit operations 轉成 einsum tensor network，並重建 final
statevector。核心方法務實、容易檢查，也很適合你的目前目標。

目前 P0 問題均已解決（batch API 移除、opt_einsum 升為 runtime dep、
UnsupportedOperationError 已加入）。剩餘主要工作為 wire semantics 測試補齊與
PennyLane 相容性宣告。

## 目前技術棧

- Python package，透過 setuptools 建置。
- PennyLane：建立 circuit 與 tape。
- NumPy：dense tensor materialization 與基本 contraction。
- opt_einsum：runtime dependency，用於所有 contraction（支援 Unicode index label）。
- pytest：目前測試框架。
- uv：目前可用於重現 dev/test 環境。
- PyTorch optional helper：只處理 statevector 之後的 Hermitian expectation
  gradient，不代表 circuit parameter gradient。

## 目前提取方法

目前 converter 採用 direct dense matrix extraction：

1. 在 `qml.tape.QuantumTape()` 中執行使用者 circuit function。
2. 讀取 `tape.operations`。
3. 對每個 operation 呼叫 `op.matrix()`。
4. 將 matrix 轉成 NumPy complex array。
5. 對 `n`-wire gate，把 matrix reshape 成 rank `2n` tensor。
6. 透過 `IndexManager` 連接每個 wire 的 input/output indices。
7. 組出完整 expression：

```text
initial_state_indices, gate_0_indices, gate_1_indices, ... -> final_wire_indices
```

這個方法不是 PennyLane compiler，也不是 decomposition engine。它的優點是簡單且
容易驗證；缺點是支援範圍受到 dense matrix extraction 限制。

## 已驗證行為

這次實際執行：

```bash
uv run --extra dev pytest -q
```

結果：

```text
25 passed, 1 skipped
```

目前測試涵蓋：

- single-qubit gates
- two-qubit entangling gates
- parameterized gates
- three-qubit chains
- Toffoli
- 超過 52 個 einsum index labels 的 circuit
- QK-Learnob-style ansatz integration
- projected kernel 與 single-qubit Hermitian expectation helper
- PyTorch gradient flow through Hermitian observable helper
- state preparation 的基本 failure case

## 主要問題

### ~~P0: Batch API 已暴露但尚未完成~~ ✅ 已解決

`build_batch_einsum` 已從 public API 完整移除（`circuit_to_einsum.py`、`__init__.py`、
測試）。batch 語義設計留待核心穩定後再實作。

### ~~P0: Large-circuit contraction 對 opt_einsum 的依賴不清楚~~ ✅ 已解決

`opt_einsum` 已升為 runtime dependency（`pyproject.toml`）。`contract_einsum` 已簡化
為直接呼叫 `oe.contract`，舊的 numpy fallback 邏輯已移除。

### ~~P0: Unsupported operations 需要明確錯誤訊息~~ ✅ 已解決

新增 `UnsupportedOperationError`（`pennylane_einsum/exceptions.py`），包含 operation
name、wires、原始失敗原因，以及支援範圍說明。`circuit_to_einsum()` 現在會捕捉所有
`op.matrix()` 拋出的例外並包裝成此型別。已加入測試確認 exception type 與訊息內容。

### ~~P1: 文件之前誇大且互相矛盾~~ ✅ 已解決

README、implementation notes 已修正，舊的 Catalyst/MLIR plan docs 已刪除。專案定位
已明確為 dense-matrix unitary converter，不是 full PennyLane compiler。

### P1: Wire semantics 測試不足

目前有基本 mapping，但正式測試尚未覆蓋 named wires、non-contiguous integer wires、
reversed multi-qubit gate wire order、output tensor axis order。

影響：wire order bug 很容易在 tensor network converter 中出現，而且難 debug。

建議：補測試，全部與 `qml.state()` 做 correctness oracle。

### P1: PennyLane templates 與 decomposition 邊界不清楚

許多 PennyLane 使用者會使用 templates 或 custom operations。現在 converter 只處理 tape
上已經存在且可直接 `op.matrix()` 的 operations，不會自動 decomposition。

影響：使用者可能以為 PennyLane 能跑的 circuit 這個 converter 也都能轉。

建議：先明確文件化；未來可考慮提供 `decompose=True`。

### ~~P1: 不支援 circuit parameter autodiff~~ ✅ 已文件化

README 已明確聲明 converter 不保留 through-circuit autodiff，`expval_hermitian_torch`
的限制也已說明（只對 observable tensor 微分，不對 circuit parameters）。

### P2: `scan_unsupported_ops.py` 在 PennyLane 0.44.1 會壞

目前 script 使用 `qml.operation.WiresEnum`，但該 attribute 在 review 環境的 PennyLane
0.44.1 不存在。

影響：README 中若把它列為 user-facing command，使用者會直接遇到錯誤。

建議：修正或從 user-facing docs 移除。它比較適合作為 developer utility。

### P2: Release hygiene 還不完整

目前 package metadata 已存在，但還缺 CI matrix、changelog、compatibility table，以及明確測過的
PennyLane version range。

影響：開源後使用者可能遇到版本相容性問題。

建議：發布前補 GitHub Actions 與版本測試矩陣。

## 優先待辦

### P0: 穩定 public API

- ~~決定 `build_batch_einsum` 要移除、隱藏，或標成 experimental。~~ ✅
- ~~修正 `contract_einsum` 對 `opt_einsum` 的 dependency 與 fallback 行為。~~ ✅
- ~~新增 project-specific conversion exception。~~ ✅
- 確保 README quick start 和測試指令能從乾淨 checkout 執行。

### P1: 強化核心 conversion correctness

- ~~補 named wires 測試。~~ ✅
- ~~補 non-contiguous integer wires 測試。~~ ✅
- ~~補 reversed multi-qubit gate wire order 測試。~~ ✅
- ~~補 randomized circuit vs `qml.state()` 測試。~~ ✅
- ~~補 custom `initial_state` 測試。~~ ✅
- ~~補 unsupported-operation tests，並檢查錯誤訊息。~~ ✅

### P1: 釐清 PennyLane 相容性

- 定義支援的 PennyLane version range。
- 測 minimum supported version 與 latest stable version。
- 決定是否支援 decomposition of templates/custom operations。
- ~~文件中明確聲明 measurements、state preparation、channels 目前不在 MVP 範圍。~~ ✅

### P2: 改善開發與發布流程

- 修正或刪除 `scripts/scan_unsupported_ops.py`。
- 新增 GitHub Actions CI。
- 新增 changelog。
- 新增 examples，且 examples 必須對應已測試支援範圍。
- 新增簡單 compatibility table。

### P3: 核心穩定後再做的功能

- 設計真正的 batch einsum semantics：明確 batch index 或 ellipsis。
- 增加 contraction path API，包裝 `opt_einsum`，之後可考慮 `cotengra`。
- 若需要保留 gradients，再設計 JAX/PyTorch native tensor output。
- 視需求加入 density matrix 或 expectation value tensor-network output。

## 發布建議

目前不建議直接以一般性 PennyLane converter 發布。比較適合先定位為：

> A converter for PennyLane circuits made of matrix-defined unitary qubit
> operations, producing final-state einsum tensor networks.

第一個公開版本建議至少滿足：

- public API 只保留穩定 conversion 與 contraction helpers
- batch API 隱藏或標 experimental
- unsupported operation 有清楚錯誤訊息
- wire-order 與 randomized circuit tests 補齊
- CI 在小型版本矩陣上通過
- 文件與實際測試支援範圍一致
