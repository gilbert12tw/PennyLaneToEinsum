# 可訓練觀察量（Trainable Observable）使用指南

這份文件說明本專案最新的功能：**把觀察量（observable）本身當作可訓練參數**。
電路 `|ψ⟩` 固定不動，我們訓練的是「要用哪個方向去量測」。

三個重點：

1. `expectation_value()` — 一行算出 `⟨ψ|O|ψ⟩`，observable 傳 torch tensor 就能反向傳播
2. `PauliDirectionObservable` — 內建的可訓練觀察量，**任何角度值都保證合法**（Hermitian + unitary）
3. 完整訓練迴圈範例 — 用 Adam 把 `⟨H⟩` 訓練到目標值

---

## 1. 基本用法：`expectation_value`

先看不訓練的情況。`expectation_value(circuit_func, observable, n_qubits)`
把電路轉成張量網路，直接縮並出 `⟨ψ|O|ψ⟩`（不會產生 `2**n` 的中間 statevector）：

```python
import pennylane as qml
from pennylane_einsum import expectation_value

def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

# observable 可以用很多種寫法，以下四種等價（都是量 wire 0 的 Z）：
ev = expectation_value(circuit, "ZI", n_qubits=2)             # Pauli 字串
ev = expectation_value(circuit, {0: "Z"}, n_qubits=2)         # dict：wire → Pauli 字元
ev = expectation_value(circuit, qml.PauliZ(0), n_qubits=2)    # PennyLane observable
import numpy as np
Z = np.array([[1, 0], [0, -1]])
ev = expectation_value(circuit, {0: Z}, n_qubits=2)           # dict：wire → 自訂矩陣
```

最後一種是關鍵：**observable 可以直接給矩陣**。矩陣是 numpy 就走 numpy；
矩陣是 **torch tensor 就自動全部升級成 torch**，梯度可以一路流回矩陣的來源。

## 2. 為什麼矩陣不能隨便給：Hermitian 與 unitary

一個合法的觀察量必須是 **Hermitian**（`H = H†`），期望值才會是實數。
如果你直接把一個 `2×2` 的 `nn.Parameter` 矩陣拿去訓練，梯度更新後它馬上就
不是 Hermitian 了，得手動投影回合法的矩陣空間，麻煩又不穩定。

本專案的解法是**參數化整個合法流形**：不訓練矩陣本身，而是訓練兩個實數角度
`(θ, φ)`，由角度生成矩陣：

```
H(θ, φ) = cos(θ)·Z + sin(θ)·(cos(φ)·X + sin(φ)·Y)

        = [[ cos(θ),          e^{-iφ}·sin(θ)],
           [ e^{iφ}·sin(θ),  -cos(θ)        ]]
```

這是單位向量 `n` 方向的 Pauli 矩陣 `n·σ`，所以對**任何** `(θ, φ)` 它同時滿足：

- **Hermitian**：`H = H†` → 是合法觀察量，期望值必為實數
- **Unitary**：`H·H† = I`，特徵值恰好是 `±1` → 和 Pauli Z 一樣可解讀成 ±1 的量測

換句話說：優化器怎麼走都不會走出合法區域，**完全不需要投影或約束**。

## 3. `PauliDirectionObservable`

上面的參數化已包成 `torch.nn.Module`：

```python
import torch
from pennylane_einsum import PauliDirectionObservable

obs = PauliDirectionObservable(theta=0.1, phi=0.1)

obs()             # forward：回傳當下的 2×2 複數 torch 矩陣 H(θ, φ)
obs.theta         # nn.Parameter，可訓練
obs.phi           # nn.Parameter，可訓練
list(obs.parameters())   # 直接餵給 optimizer
```

幾個特殊值可以幫助理解——它就是「可以連續旋轉的 Pauli 矩陣」：

| θ | φ | H(θ, φ) |
|---|---|---------|
| 0 | 任意 | Z |
| π/2 | 0 | X |
| π/2 | π/2 | Y |

把 `obs()` 的輸出當一般矩陣塞進 `expectation_value` 即可：

```python
ev = expectation_value(circuit, {0: obs()}, n_qubits=1)   # torch scalar，帶梯度
```

因為 observable 是 torch tensor，`opt_einsum` 縮並時走 torch backend，
`ev.backward()` 的梯度會流回 `obs.theta` 和 `obs.phi`。
（注意：梯度只通過**觀察量**，不通過電路的 gate 參數——本專案不對 gate 做 autodiff。）

## 4. 完整訓練範例

目標：電路固定，訓練 `(θ, φ)` 讓 `⟨H⟩` 逼近目標值 `-0.9`。
（完整可執行版本在 `examples/train_observable.py`）

```python
import pennylane as qml
import torch
from pennylane_einsum import PauliDirectionObservable, expectation_value

# 1. 固定的電路（|ψ⟩ 不變）
def circuit():
    qml.Hadamard(wires=0)
    qml.RY(0.7, wires=0)
    qml.RZ(0.3, wires=0)

# 2. 可訓練的觀察量 + 一般的 torch optimizer
obs = PauliDirectionObservable(theta=0.1, phi=0.1)
opt = torch.optim.Adam(obs.parameters(), lr=0.1)

target = -0.9

# 3. 標準的 torch 訓練迴圈
for step in range(200):
    opt.zero_grad()
    ev = expectation_value(circuit, {0: obs()}, n_qubits=1)  # ⟨ψ|H(θ,φ)|ψ⟩
    loss = (ev - target) ** 2
    loss.backward()      # 梯度流回 obs.theta / obs.phi
    opt.step()

print(float(expectation_value(circuit, {0: obs()}, 1).detach()))  # ≈ -0.9
```

執行 `python examples/train_observable.py` 可以看到收斂過程：

```
target <H> = -0.9
start:  <H> = +0.4870  (theta=0.100, phi=0.100)
step  40: <H> = -0.89..  loss = ...
...
final:  <H> = -0.9000
converged to target: True
```

每一步 `obs()` 都會用**更新後的角度**重新生成矩陣，而生成出來的矩陣永遠是
合法觀察量——這就是第 2 節說的「訓練角度、不訓練矩陣」的好處。

## 5. 進階：自己定義可訓練觀察量

`PauliDirectionObservable` 只是一種參數化。只要你的 `nn.Module` forward
輸出的矩陣**恆為 Hermitian**，就可以套同樣的模式。例如訓練一組 Pauli 係數：

```python
import torch
import torch.nn as nn

X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

class PauliSumObservable(nn.Module):
    """H = a·X + b·Y + c·Z，係數 a, b, c 是實數 → 恆為 Hermitian。"""
    def __init__(self):
        super().__init__()
        self.coeffs = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))

    def forward(self):
        a, b, c = self.coeffs
        return a * X + b * Y + c * Z

obs = PauliSumObservable()
ev = expectation_value(circuit, {0: obs()}, n_qubits=1)
```

差別在於這個版本**不保證 unitary**（特徵值不一定是 ±1，範數會隨係數變大），
而 `PauliDirectionObservable` 因為限制在單位球面上，兩個性質都保住了。
選哪個取決於你要的搜尋空間。

## 附錄：`expectation_value` 其他參數

```python
expectation_value(
    circuit_func,          # 一般的 PennyLane 電路函式（不用 QNode）
    observable,            # 上面介紹的任何一種格式；也接受 qml.Hamiltonian（逐項線性相加）
    n_qubits,
    params=None,           # 傳給 circuit_func 的參數（支援 batch）
    lightcone=True,        # 只縮並 observable 因果錐內的 gate，大電路可大幅加速
)
```
