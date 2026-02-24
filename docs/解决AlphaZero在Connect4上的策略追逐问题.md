# 解决 AlphaZero 在 Connect4 上的策略追逐（Policy Chasing）问题

**——打破已解决博弈中的价值退化，实现训练收敛**

---

## 一、问题本质：为什么 Connect4 对 AlphaZero 如此困难？

### 1.1 价值退化的根本原因

Connect4 是一个**已解决的博弈**：先手（X）在完美策略下必胜，后手（O）必败。这导致了一个 AlphaZero 框架中的根本性困难：当 AI 执后手时，MCTS 搜索树中**所有动作的 Q 值都趋近于 -1**。

在围棋或国际象棋这类未解决的博弈中，双方几乎总是存在有意义的动作差异——某些走法导致优势，另一些导致劣势。MCTS 的访问分布能够反映这些差异，为策略网络提供清晰的梯度信号。但在 Connect4 中，一旦先手确保了理论胜利，后手的每一步应对都评估为 Q ≈ -1。此时 PUCT 选择公式发生退化：当所有动作的 Q(s,a) 相同时，选择完全由策略先验 P(s,a) 驱动，而产生的访问分布仅仅是网络当前（任意的）信念的镜像。在这些自我强化的目标上训练，会产生一个反馈循环——**网络本质上是在学习预测自己的噪声**。

### 1.2 经验证据

这个问题在社区中有广泛的记录：

- 流行的 `alpha-zero-general` 框架有一个已记录的问题，Connect4 模型会"陷入总是与自己平局"的状态，因为 Q=0 的初始化使未访问节点看起来像平局。

- `cemkaraoguz/AlphaZero-Connect4` 仓库记录了"唯一状态数量随时间递减"以及"目标值的方差可能削弱训练进展"。

- 即使是经过精心调优的 AlphaZero.jl 实现（5个残差块、128通道、600次MCTS模拟），也承认训练出的智能体"在做更长远的战略决策方面仍然不完美"。该项目的作者还指出，"现有的许多 Python 实现的 AlphaZero 都无法学习出一个能打败深度为2的 minimax 基线的玩家"。

- Neumann 和 Gros（2022, 2024）专门研究了 AlphaZero 在 Connect4 上的缩放定律，发现即使是最大的智能体也接近但永远无法达到完美策略。他们2024年的论文还揭示了某些游戏中的**逆向缩放**现象——更大的模型反而表现更差。

### 1.3 问题的形式化描述

设 $s$ 为后手的某个局面，$a_1, a_2, ..., a_k$ 为合法动作。在完美对弈下：

$$Q(s, a_i) \approx -1, \quad \forall i \in \{1, 2, ..., k\}$$

此时 PUCT 公式：

$$\text{UCB}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

由于 $Q$ 项对所有动作几乎相同，选择完全由 $P(s, a)$（策略先验）控制。而策略先验来自上一轮训练的网络，于是访问分布 $\pi$ 仅仅反映了网络的当前偏好，训练目标 $\pi$ 又被用来更新 $P$，形成**自循环**。

---

## 二、KataGo 的方案：通过多重辅助信号打破退化

David Wu 的 KataGo 论文（"Accelerating Self-Play Learning in Go"，2019）以及持续更新的 KataGoMethods.md 文档，包含了目前最全面的解决训练不稳定性的技术集合。核心洞察是：**没有单一技术足够——你需要多个正交的信号**。

### 2.1 辅助训练目标

KataGo 训练网络预测棋盘所有权（终局时每个交叉点由哪个玩家控制）、最终得分分布，以及多个时间尺度的短期未来 MCTS 价值（约6、16和50步后）。

消融实验表明，移除辅助的所有权和得分目标导致了**1.65倍的学习效率下降**——这是所有测试技术中最大的退化。论文解释了原因：仅使用最终的二元结果时，网络只能"猜测"是棋盘局面的哪个方面导致了失败；而有了所有权目标，网络能直接得到关于棋盘哪个区域预测错误的反馈。

**Connect4 的对应方案：**
- 预测终局时每一列由哪个玩家占据
- 预测威胁数量（连续3子的数量）
- 预测终局时各玩家的连接组数量

### 2.2 动态得分效用（Dynamic Score Utility）

KataGo 的 MCTS 效用函数融合了胜负和得分：`总效用 = 胜负效用 + 静态得分 × staticScoreUtilityFactor + 动态得分 × dynamicScoreUtilityFactor`。

动态分量（训练时默认0.30）在根节点当前预测得分处重新居中，使得智能体始终有动力相对于当前预期进行改善。

在 Connect4 中，如果"得分"定义为到终局的步数，那么延迟失败5步的走法会获得比立即输掉的走法更高的效用——即使两者的胜负效用都是 -1。

sigmoid 饱和函数 `f(x) = x / (1 + |x|/b)` 防止不稳定性，同时保持在当前预期值附近的灵敏度。

### 2.3 短期价值目标

KataGo 的辅助价值头预测接下来约6步内指数加权的未来 MCTS 价值，而非仅从远期终局结果学习。

这是经典的偏差-方差权衡：这些预测是有偏的（它们反映当前网络的评估，而非真实值），但方差大幅降低。对于失败局面，6步预测仍能有意义地区分"即将丢失关键棋组"和"维持着脆弱的防守阵型"。

### 2.4 其他 KataGo 技术

- **策略目标剪枝（Policy Target Pruning）**：从训练目标中移除强制探索的 playout，使 Dirichlet 噪声不会污染策略信号。

- **根节点 Softmax 温度 > 1**（1.03–1.25）：防止策略坍缩到单个任意走法，迫使其通过 MCTS 找到真正的差异来锐化。

- **策略惊喜加权（Policy Surprise Weighting）**：自动加权 MCTS 结果与先验不一致的局面，创建一种自我纠正机制：振荡的局面获得更多训练关注。

- 乐观策略头（Optimistic Policy Head）通过预测哪些走法具有最高的"翻盘潜力"，为失败走法创建有意义的排序。

---

## 三、Moves Left Head（剩余步数头）：直接解决"所有走法同样输"的问题

### 3.1 Leela Chess Zero 的方案

Leela Chess Zero（Lc0）在 v0.25（2020年4月）中引入的 **Moves Left Head（MLH）** 是对 Connect4 最直接适用的技术。它添加了第三个输出头，预测游戏剩余的步数。

Lc0 团队精确描述了 MLH 要解决的问题："当根节点 Q 值极端（接近 -1 或 +1）时，所有走法都是赢或输，树搜索无法在它们之间做出选择。搜索非常平坦，因为所有走法具有相同的价值，搜索无法锁定任何一个。"

### 3.2 工作原理

在失败局面中，MLH 通过延迟不可避免的结局来区分走法：导致20步后才输的走法获得比5步就输的走法更高的剩余步数预测。

测试结果显示 MLH 带来了 **47–56 Elo 的提升**，并且游戏平均缩短了70步，同时没有强度损失。

实现使用了 `MovesLeftFactor`、`MovesLeftThreshold` 和 `MovesLeftScale` 参数，仅在 Q 接近 ±1 时激活。

对于 Connect4 来说，最大游戏长度仅为42步，这个信号相对于国际象棋会有更高的区分度。

### 3.3 Lc0 的其他相关创新

- **WDL（Win/Draw/Loss）头**：将标量价值分解为三个概率（通过 softmax），提供比单个 tanh 输出在 -1 附近更丰富的梯度。这正是你的代码中已经采用的方案。

- **Q_ratio 训练**：混合嘈杂的二元游戏结果和每个局面的 MCTS 评估，平滑价值目标。

- **WDL 重缩放/蔑视（Contempt）实现**：根据当前局面对 WDL 输出进行动态重缩放。

- Lc0 训练历史中记录了"策略震荡"现象——与你描述的 policy chasing 完全一致。

---

## 四、MCTS-Solver：对已证明的失败进行形式化处理

### 4.1 经典 MCTS-Solver

MCTS-Solver 算法（Winands, Björnsson, and Saito, 2008）修改了 MCTS 反向传播以传播博弈论价值：当一个子节点返回已证明的胜利（∞）时，父节点立即标记为胜；当一个子节点返回已证明的失败（-∞）时，算法检查所有兄弟节点——只有当每个子节点都是已证明的失败时，父节点才能标记为败。

这防止了在已解决的子树上浪费搜索精力，并通过树传播确定性。

### 4.2 AlphaZero 中的集成

Czech、Korus 和 Kersting（2020）开发了 MCGS（Monte-Carlo Graph Search）终端求解器，将其推广到有向无环图（处理转位），并包含 epsilon-greedy 探索以防止搜索卡住。

他们的 Exact-Win 变体通过使用子树的 WIN/LOSS/DRAW/UNKNOWN 信息来剪枝走法，以 **61% 的胜率**击败了 Leela Zero。

对于 Connect4，由于博弈树规模可控，靠近终局的局面通常可以完全求解。集成 MCTS-Solver 可以显著减少策略追逐发生的比例——已证明解决的子树完全消除了振荡。

---

## 五、探索机制的自适应调整

### 5.1 FPU（First Play Urgency）的自然适应

FPU 设置与失败局面有一个微妙而优雅的交互：

- 绝对 FPU = -1 时：在胜利局面中，未访问节点看起来很糟（Q = -1 vs 胜利走法的 Q > 0），搜索快速收窄
- 但在失败局面中，所有已访问走法的 Q ≈ -1，未访问走法在 FPU = -1 下**看起来并不更差**——搜索自然扩展，探索更多替代方案

Lc0 的 FPU 减少方法（`FPU = parent_Q - reduction`）提供了类似的自适应行为，并且允许更精细的控制。

你的代码中已经实现了动态 FPU：

```python
scale = (1.0 + self.Q) / 2.0
effective_fpu = fpu_reduction * scale
fpu_value = self.Q - effective_fpu * math.sqrt(seen_policy)
```

这在劣势时（Q 接近 -1）会降低 FPU reduction，使搜索更广泛——这正是失败局面中想要的行为。

### 5.2 动态 c_puct

完整的 AlphaZero 论文（非预印本）使用对数增长的 c_puct：`c_puct = base + log((1 + N + c_base) / c_base) × factor`。在高访问计数时，探索增加，防止过早锁定单个走法。

### 5.3 Dirichlet 噪声的调整

`cemkaraoguz/AlphaZero-Connect4` 的实验确认 Dirichlet 噪声保持了"唯一状态比例高于基线"，但减慢了收敛速度。

推荐方案：训练过程中**递减噪声影响**——早期高噪声促进探索，后期低噪声促进利用。你的代码中已经实现了 `noise_steps` 衰减机制，这是正确的方向。

---

## 六、游戏长度缩放奖励：最直接的实用方案

### 6.1 Oracle 团队的方案

Oracle/Medium 博客系列（"Lessons from AlphaZero: Connect Four"）直接面对了失败局面问题，使用**游戏长度缩放的终局价值**：

奖励公式 `1.18 - (9n/350)` 使快速胜利比慢速胜利价值更高，慢速失败比快速失败价值更高（更接近0）。

这创建了一个分级的价值信号，打破了 Q ≈ -1 的退化。比如：
- 第10步就输：value ≈ -0.92
- 第30步才输：value ≈ -0.59
- 第42步才输（最大长度）：value ≈ -0.11

他们的"强"训练模式（最快胜利/最慢失败）在结合神经网络和 MCTS 时，在测试局面上达到了 **99.76% 的准确率**。

### 6.2 实现建议

在你的代码中，这可以在 `game.py` 的 `batch_self_play` 方法中实现：

```python
# 替代原来的 winner_z = np.full(T, winner, dtype=np.int32)
if winner != 0:
    for t in range(T):
        remaining = T - t
        total_moves = T
        # 快胜慢败缩放
        if (winner == 1 and traj['players'][t] == 1) or \
           (winner == -1 and traj['players'][t] == -1):
            # 赢方视角：越快赢越好
            winner_z[t] = 1.0 - 0.3 * (remaining / 42.0)
        else:
            # 输方视角：越慢输越好
            winner_z[t] = -(1.0 - 0.3 * (remaining / 42.0))
```

---

## 七、来自象棋社区的经验：DTM 和 Tablebase 方案

### 7.1 距离到杀棋（DTM）

象棋社区的终局数据库（Endgame Tablebases）使用 DTM（Distance-to-Mate）和 DTZ（Distance-to-Zeroing）提供分级的连续价值信号，自然解决了退化问题。

一个50步后才输的局面与12步就输的局面有本质不同，这种差异可以作为主要或辅助训练目标。

### 7.2 Lc0 对终局数据库的研究

Haque 等人（2022）测试了 Lc0 对 Gaviota 数据库在所有3-4子残局中的表现，发现错误率与**"决策深度"**强相关——需要更长获胜序列的局面有更高的错误率。

即使是强大的 T60 网络在简单残局中也会犯可测量的错误，确认这是一个基本挑战而非训练bug。

有趣的是，他们的监督学习"J-networks"在残局准确率上优于自我对弈的"T-networks"，表明**对已知结果的局面，监督学习可能比自我对弈更高效**。

### 7.3 Connect4 完全解决的优势

由于 Connect4 已被完全解决（John Tromp 的数据库涵盖所有局面），可以考虑：

1. **Tablebase 重标记**：用完美值替换自我对弈的结果
2. **混合训练**：自我对弈数据 + 从完美解中采样的监督数据
3. **DTM 辅助目标**：使用完美解中的"距离到终局"作为辅助损失

---

## 八、成功的开源实现及其参数配置

### 8.1 架构和超参数共识

最成功的开源 Connect4 实现收敛于类似的超参数范围：

| 参数 | 推荐值 | 来源 |
|------|--------|------|
| 残差块数量 | 5 | AlphaZero.jl |
| 通道数 | 128 | AlphaZero.jl, Oracle |
| 参数量 | ~1.6M | 经验甜点 |
| MCTS 模拟次数（训练） | 200-600 | AlphaZero.jl |
| 温度策略 | τ=1.0 前10-20步，然后衰减到 0.3 | 多个实现 |
| Dirichlet α | 0.3-1.0（随分支因子缩放） | AlphaZero 论文 |

AlphaZero.jl 在 Connect4 上使用了 5 个残差块和 128 通道的网络架构。

### 8.2 AlphaZero.jl 的特殊设计

AlphaZero.jl 使用了增长的重放缓冲区（从400K样本增长到1M），以快速洗去早期低质量数据。

该项目还利用水平对称性增强（Connect4 唯一的对称轴）来翻倍有效训练数据。

### 8.3 Expert Iteration（ExIt）

`fast-alphazero-general` 使用 Expert Iteration（ExIt）在笔记本电脑上经过200次迭代在约1天内训练出了强力的 Connect4 智能体。

ExIt 通过将专家（MCTS）和学徒（神经网络）解耦，可以更稳定——策略网络简单地模仿 MCTS，而不是追逐移动的目标。

---

## 九、针对你的代码的具体建议

根据对你的代码库的分析，以下是**按优先级排序的改进建议**：

### 9.1 【高优先级】改进价值目标——游戏长度缩放

你的代码已经有了 `steps_to_end` 和 `steps_head`，这是很好的基础。关键改进是将步数信息融入 MCTS 的**效用函数**中，而不仅仅是辅助损失。

在你的 `Network.py` 的 `predict` 方法中，已经有了类似 KataGo 的 steps-value 混合：

```python
advantage_sign = torch.tanh(5.0 * value_base)
value = (1 - self.lambda_s) * value_base - self.lambda_s * advantage_sign * steps_adjusted
```

**建议增强**：将 `lambda_s` 从 0.1 提高到 0.2-0.3（接近 KataGo 的 dynamicScoreUtilityFactor），并考虑在训练早期使用更高的值，后期逐步降低。

### 9.2 【高优先级】添加 Moves Left Head 的 MCTS 集成

你的网络已经有 `steps_head`，但它目前只用于辅助损失和 `predict` 中的混合。参考 Lc0 的做法，在 MCTS 搜索中直接利用 MLH：

```python
# 伪代码：在 MCTS 回传时调整 value
if abs(value) > 0.9:  # 极端 Q 值
    moves_left_bonus = moves_left_factor * (predicted_moves_left / max_moves)
    if value < 0:
        value = value + moves_left_bonus  # 输得慢 = 稍好
    else:
        value = value - moves_left_bonus  # 赢得快 = 稍好
```

### 9.3 【中优先级】实现策略目标剪枝

在 `batch_self_play` 中，将 Dirichlet 噪声产生的额外探索从训练目标中去除：

```python
# 当前做法：直接用 MCTS visits 作为策略目标
action_probs[valid_mask] = visit[valid_mask] / visit[valid_mask].sum()

# 改进：剪枝低访问走法（去噪声）
visit_threshold = 0.02 * visit.sum()  # 低于总访问2%的走法归零
pruned = visit.copy()
pruned[pruned < visit_threshold] = 0
action_probs[valid_mask] = pruned[valid_mask] / pruned[valid_mask].sum()
```

### 9.4 【中优先级】集成 MCTS-Solver

在你的 C++ `MCTS.h` 的 `backprop` 方法中，添加终局状态的确定性传播：

```cpp
// 在 backprop 中，当检测到终局时
if (is_terminal) {
    // 不展开，直接传播确定值
    // 如果所有子节点都是已证明的失败，标记当前节点为已证明的胜利
    // 这防止了对已解决子树的无效搜索
}
```

### 9.5 【低优先级但有效】添加更多辅助目标

考虑增加：
- **威胁预测头**：预测每一列的"威胁等级"（双方各有多少潜在的四连）
- **短期价值头**：预测6步后的 MCTS 评估值（方差更低的学习信号）

---

## 十、总结：收敛配方

策略追逐问题不是单一故障，而是复合的——退化的 Q 值、嘈杂的策略目标和自我强化的反馈循环共同作用。最有效的策略是组合**来自正交类别的 3-5 种技术**：

1. **打破价值退化**：使用游戏长度缩放奖励（如 `1.18 - 9n/350`）或 Moves Left Head，使"20步后才输"的得分显著高于"5步就输"
2. **添加辅助训练目标**：所有权预测、威胁计数或短期价值预测，在二元胜负无法提供梯度时提供梯度
3. **在 MCTS 中使用 KataGo 风格的得分效用**（动态得分效用因子 ~0.30），使搜索本身能区分同样失败的走法
4. **实现策略目标剪枝**：将 Dirichlet 探索噪声与训练信号解耦
5. **集成 MCTS-Solver**：对终局附近可处理的局面，完全消除已证明解决的子树中的策略追逐

更深层的洞察是：在已解决的博弈中，**失败方的近均匀策略实际上接近正确**——所有走法确实都输。真正的目标不是强制收敛到单个"最佳"失败走法，而是确保价值头准确预测 -1，策略头对**胜利方**的走法准确。接受失败方的优雅近均匀性，同时将训练信号集中在胜利方的走法上，可能是最务实的路径。但当失败方策略质量重要时——例如创建一个给对手最大挑战的防守者——上述技术工具箱可以将混沌振荡的训练转变为收敛到稳定、有战略意义的策略。

---

## 参考文献

- David Wu. "Accelerating Self-Play Learning in Go." (2019) arXiv:1902.10565
- KataGo Methods Documentation: github.com/lightvector/KataGo/docs/KataGoMethods.md
- Leela Chess Zero: Moves Left Head PR #961, WDL rescale/contempt v0.30.0
- Winands, Björnsson, Saito. "Monte-Carlo Tree Search Solver." (2008)
- Czech, Korus, Kersting. "Monte-Carlo Graph Search for AlphaZero." (2020) arXiv:2012.11045
- Haque et al. "On the Road to Perfection? Evaluating Leela Chess Zero Against Endgame Tablebases." ACG 2021
- AlphaZero.jl: jonathan-laurent.github.io/AlphaZero.jl
- Neumann, Gros. "AlphaZero Neural Scaling and Zipf's Law." (2024) arXiv:2412.11979
