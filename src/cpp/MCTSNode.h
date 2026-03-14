#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace AlphaZero
{
    /**
     * WDL 三元组封装：(draw, player1_win, player2_win)，绝对视角。
     * 消除手动操作散落的 d/p1w/p2w 三变量。
     */
    struct WDLValue
    {
        float d = 0.0f, p1w = 0.0f, p2w = 0.0f;

        static constexpr WDLValue draw()     { return {1, 0, 0}; }
        static constexpr WDLValue p1_wins()  { return {0, 1, 0}; }
        static constexpr WDLValue p2_wins()  { return {0, 0, 1}; }
        static constexpr WDLValue uniform()  { return {1.f/3, 1.f/3, 1.f/3}; }

        /// 计算当前落子方视角的 Q 值
        float q(int turn) const {
            return (turn == 1) ? (p1w - p2w) : (p2w - p1w);
        }

        /// 逐层衰减：向 uniform(⅓,⅓,⅓) 混合
        WDLValue decayed(float gamma) const {
            constexpr float u = 1.0f / 3.0f;
            return {gamma*d + (1-gamma)*u, gamma*p1w + (1-gamma)*u, gamma*p2w + (1-gamma)*u};
        }
    };

    /// 将 check_winner() 返回值转换为 WDLValue
    inline constexpr WDLValue winner_to_wdl(int winner) {
        if (winner == 1)  return WDLValue::p1_wins();
        if (winner == -1) return WDLValue::p2_wins();
        return WDLValue::draw();
    }

    // ======== SearchConfig ========

    /**
     * MCTS 搜索配置。所有搜索参数集中管理，运行时可安全修改。
     * BatchedMCTS 拥有一份，所有子树通过 const 指针共享。
     */
    struct SearchConfig
    {
        float c_init = 1.25f;           ///< PUCT 初始常数
        float c_base = 19652.0f;        ///< PUCT 对数基数
        float dirichlet_alpha = 0.3f;   ///< Dirichlet 噪声 alpha（≤0 禁用噪声）
        float noise_epsilon = 0.25f;    ///< Dirichlet 噪声混合权重 ε
        float fpu_reduction = 0.4f;     ///< First Play Urgency 衰减系数
        float mlh_slope = 0.0f;         ///< Moves Left Head 斜率（0=禁用，Connect4 用）
        float mlh_cap = 0.2f;           ///< MLH 最大影响上限
        float score_utility_factor = 0.0f; ///< KataGo-style 分差 utility 权重（0=禁用，Othello 用）
        float score_scale = 8.0f;       ///< 分差 atan 映射的缩放分母
        float value_decay = 1.0f;       ///< Backprop 逐层衰减（1.0=禁用）
        bool use_symmetry = true;       ///< 是否启用随机对称增强
        int vl_count = 1;              ///< Virtual loss 强度（每次 VL 模拟给子节点加的访问次数）
    };

    // ======== Edge ========

    /**
     * 搜索树边：连接父节点到子节点的动作。
     * prior 和 noise 存储在边上（而非节点），因为它们是动作的属性。
     */
    struct Edge
    {
        int32_t action = -1;    ///< 动作索引
        int32_t child = -1;     ///< 子节点索引（-1=延迟分配，未访问）
        float prior = 0.0f;     ///< NN 策略先验概率 P(a)
        float noise = 0.0f;     ///< Dirichlet 噪声（仅根节点边有效）
    };

    // ======== MCTSNode ========

    /**
     * MCTS 搜索树节点。64 字节 cache-line 对齐。
     *
     * WDL 使用累加值（W_d, W_p1w, W_p2w），按需除 N 得到均值。
     * 子节点通过 Edge 池间接引用，不再嵌入固定大小数组。
     */
    struct alignas(64) MCTSNode
    {
        // WDL 累加（绝对视角）—— 12 bytes
        float W_d = 0.0f;
        float W_p1w = 0.0f;
        float W_p2w = 0.0f;

        int32_t n_visits = 0;       ///< 真实访问次数 N —— 4 bytes
        int32_t n_inflight = 0;     ///< VL in-flight 计数（仅 VL 搜索期间非零）—— 4 bytes

        float M_sum = 0.0f;         ///< 剩余步数累加 —— 4 bytes

        // Edge 池引用 —— 8 bytes
        int32_t num_edges = 0;      ///< 合法动作数（展开后设置）
        int32_t edge_offset = -1;   ///< Edge 池中的起始偏移

        // 树结构 —— 8 bytes
        int32_t parent = -1;        ///< 父节点索引（-1=根节点）
        int32_t parent_edge_idx = -1; ///< 父节点中指向此节点的 Edge 索引

        // 元数据 —— 3 bytes
        int8_t turn = 1;            ///< 该节点的落子方（1 或 -1）
        bool is_expanded = false;
        bool is_terminal = false;

        // 终局 WDL 缓存（仅 is_terminal 时有效）—— 12 bytes
        float term_d = 0.0f;
        float term_p1w = 0.0f;
        float term_p2w = 0.0f;

        // 总计: 12+4+4+4+8+8+3+1(pad)+12 = 56 bytes, 对齐到 64

        /// WDL 均值（绝对视角）
        WDLValue mean_wdl() const {
            if (n_visits == 0) return WDLValue::uniform();
            float inv = 1.0f / static_cast<float>(n_visits);
            return {W_d * inv, W_p1w * inv, W_p2w * inv};
        }

        /// 当前落子方视角的 Q 值
        float mean_q() const { return mean_wdl().q(turn); }

        /// 指定视角的 Q 值
        float mean_q(int t) const { return mean_wdl().q(t); }

        /// 剩余步数均值
        float mean_M() const {
            return (n_visits == 0) ? 0.0f : M_sum / static_cast<float>(n_visits);
        }

        /// 终局 WDL
        WDLValue get_terminal_wdl() const { return {term_d, term_p1w, term_p2w}; }

        /// 设置终局 WDL
        void set_terminal_wdl(WDLValue w) { term_d = w.d; term_p1w = w.p1w; term_p2w = w.p2w; }
    };

    // ======== NodePool ========

    /**
     * 节点 + 边的线性池分配器。
     * 两个独立的 flat vector 提供 cache-friendly 的顺序访问。
     * 通过 int32_t 索引引用，避免指针和堆碎片。
     */
    class NodePool
    {
    public:
        explicit NodePool(int initial_node_cap = 2048, int initial_edge_cap = 0)
            : node_count_(0), edge_count_(0)
        {
            if (initial_edge_cap <= 0) initial_edge_cap = initial_node_cap * 4;
            nodes_.resize(static_cast<size_t>(initial_node_cap));
            edges_.resize(static_cast<size_t>(initial_edge_cap));
        }

        /// 分配一个新节点，返回索引。不足时自动扩容。
        int32_t allocate_node()
        {
            if (node_count_ >= static_cast<int32_t>(nodes_.size()))
                nodes_.resize(nodes_.size() * 2);
            int32_t idx = node_count_++;
            nodes_[idx] = MCTSNode{};
            return idx;
        }

        /// 分配一组连续边，返回起始偏移。
        int32_t allocate_edges(int32_t count)
        {
            int32_t offset = edge_count_;
            int32_t new_count = edge_count_ + count;
            if (new_count > static_cast<int32_t>(edges_.size()))
                edges_.resize(std::max(static_cast<size_t>(new_count),
                                       edges_.size() * 2));
            for (int32_t i = 0; i < count; ++i)
                edges_[offset + i] = Edge{};
            edge_count_ = new_count;
            return offset;
        }

        MCTSNode& node(int32_t idx) { return nodes_[idx]; }
        const MCTSNode& node(int32_t idx) const { return nodes_[idx]; }

        Edge& edge(int32_t offset, int32_t i) { return edges_[offset + i]; }
        const Edge& edge(int32_t offset, int32_t i) const { return edges_[offset + i]; }

        void reset() { node_count_ = 0; edge_count_ = 0; }
        int node_count() const { return node_count_; }
        int edge_count() const { return edge_count_; }

    private:
        std::vector<MCTSNode> nodes_;
        std::vector<Edge> edges_;
        int32_t node_count_ = 0;
        int32_t edge_count_ = 0;
    };
}
