# TRUCE-Rec 项目总目标（GOAL — 长期执行者必读）

> 本文件是项目的标准 goal 注入文本。任何长期执行 agent（codex/GPT-5.5 等）每个 session
> 必须先读本文件 + `CLAUDE.md` + `AGENTS.md` + `docs/CALM_REC_RUNBOOK.md` +
> `docs/method_calm_rec_spec.md` + `docs/PROJECT_MEMORY.md`（按此顺序），再行动。
> 规则都已写好，**不要重新推导、不要问用户规则**。

## 一、最终目标（唯一的成功定义）

把 TRUCE-Rec 推进到**顶会投稿就绪**，即同时满足：

1. **主表**：自研方法 CALM-Rec（Calibrated trust over Attribute-anchored Latent
   Multi-intent）在 8 域 × 8 冻结官方 baseline 的同候选协议下完整对比。
   beauty 必须先 SOTA（beat ProEx NDCG@10 = 0.1506），且 **falsifiability 合同全过**
   （spec §8：full_beats_sota + trust_beats_placebo + multi_intent_beats_K1 +
   reliability_signal_real），再 rollout 其他 7 域（共享超参，≥5/7 域赢，带 CI）。
2. **三个必做附加实验**（`docs/followup_experiment_plan.md`）：observation（用 baseline
   模型做不确定性动机图）、ablation（spec §8 的 ~22 行预注册表）、hyperparameter 扫描
   + 一张框架总览图。
3. **论文**：基于真实 metrics 写完（`paper/` 已有 intro/method/notation/related 草稿），
   过 `docs/top_conference_review_plan.md` 的内部评审 gate。

## 二、不可违反的纪律（违反 = 工作无效）

- 实验**只在服务器**跑；本地只开发/commit；绝不从服务器 push。
- 证据分级（L0 smoke → L5 paper-result）：backend=hashed 是合同检查不是证据；
  paper 数字只能来自 backend=qwen 的正式跑。
- 评测数据只认 `data/processed/frozen_week8_beauty`（从 pony external_tasks 转换，
  973 test + 973 valid × 101 候选）。**uncertainty-llm4rec 的面板候选集不同，禁止使用**。
- 泄漏控制（spec §7）：train-only 统计去 held-out targets；ρ/τ 只在 validation 上选；
  test 永不参与选择。代码已强制，保持。
- 诚实条款（spec §9）：只在 973 beauty 赢的方法 = 撤回；ρ 不如 placebo → 去掉 shrinkage
  headline；K=1≈K=4 → 降级卖点。stage-2.5 gate AUC≤0.55 → 自动丢 trust headline。
- 每完成一步：更新 agentmemory + PROJECT_MEMORY.md 等文档 + commit/push。
- 与 Pony/TGL-Rec 完全独立：共享 baseline 证据与数据 setting，绝不混方法代码。

## 三、当前状态快照（2026-06-13 凌晨）

- 方法已锁（ARIS 9.0/10）且**全部实现**：纯 Python 推理核心（40 单测绿）+ 真实 Qwen3-8B
  encoders（`src/llm4rec/methods/calm_qwen.py`）+ Stage-B LoRA 训练循环
  （`scripts/train_calm_stage_b.py`，完整 CALMLossSpec）+ 高效正式评测器
  （`scripts/eval_calm_beauty.py`：信号缓存一次，Stage-C ρ 网格/2.5 gate/ladder/placebo/
  paired bootstrap 全从缓存推导）。torch↔python 打分 1e-9 parity 有测试钉死。
- 注意一个已记录的简化：π 的 D_uk 项推迟（γ=0，记录在 artifacts meta）；若 beauty
  不 SOTA，这是首批复查点之一。
- 冻结数据已就绪（服务器 `data/processed/frozen_week8_beauty` + weak labels
  `outputs/calm/beauty_frozen`，672/1184 items 有主导 facet）。
- **服务器队列 v3 进行中**（`~/projects/gpu_queue3.sh`）：TGL full 跑完后自动执行
  Stage-B smoke（曾因 tau 图复用 bug 失败，已修复 commit b7c4ede）→ Stage-B 全量。
- 分支：`feat/calm-qwen-stage-b`（GitHub 已推送；服务器同分支检出）。

## 四、决策树（严格按此执行）

1. **等 Stage-B 完成**（`outputs/calm/beauty_frozen/stage_b/`：lora/ + extras + anchors +
   meta + 训练日志）。失败 → 读 log 修 bug → 重跑（先 smoke 后全量）。
2. **正式评测**：`python scripts/eval_calm_beauty.py --processed-dir
   data/processed/frozen_week8_beauty --weak-labels outputs/calm/beauty_frozen
   --stage-b-dir outputs/calm/beauty_frozen/stage_b --qwen-model-path
   ~/models/Qwen/Qwen3-8B --out outputs/calm/beauty_frozen/eval --sota-ndcg10 0.1506`
   → 读 `calm_rec_verdict.json`。
3. **判定**：
   - 四 checks 全过 + ≥0.1506 → beauty SOTA，进 rollout。
   - 不过 → 按 spec §11 失败模式表对症（ρ 塌缩/锚变流行度代理/意图坍缩等都有 detector
     和 fix），先尝试修复重训；仍不行 → 三席 ARIS 重设计（runbook §4，模板在
     `outputs/method_redesign_discussion/iterations/`）。**不准 rollout 失败方法**。
4. **8 域 rollout**：每域跑 converter（`scripts/convert_frozen_task.py`，源 = pony
   external_tasks 各域导出）→ weak labels → Stage-B → eval。逐域串行。
   注意 K=1 消融的严格版需独立 Stage-B run（--n-intents 1）。
5. **附加实验 + 总览图 → 填表 → 论文 → 内部评审 gate → 告知用户投稿就绪。**
   正式分数按 `source_event_id,user_id,item_id,score` schema 导出存档。

## 五、运维要点（同 TGL，实测）

- 服务器 python：`~/miniconda3/envs/tglrec-lora/bin/python`（复用环境，代码独立）。
- 服务器不通 GitHub：bundle-over-scp 同步（bundle 放 `~/projects/TRUCE-Rec/`，
  fetch 到 `refs/remotes/bundle/calm` 再 reset --hard）。
- GPU 与 TGL-Rec 串行共享；启动前 `nvidia-smi` 确认空闲 ≥40GB；`setsid nohup` + 落盘日志。
- 轻量证据（verdict/metrics/manifest，非多 GB）打包回本地 commit；重产物留服务器。

## 六、汇报格式（每个复杂任务结束时）

what changed / what was tested / complete or blocked / next concrete plan /
current gate toward submission-ready。
