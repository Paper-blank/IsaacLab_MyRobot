# 外骨骼 Sim2Real 训练与摩擦力随机化实施指南

> **适用项目**：`Isaac-MyExoskeleton-Flat-v0`（Isaac Lab + PhysX）  
> **文档目的**：帮助初学者理解仿真到真实（Sim2Real）迁移，并给出一套可以逐步落地的外骨骼行走训练方案。  
> **文档状态**：方案和代码示例需要在本项目中做小规模实验后才能用于真实硬件。文中“建议”不等于已经验证的实验结论。

## 阅读前说明

本文件位于项目根目录，讨论的是**训练和迁移策略**。`scripts/ruleresult.md` 是另一个文档，主要描述角度、角速度、角加速度的输出格式和变时间间隔记录，两者不要混淆。

本文覆盖用户提供的七篇论文，并在附录中区分预印本和正式版本。部分工作先有预印本、后有会议或期刊版本，年份可能因此不同；文中不臆造实验数字或“本项目已经达到”的效果。需要正式引用时，应以论文首页、出版社页面或作者公开版本再次核对。

全文每个重要概念都先用通俗语言解释“是什么、为什么”，再给出“怎么做”的技术细节。代码片段优先展示项目实际使用的 Isaac Lab 写法；Isaac Gym、MuJoCo、PyBullet 的片段只作为迁移对照，不能直接混用。

## 第一章：什么是 Sim2Real Gap（入门介绍）

### 1.1 生活化理解：仿真地板和真实地板不是同一块地板
 
把仿真器想象成一间非常整洁的实验室：地板平整、摩擦系数固定、每次启动机器人都从同一个姿势开始，传感器也不会延迟。真实世界更像穿着鞋走出实验室：地面可能有灰尘、水分和磨损，鞋底每一步的接触都略有不同，电机有齿隙，传感器读数有噪声，网络还可能晚几毫秒。

**仿真到真实差距（Sim2Real Gap）**，就是策略在这两种“世界”里表现不同的现象。策略不是只记住一组关节角，而是在观察、施加动作、得到反馈的循环中学会控制。如果循环中的物理规律变了，原来刚好有效的动作就可能失效。

### 1.2 哪些差异最容易让外骨骼失败

| 参数或现象 | 通俗解释 | 对行走的直接影响 |
| --- | --- | --- |
| 接触动态/滑动摩擦（dynamic/sliding friction） | 脚已经滑起来后，地面对脚的阻力 | 太小会打滑、推不动身体；太大可能让脚“粘住”而绊倒 |
| 接触静摩擦（static friction） | 脚还没有滑动时，地面抵抗启动滑动的能力 | 决定能否稳定支撑、蹬地和换步；只随机动态摩擦会漏掉这一种情况 |
| 质量（mass）和转动惯量（inertia） | 背包里多放或少放东西，摆腿的“沉重程度”不同 | 同一电机力矩产生的加速度不同，落脚时机也会改变 |
| 关节阻尼（damping） | 关节运动时像有一个速度相关的刹车 | 阻尼过大动作迟钝，过小容易抖动和过冲 |
| 关节/减速器摩擦（joint/transmission friction） | 电机和齿轮箱内部的阻力，不是脚和地面的摩擦 | 小动作可能根本动不起来，换向时出现死区和迟滞 |
| 执行器响应、延迟（actuator dynamics/latency） | 命令发出后，电机不是瞬间到位 | 策略以为脚已到位，实际上脚还在路上，容易错过支撑窗口 |
| 传感器噪声和偏置（sensor noise/bias） | IMU、编码器像有轻微抖动或零点偏移 | 策略可能把噪声当成身体倾斜，产生不必要的修正 |
| 接触模型、地形和碰撞网格 | 仿真的脚底形状、地面平整度与真实鞋底不同 | 接触力、滑移和落脚冲击的分布改变 |

**为什么这些小差异会累积成跌倒？** 行走是一个连续的闭环过程：一只脚的误差会改变身体速度，身体速度又会改变下一只脚的落点。如果每一步误差都朝同一个方向积累，几步后就会超出关节限位或支撑多边形，最终倒地。因此，不能只在仿真标称参数上训练一个“看起来很漂亮”的动作。

### 1.3 用一个数字量化差距

在**同一个任务、相同时间长度、相同初始状态分布、相同奖励定义和相同评估协议**下，可以用下面的粗粒度指标描述差距：

```text
ΔR = Σ R_sim − Σ R_real
   = Σ_t r_sim,t − Σ_t r_real,t
   = R_sim − R_real
```

这里的 `Σ` 表示在相同时间窗内把每一步奖励加起来；也可以写成 `Delta R = sum(R_sim) - sum(R_real)`，两种写法含义相同。

更严格地，如果第 `t` 步奖励分别是 `r_sim,t` 和 `r_real,t`，则
`ΔR = Σ_t r_sim,t − Σ_t r_real,t`。下面的 `R_sim`、`R_real` 指对应时间窗的累计回报；如果数据已经是累计回报，就直接做一次相减，不要重复求和。

- `R_sim`：仿真环境一段轨迹的累计回报（reward）。
- `R_real`：真实机器人对应任务的一段轨迹累计回报。
- `Delta R > 0`：通常表示仿真比真实更“乐观”，真实迁移损失较大。
- `Delta R < 0`：可能是真实数据的奖励计算方式不同，也可能是仿真过于保守；不能单独据此下结论。

这个定义只适合做第一层比较。真实硬件通常拿不到仿真中的全部接触力和精确状态，因此不能机械地复用所有仿真奖励；应先定义由编码器、IMU 和足底传感器可计算的奖励子集或离线代理，并在相同时间窗、初始条件和速度命令下比较。真实评估还应同时记录：

1. **任务成功率**：在规定时长内完成不跌倒行走的比例。
2. **平均前进速度误差**：实际 `v_x` 与目标速度的差。
3. **跌倒率和急停次数**：安全指标优先于回报。
4. **步态稳定性**：左右脚接触时序、滑移率、支撑时间和身体姿态。
5. **多随机种子均值和方差**：例如 `E[R_sim] - E[R_real]`，避免一次轨迹误导判断。

不要把不同奖励尺度的两个实验直接相减；如果奖励尺度不同，应先统一定义，或补充相对指标：

```text
relative_gap = Delta R / (abs(R_sim) + epsilon)
```

其中 `epsilon` 是防止除零的小数。这个指标仍然不能替代安全测试。

### 1.4 倒地后是否应该 reset？

如果训练目标是“站立并向前行走”，倒地通常应视为当前 episode 失败，并立即结束该 episode、清理状态、重新生成初始姿态。否则策略会继续在躺倒状态收集数据，可能学到“躺着也能拿到存活奖励”或“从地面爬行”的另一项任务。

在本项目中，`bad_orientation` 以约 65° 的姿态偏差判断倾倒，`reset_scene_to_default` 在 `mode="reset"` 时恢复场景和关节目标。因此，**倒地 reset 的基本机制已经存在**。仍需检查三点：

- 倒地环境是否真的触发了 `terminated`，而不是只触发时间截断；
- 终止惩罚是否足以抵消存活、站立等奖励；
- reset 后是否清空了动作、观测归一化和记录器的历史，避免下一 episode 继承上一 episode 的速度。

如果以后要训练“主动起身”，应另建起身任务或单独的 gait mode，不能让同一个倒地处理逻辑含糊地同时承担“行走”和“起身”两个目标。

## 第二章：近五年高质量论文总结

### 2.1 论文信息和阅读口径

下面七篇来自用户提供的论文清单，条目年份处于 2021--2026 窗口（其中一些是预印本）。主表严格使用这七篇，不为了凑年份加入未经核验的条目。对于预印本与正式版本年份可能不一致的论文，正文会同时注明两种口径。

### 2.2 DROPO: Sim-to-Real Transfer with Offline Domain Randomization（2023 正式版；2022 预印本）

**通俗解释：** 不再凭经验拍脑袋决定“摩擦系数从多少随机到多少”，而是先收集少量真实轨迹，再离线地调整仿真参数分布，让仿真更像真实世界。可以把它理解成先用真实地面留下的脚印，反推仿真地面应该有多滑。

**技术细节与项目启示：** 该方法在 HockeyPuck 示例中涉及 18 个动力学参数（包括冰球质量、冰球与表面摩擦、接触时间常数等），并对这些参数的分布做优化；这些参数不能直接等同于外骨骼参数。对外骨骼而言，真实机器人短时间行走、刹停和轻微滑移轨迹可以用于校准摩擦、质量、阻尼和延迟的分布。建议先记录真实数据，再决定随机化区间；不要把 `[0, 2]` 之类的宽区间当成默认答案。

### 2.3 Impact of Static Friction on Sim2Real in Robotic Reinforcement Learning（2025 IROS）

**通俗解释：** 很多训练只考虑“脚已经滑动时有多大阻力”，却忘了“脚还没滑时，启动滑动需要多大力”。真实机器人正是在支撑和蹬地的临界瞬间受静摩擦影响，因此遗漏它会让仿真策略在真实地面上突然打滑。

**技术细节与项目启示：** 该工作把静摩擦纳入域随机化参数空间，说明接触参数至少应成对考虑：动态摩擦和静摩擦，并保持合理关系（通常静摩擦不低于动态摩擦）。本项目的 `randomize_material` 已有两个字段，但后续应检查它们是否按 episode 生效、是否覆盖到脚底碰撞形状，以及采样是否产生物理一致的组合。

### 2.4 PolySim: Bridging the Sim-to-Real Gap for Humanoid Control via Multi-Simulator Dynamics Randomization（2025 arXiv 预印本）

**通俗解释：** 不只在一个仿真器里练习，而是在多个物理实现不同的仿真器里练习。就像让学生在不同跑道、不同鞋底上反复训练，减少对某一条“理想跑道”的依赖。

**技术细节与项目启示：** 该方法通过多个异构仿真器进行动力学级别随机化，降低单一仿真器的归纳偏差。当前项目使用 Isaac Lab/PhysX，不能直接把另一个仿真器接进 PPO 就算完成 PolySim；可先用第二个仿真器或离线动力学模型做交叉评估，比较相同策略在摩擦、接触和关节响应上的失败模式，再决定是否投入多仿真器训练。

### 2.5 A Survey of Sim-to-Real Methods in RL: Progress, Prospects and Challenges with Foundation Models（2025 arXiv 预印本）

**通俗解释：** 这类综述像一张“差距地图”：它提醒我们，仿真与真实的差异不只在物理参数，也可能在状态、动作、奖励和观测方式上。只调摩擦而忽略传感器延迟，仍然可能迁移失败。

**技术细节与项目启示：** 可按 MDP（马尔可夫决策过程）的状态、动作、奖励、动力学等维度逐项排查：状态是否可观测、动作是否有延迟、奖励是否能在真实环境计算、动力学参数是否匹配。建议为本项目建立 gap 清单，而不是只维护一个“摩擦随机化开关”。

### 2.6 Robot Learning from Randomized Simulations: A Review（2022 正式版；2021 预印本）

**通俗解释：** 域随机化很像给学生出一组难度不同的题：题型太单一，学生只会背答案；题型无限混乱，学生连基本方法都学不会。

**技术细节与项目启示：** 综述指出，过度随机化会让策略变得保守，甚至降低真实标称环境的性能。因此随机范围要围绕测量到的真实参数设计，并用课程学习从窄到宽逐步扩展。每次扩大范围都要记录成功率、跌倒率和训练稳定性。

### 2.7 Learning Agility and Adaptive Legged Locomotion via Curricular Hindsight Reinforcement Learning（2023 预印本；正式期刊版 2024）

**通俗解释：** 先让机器人在容易的地面上学会基本步态，再逐渐加入推搡和低摩擦滑动；如果一次失败，利用事后信息分析“它本来完成了什么”，帮助学习更稳健的动作。

**技术细节与项目启示：** 该工作把课程式事后强化学习（Curricular Hindsight Reinforcement Learning, CHRL）中的课程学习（Curriculum Learning）与事后经验回放（Hindsight Experience Replay, HER）结合，并通过外力和低足部摩擦增加难度。外骨骼训练可先固定正前方低速行走，再逐步放宽摩擦范围、加入侧向推力和观测噪声。事后回放不等于把失败轨迹标成成功，必须保持奖励定义和安全终止逻辑一致。

### 2.8 Enhancing Sim2Real Transfer for Torque-Controlled Robots through Real2Sim Dynamics Estimation and Reinforcement Learning（用户清单 2025；IEEE/ASME AIM 正式版 2026）

**通俗解释：** 先用真实机器人的运动反过来估计仿真参数，再训练策略，最后再回到真实机器验证，形成“真实到仿真再到真实”的闭环。这样可以先把仿真的起点校准到真实附近。

**技术细节与项目启示：** 工作中结合轨迹匹配、遗传算法搜索动力学参数和域随机化。对本项目，建议先用低速安全动作测量关节角、速度、驱动电流/力矩和足端接触，再拟合质量、阻尼、摩擦和延迟；之后在拟合值周围做小范围随机化。该流程需要真实数据和辨识代码，不能仅凭一次仿真回报声称完成 Real2Sim2Real。

### 2.9 论文汇总表

| 论文名称 | 年份 | 核心方法 | 关键创新（大白话） | 适用场景 |
| --- | ---: | --- | --- | --- |
| DROPO: Sim-to-Real Transfer with Offline Domain Randomization | 2023 | 离线域随机化、真实轨迹校准参数分布 | 用少量真实数据决定随机化分布，不靠手工猜范围 | 有少量真实轨迹、动力学参数难测的机器人 |
| Impact of Static Friction on Sim2Real in Robotic Reinforcement Learning | 2025 | 静摩擦感知域随机化 | 把“尚未滑动时的阻力”加入随机化，补上传统遗漏 | 腿式/足端接触任务；其他接触任务需另行验证 |
| PolySim: Bridging the Sim-to-Real Gap for Humanoid Control via Multi-Simulator Dynamics Randomization | 2025 预印本 | 多仿真器动力学随机化 | 让策略适应多种仿真器，减少单一仿真器偏差 | 人形机器人、接触丰富的步行和全身控制 |
| A Survey of Sim-to-Real Methods in RL: Progress, Prospects and Challenges with Foundation Models | 2025 预印本 | 从 MDP 各维度系统分类迁移方法 | 提醒同时检查状态、动作、奖励和动力学差距 | 设计完整 sim2real 排查清单 |
| Robot Learning from Randomized Simulations: A Review | 2022 | 域随机化综述 | 指出过度随机化会让策略保守或性能下降 | 选择随机化范围和实验基线 |
| Learning Agility and Adaptive Legged Locomotion via Curricular Hindsight Reinforcement Learning | 2023 预印本/2024 正式版 | 课程学习、事后经验回放、外力/低摩擦扰动 | 从简单步态逐渐训练到滑动和受扰动步态 | 腿式机器人鲁棒行走 |
| Enhancing Sim2Real Transfer for Torque-Controlled Robots through Real2Sim Dynamics Estimation and Reinforcement Learning | 2026 正式版（用户清单标 2025） | 轨迹匹配、参数优化、域随机化 | 先用真实轨迹校准仿真，再训练并回到真实验证 | 力矩控制、参数差异明显的机械臂或外骨骼 |

### 2.10 补充阅读：RMA: Rapid Motor Adaptation for Legged Robots（2021）

这篇 RSS 论文不在用户给出的七篇主清单中，但可以帮助理解“随机化之外如何适应未知动力学”。它在仿真训练时让教师或适应模块看到特权环境变量，再用本体观测历史学习一个适应表示；部署时不需要直接测量这些隐藏变量，而是根据最近的观测和动作历史估计它们。

对外骨骼的启示是：如果真实地面、负载或传动摩擦变化很快，单纯依赖固定策略可能不够，可以考虑增加轻量的历史编码器或适应模块。不过这会改变观测和训练结构，不能把 RMA 的适应模块直接接到当前八维动作 PPO checkpoint 上；应作为独立实验分支验证。

## 第三章：域随机化原理与摩擦力随机化详解

### 3.1 域随机化（Domain Randomization）是什么

**通俗解释：** 不让策略只在一个“标准世界”里考试，而是在一组有变化的世界里考试。例如每个 episode 换一种地面摩擦、机器人重量或电机响应。策略如果只依赖某个精确数字，就会在训练中经常失败；它会被迫学到对变化更稳健的控制规律。

**技术细节：** 设仿真参数为 `p`，从分布 `p ~ P(p)` 采样。每个 episode 重置时固定一组参数，轨迹内通常保持不变；也可以对外力、噪声和延迟按时间间隔采样。目标不是把参数范围无限扩大，而是让 `P(p)` 覆盖真实世界可能出现的区域，并尽量保留物理相关性。策略通常用近端策略优化（Proximal Policy Optimization, PPO）等强化学习算法训练。

典型流程如下：

```text
测量真实参数/轨迹
        -> 选择参数分布 P(p)
        -> reset 时采样一组仿真参数
        -> PPO 等算法训练
        -> 在固定参数矩阵和随机参数矩阵上评估
        -> 用真实失败数据重新校准 P(p)
```

### 3.2 三类摩擦不要混为一谈

#### （1）接触动态/滑动摩擦 `mu_dynamic`

**是什么：** 脚已经相对地面滑动时，接触面抵抗继续滑动的能力。  
**为什么重要：** 它影响脚蹬地时能传递多少水平力，也影响脚拖地时会不会突然滑走。  
**怎么做：** 可以把它作为无量纲接触参数，在仿真中按均匀分布采样。常见起始范围可选：

```text
mu_dynamic ~ U(0.2, 1.0)   # 从较滑到较粗糙
或
mu_dynamic ~ U(0.4, 1.25)  # 更保守的中高摩擦起点
```

这些是实验起点，不是外骨骼鞋底的真值。应通过斜面、拖拽或低速滑移实验校准。

#### （2）接触静摩擦 `mu_static`

**是什么：** 脚还没有开始滑动时，接触面抵抗“启动滑动”的最大能力。  
**为什么重要：** 支撑脚换重心、蹬地和刹停都发生在“快要滑、但还没滑”的临界区。只随机动态摩擦，会让策略从未见过真实的起滑门槛。

**怎么做：** 将静摩擦显式加入参数空间，并与动态摩擦相关采样。通常要求 `mu_static >= mu_dynamic`，但具体接触模型仍需以仿真器定义为准。一个可解释的示例是：

```text
mu_dynamic ~ U(0.2, 1.0)
delta      ~ U(0.05, 0.15)
mu_static  = mu_dynamic + delta
```

如果要限制最大值，应在 `clip` 后再次检查 `mu_static >= mu_dynamic`。不要分别独立采样后得到静摩擦小于动态摩擦的组合。

#### （3）关节/传动摩擦 `joint_friction`

**是什么：** 电机轴承、减速器、密封件和传动机构内部的阻力，发生在关节内部，不是脚与地面的接触摩擦。  
**为什么重要：** 低速启动、换向和保持小角度时，传动摩擦会形成死区；如果仿真关节摩擦为零，策略可能学会真实电机无法实现的细小动作。

**怎么做：** 可先用示例范围 `U(0, 0.3)`，但必须注明参数单位和模型语义。对某些 actuator，这个数可能是力矩损失上限 `[N·m]`，对另一些模型可能是归一化系数，不能把它当作接触 `mu` 直接填写。建议将库仑摩擦和黏性摩擦分开辨识：

```text
tau_friction = coulomb * sign(q_dot) + viscous * q_dot
```

若当前隐式 PD actuator 没有可运行时修改的摩擦字段，应通过自定义 actuator、力矩修正事件、或生成带不同 URDF 动力学参数的资产变体实现；不要用 `randomize_rigid_body_material` 代替关节摩擦。

### 3.3 一个适合并行环境的采样示例

下面是与 Isaac Lab 的批量环境相近的 PyTorch 伪代码。它展示采样逻辑，不会自动把数值写入 PhysX；真正写入应放进经过验证的 EventTerm 或 actuator 接口。

```python
import torch


def sample_friction(num_envs: int, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Sample physically consistent contact and joint friction values."""
    dynamic = 0.2 + (1.0 - 0.2) * torch.rand(num_envs, device=device)
    static_gap = 0.05 + (0.15 - 0.05) * torch.rand(num_envs, device=device)
    static = torch.maximum(dynamic + static_gap, dynamic)
    joint = 0.3 * torch.rand(num_envs, device=device)
    return {
        "mu_dynamic": dynamic,
        "mu_static": static,
        "joint_friction": joint,
    }

values = sample_friction(
    num_envs=8,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
assert torch.all(values["mu_static"] >= values["mu_dynamic"])
```

如果只是验证采样逻辑，下面的 NumPy 版本更直观；正式的并行训练建议使用带明确 `device` 的 Torch 张量和环境自己的随机种子。

```python
import numpy as np


rng = np.random.default_rng(2026)
friction_sliding = rng.uniform(0.2, 1.0)
friction_static = friction_sliding + rng.uniform(0.05, 0.15)
joint_friction = rng.uniform(0.0, 0.3)
print(friction_sliding, friction_static, joint_friction)
```

单个 episode 的真实实现应接收 `env_ids`，只更新正在 reset 的环境，并记录随机种子和实际采样值。不要在 4096 个环境中用 Python 循环逐个写 PhysX 属性；优先使用批量张量接口，或先在 8--32 个环境上做 smoke test。

### 3.4 过度随机化的风险和范围设计原则

**通俗解释：** 如果学生每天面对完全不同的语言、题型和规则，可能学会“什么都不做最不容易错”。策略在仿真中站着不动，反而成了平均意义上最安全的选择。

**技术风险：**

- 摩擦范围过宽，任何单一动作都无法同时适应，梯度信号变弱，训练不收敛；
- 策略变得过度保守，真实标称地面上的速度和步幅下降；
- 独立采样产生不物理组合，例如极低静摩擦配极高动态摩擦；
- CPU 材质写入或过多材质 bucket 使 reset 变慢，训练实际吞吐下降；
- 只在训练时随机化，评估时参数范围却不同，结果无法比较。

**推荐原则：**

1. 先做真实参数测量和标称仿真基线，再设范围；
2. 先窄后宽，每次只扩大一个参数族，便于定位问题；
3. 对有物理关系的量做相关采样，并设置上下限；
4. 每个 episode 记录采样参数、种子和失败原因；
5. 同时保留“固定标称参数评估”和“随机参数评估”；
6. 真实数据足够时，用离线校准方法更新分布，而不是无限扩大范围。

## 第四章：本项目 Sim2Real 训练策略改动方案

### 4.1 先看当前代码已经做了什么

下面的结论来自当前仓库，而不是假设：

| 项目 | 当前设置 | 解释 |
| --- | --- | --- |
| 任务配置 | `source/isaaclab_assets/envs/my_exoskeleton_env_cfg.py` | Manager-based Isaac Lab 环境 |
| 控制周期 | `sim.dt=0.005 s`、`decimation=4`，即 `0.02 s`（50 Hz） | 物理每 5 ms 更新一次，策略每 4 个物理步更新一次 |
| 初始命令 | `lin_vel_x=0.2 m/s`，`lin_vel_y=0`，`ang_vel_z=0` | 第一阶段目标是朝外骨骼正前方直行 |
| 接触摩擦 | `startup`；静摩擦 `[0.6, 1.2]`，动态摩擦 `[0.5, 1.0]`，32 个 bucket | 每次启动环境时随机并预分配材质，当前不是每 episode 重新采样连续值 |
| 基座质量 | `base_link` 当前约 `1.83 kg`，`operation="add"`，增量 `[-0.5, 0.5] kg` | 这是绝对加法，约为该 link 的 ±27%；不等同于标称值的 ±15% |
| 比例-微分（Proportional-Derivative, PD）增益 | 刚度、阻尼均按 `[0.8, 1.2]` 缩放 | 已有一部分执行器随机化 |
| 观测噪声 | policy/critic 当前为一个拼接观测项，`enable_corruption=False` | 还没有按物理量区分的传感器噪声 |
| 倒地处理 | 姿态偏差约 65° 终止；终止后 reset | 适合把倒地定义为行走失败，但要检查日志确认事件真的触发 |

特别注意：URDF 中部分脚部碰撞形状的摩擦值曾设为 `50.0`，而当前 `SceneEntityCfg("robot")` 没有限定 `body_names`，材料事件会尝试覆盖整台机器人碰撞形状。这个差异很大，说明首先要确认“哪个参数最终写入 PhysX”，再谈扩大随机范围；如果目标只是脚地接触，可改为只选 `Lfoot`、`Rfoot`，并做后端 smoke test。
即使只改脚部，最终接触效果还受地面材质和 PhysX 的摩擦组合规则（combine mode）影响；应同时读取脚和地面的最终材质属性，不能只看脚的参数。

### 4.2 改动 1：在 episode reset 时随机化接触摩擦

#### 目标和原则

每次 episode 开始时，为正在 reset 的环境采样一组接触摩擦参数，并在该 episode 内保持不变。这样策略会见到不同地面条件，同时不会因为每个物理步都改变地面而学到不真实的控制规律。

#### Isaac Lab/PhysX 的推荐落地顺序

当前使用的是 `mdp.randomize_rigid_body_material`。先保留 `startup` 做基线，再在少量环境上验证 `reset`：

```python
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab_tasks.core.velocity import mdp
from isaaclab_assets.tasks.robot_contract import ACTUATED_JOINT_NAMES


randomize_material = EventTermCfg(
    func=mdp.randomize_rigid_body_material,
    # 基线阶段可用 "startup"；确认 reset 性能后再改为 "reset"。
    mode="reset",
    params={
        # 只想模拟脚-地接触时，限定到两个脚；若要随机化全身接触，去掉 body_names。
        "asset_cfg": SceneEntityCfg(
            "robot",
            body_names=["Lfoot", "Rfoot"],
            preserve_order=True,
        ),
        "static_friction_range": (0.6, 1.2),
        "dynamic_friction_range": (0.5, 1.0),
        "restitution_range": (0.0, 0.05),
        "num_buckets": 32,
        "make_consistent": True,
    },
)
```

这里的 `mode="reset"` 表示 EventManager 在每个环境 reset 时调用事件，但 PhysX 实现会在事件构造时预采样材质 bucket，reset 时主要是为 `env_ids` 重新分配已有 bucket。它不是每次 reset 都无限创建新材质，也不是连续地生成任意精度的摩擦值。

PhysX 材质写入使用 CPU 张量，并受场景唯一材质数量（约 64000）的限制；官方实现也建议材料随机化优先在初始化阶段使用。因此实际操作应是：

1. 先用 8--32 个环境、16--32 个 bucket 做重复 reset smoke test；
2. 记录 reset 耗时、每个环境实际 bucket 和跌倒率；
3. 性能和稳定性通过后再扩大并行环境；
4. 如果 `reset` 太慢，使用预生成资产变体或自定义 GPU-safe 事件，不要硬把 CPU 写入塞进高频循环。

#### 将静摩擦和动态摩擦关联起来

`make_consistent=True` 能避免一部分不合理组合，但它未必等同于“静摩擦比动态摩擦高固定间隔”。需要固定间隔时，可在自定义事件中采用：

```python
dynamic = uniform(0.2, 1.0)
gap = uniform(0.05, 0.15)
static = dynamic + gap
static = clamp(static, min=dynamic, max=static_upper)
```

如果裁剪到了 `static_upper`，要重新检查 `static >= dynamic`。每个 `env_id` 的最终值应写入日志，便于复现真实失败。

#### 新增关节/传动摩擦

接触材质 API 不会改变电机减速器的摩擦。当前 Isaac Lab 版本提供了 `mdp.randomize_joint_parameters`，可先用以下配置做小规模验证：

```python
# 单独复制此事件时也需要这些导入。
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab_tasks.core.velocity import mdp
from isaaclab_assets.tasks.robot_contract import ACTUATED_JOINT_NAMES


randomize_joint_friction = EventTermCfg(
    func=mdp.randomize_joint_parameters,
    # 该事件使用 CPU 属性写入；确认版本支持后再考虑 reset。
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=list(ACTUATED_JOINT_NAMES),
            preserve_order=True,
        ),
        "friction_distribution_params": (0.0, 0.3),
        # abs 表示直接采样 U(0, 0.3)；add 则是在默认值上再增加。
        "operation": "abs",
        "distribution": "uniform",
    },
)
```

这里的 `0.3` 是工程起始上限，单位和含义由 actuator/物理后端决定；它不是脚地接触的无量纲 `mu`。上例的 `abs` 语义是直接采样 `U(0, 0.3)`；若想在已标定的默认值上增加扰动，才使用 `add`。PhysX 会同时处理静态、动态和黏性关节摩擦，并限制动态值不超过静态值；Newton 的字段支持不同，必须分别验证。若要每 episode 改变关节摩擦，先测 CPU 写入成本；不支持时可预生成多组 URDF 或实现自定义 actuator。

#### 其他仿真器的对照写法

这些片段只用于理解 API 差异，不代表本项目要同时接入三个仿真器。

**旧版 Isaac Gym：** 不同版本可能称为 `actor_params`，更稳妥的通用流程是取得刚体形状属性、修改后写回：

```python
props = gym.get_actor_rigid_shape_properties(env, actor)
for prop in props:
    prop.friction = float(mu_dynamic)
gym.set_actor_rigid_shape_properties(env, actor, props)
```

请以本地 Gym 版本签名为准；不要假定存在统一的 `actor_params["friction"]` 字段。
某些项目封装会把同一组刚体形状属性包成 `actor_params`，因此资料中也常写作“设置 `actor_params` 的 `friction`”；它只是封装层命名，原生接口仍应以当前 Gym 版本为准。

**MuJoCo：** 常见 modder 封装可以写成下面这样，但 `modder.mod` 不是 MuJoCo 官方统一 API：

```python
# 自定义 modder 的示意写法
modder.mod(geom_name, "friction", [mu_dynamic, torsional, rolling])

# 原生模型写法（字段语义为滑动、扭转、滚动摩擦）
model.geom_friction[geom_id, :] = [mu_dynamic, torsional, rolling]
```

**PyBullet：** Bullet 通常没有独立的 `static_friction` 字段，静摩擦由接触求解近似；不能把 Isaac Lab 的静/动态两个数原样搬过去。

```python
p.changeDynamics(
    bodyUniqueId=body_id,
    linkIndex=link_id,
    lateralFriction=float(mu_dynamic),
    spinningFriction=0.01,
    rollingFriction=0.01,
)
```

### 4.3 改动 2：课程学习与摩擦力渐进随机化

#### 为什么不能一开始就用极端范围

训练早期策略还不会站稳和迈第一步。如果同时面对接近冰面的摩擦、很重的身体、很大的延迟和传感器噪声，失败轨迹之间没有共同的改进方向，PPO 很难得到稳定梯度。先让策略在较窄范围内学会基本步态，再逐步加入困难因素，通常更容易收敛。

#### 三阶段工程起点

下表的动态摩擦范围遵循任务要求；静摩擦应在动态摩擦上方留出小间隔。数值是起始假设，不是真实鞋底测量值。

| 阶段 | 训练目标 | 动态摩擦 `mu_dynamic` | 静摩擦建议 | 关节摩擦 | 其他扰动 |
| --- | --- | --- | --- | --- | --- |
| 第一阶段（前期） | 先学会站稳、抬脚、向前迈步 | `[0.6, 1.0]` | `dynamic + U(0.05, 0.10)` | `U(0, 0.10)` 或先固定 | 不加或极小侧向推力，观测噪声小 |
| 第二阶段（中期） | 适应常见地面和轻微滑移 | `[0.4, 1.2]` | `dynamic + U(0.05, 0.15)` | `U(0, 0.15)` | 加小外力、质量 ±10% |
| 第三阶段（后期） | 覆盖极端但仍可能出现的情况 | `[0.2, 1.4]` | `dynamic + U(0.05, 0.15)`，再做上限裁剪 | `U(0, 0.30)` | 质量 ±15%、PD、噪声、延迟和推力联合随机化 |

阶段推进可按全局控制步数，也可按最近窗口的成功率推进。以下阈值只是工程起点，需按任务重新调参：例如最近 100 个 episode 成功率 `>=0.8` 且跌倒率 `<=0.1` 才进入下一阶段。

#### 两种可落地实现

**方案 A：三份配置、三次训练（推荐初次实施）。** 分别建立 `stage1`、`stage2`、`stage3` 配置，每阶段保存 checkpoint，再用上一阶段 checkpoint 初始化下一阶段。优点是范围和结果容易复现，缺点是需要管理多个训练目录。

**方案 B：单次训练内动态调度。** 新增一个任务自有的 curriculum term，根据 `common_step_counter` 或成功率修改当前 stage，并让自定义 reset 事件按 stage 采样。Isaac Lab 的 `EventTermCfg` 中固定的 tuple 不会自动在运行时变成新的 bucket；如果修改了范围，必须重新采样 bucket 或预先建立覆盖全范围的 bucket 并按 stage 选择。不要只修改配置对象后就假定 PhysX 已经改变。

伪代码如下：

```python
def choose_friction_stage(control_step: int, success_rate: float) -> int:
    if control_step < 16_000:
        return 1
    if control_step < 32_000 or success_rate < 0.8:
        return 2
    return 3


def reset_with_stage(env_ids, stage: int):
    ranges = {
        1: (0.6, 1.0),
        2: (0.4, 1.2),
        3: (0.2, 1.4),
    }
    dynamic = sample_uniform(ranges[stage], env_ids)
    static = dynamic + sample_uniform((0.05, 0.15), env_ids)
    write_contact_material(env_ids, static=static, dynamic=dynamic)
```

实际代码应把 `sample_uniform`、`write_contact_material` 替换成当前后端支持的批量 API，并记录 `stage` 和每个环境的样本。

### 4.4 改动 3：摩擦之外的动力学随机化

#### 质量和转动惯量

**通俗解释：** 真实外骨骼可能安装电池、绑带或不同型号的支架，重量和重心会变化。  
**怎么做：** 先以标称质量的 ±15% 为最终阶段目标。Isaac Lab 中应使用相对缩放，而不是当前的绝对加法：

```python
randomize_base_mass = EventTermCfg(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",  # CPU 写入，先在 startup 验证
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        "mass_distribution_params": (0.85, 1.15),
        "operation": "scale",
        "distribution": "uniform",
        "recompute_inertia": True,
    },
)
```

`operation="scale"` 才表示“乘以 0.85--1.15”。如果以后要每 episode 变化，先确认后端是否支持 reset 写入和惯量重算；质量和惯量常常更适合启动期随机化或资产变体。

#### 关节阻尼和 PD 增益

关节阻尼可在标称值附近做比例缩放，PD 刚度和阻尼可以先用当前 `[0.8, 1.2]`，后期再根据真实电机响应扩大。随机化应保持正值，并检查最大力矩、最大速度和关节限位仍然满足硬件约束。

#### 观测噪声、偏置和延迟

当前项目把 49 维 actor 观测拼成一个 `ObservationTermCfg`，并关闭了 `enable_corruption`。如果直接给整个向量加同一尺度噪声，角度、速度、IMU 和接触力的单位会混在一起。推荐拆成按物理量的观测项，或在 `exoskeleton_observation` 内按切片注入：

```python
q_noisy = q + torch.randn_like(q) * 0.005       # rad，示例
qd_noisy = qd + torch.randn_like(qd) * 0.03     # rad/s，示例
imu_noisy = imu + torch.randn_like(imu) * 0.02  # 单位按 IMU 定义
```

噪声标准差必须用真实传感器数据估计；还应加入固定偏置（每 episode 一个）和 1--2 个控制周期的随机观测/动作延迟。噪声和延迟不要在没有安全限幅的真实硬件上直接开启。

#### 初始状态和外部扰动

可随机化初始关节角、根部小速度、轻微侧向推力和 yaw 扰动。当前项目已有每 3--6 秒一次的侧向推力事件；它应在课程后期逐渐启用，并记录推力发生时间，方便解释失败。

### 4.5 改动 4：训练、评估和“站着不走”诊断

#### 推荐训练流程

```text
真实参数/轨迹采集
    -> 标称仿真基线
    -> 摩擦和质量等参数随机化
    -> 课程学习逐步扩大难度
    -> PPO 多随机种子训练
    -> 固定参数矩阵 + 随机参数矩阵评估
    -> 低速台架真实测试
    -> 用失败数据校准下一轮分布
```

#### 当前行走奖励的关键点

当前配置已经有朝外骨骼正前方行走的信号：

| 奖励项 | 权重 | 作用 |
| --- | ---: | --- |
| `track_linear_velocity` | `+1.5` | 跟踪命令的线速度 |
| `fixed_forward_walk` | `+3.0` | 第一阶段固定 `0.2 m/s`，低速静止时不给满分 |
| `track_yaw_velocity` | `+1.0` | 第一阶段目标 yaw 速度为 0，抑制转向 |
| `walk_contact`、`walk_human_knee`、`walk_swing_height` | `+1.0`、`+1.5`、`-15` | 鼓励交替接触、合理膝角和抬脚 |
| `leg_phase_symmetry`、`leg_amplitude_match` | `+2.0`、`+1.0` | 左右腿相位和幅度协调 |
| `termination_penalty` | `-200` | 倒地/终止的强惩罚 |
| 姿态、身高、关节速度、动作率、限位和功率项 | 负权重 | 防止抖动、过冲和不安全动作 |

完整权重以 `MyExoskeletonRewardsCfg` 为准；上表只列诊断最相关项。站着不走时，优先检查：

1. `fixed_forward_walk` 是否在日志中非零，且 gait mask 是否处于 walk 阶段；
2. `base_velocity` 是否真的发出固定 `0.2 m/s` 命令，而不是被 curriculum 改成了 standing；
3. 线速度和前进奖励是否被终止、姿态或功率惩罚完全抵消；
4. 脚部接触传感器名称、body 顺序和接触阈值是否正确；
5. 倒地 reset 是否在策略刚开始迈步时过早触发；
6. 训练是否只跑了约 100 个 PPO iteration，尚未经历第一阶段的完整 curriculum。

不要单纯把前进奖励权重无限增大。先画出每个 reward term 的平均值和方差，确认“站着”究竟是速度奖励没生效、惩罚过强，还是步态事件没有触发。

#### 评估指标

每个 checkpoint 至少在以下矩阵上评估：

- 摩擦：低、中、高三个固定点，以及随机分布；
- 质量：标称、`-15%`、`+15%`；
- PD/阻尼：标称和边界；
- 噪声/延迟：关闭、标称、偏高；
- 随机种子：至少 3 个，建议 5 个。

记录成功率、平均回报、跌倒率、前进速度误差、横向速度、yaw 偏差、足端滑移率、每步能耗、reset 耗时和 `Delta R`。真实测试还要记录急停次数和保护器触发次数。

#### 命令行参数和配置覆盖

命令中出现多个 `--`，通常表示多个独立的命令行选项；每个选项由入口脚本的 parser 解析。例如 `--task` 是任务选择，`--imu 0` 是脚本自定义的传感器开关。项目训练脚本还会把形如 `env.events.randomize_material.mode=reset` 的 token 转交给 Hydra 作为配置覆盖，它不是 `--` 选项。

示意用法（具体启动器和引号按本地脚本为准）：

```bash
python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Flat-v0 \
  env.events.randomize_material.mode=reset \
  env.ruleresult_enabled=false
```

不要把 `--task=...`、`--imu=0`、`env.events...=...` 混写成一个字符串；shell 会先按空格分词，parser/Hydra 再分别处理。

## 第五章：实施步骤与注意事项

### 5.1 Step-by-step 实施清单

#### Step 1：锁定安全边界和数据记录格式

先确定关节角、速度、力矩、IMU、接触力的单位、方向和限位；配置软件急停、速度/加速度/力矩 watchdog。真实测试必须有人旁监督，仿真成功不是实机安全证明。

#### Step 2：采集少量真实基线轨迹

在低速、吊具或安全支撑下采集站立、微小屈膝、单步、刹停和轻微滑移轨迹。记录硬件时间戳，不能只记录程序循环时间。保存软件版本、固件、地面材料、鞋底状态和电池电量。

#### Step 3：建立标称仿真基线

关闭大范围随机化，确认机器人能在标称参数下站立和向前迈步。绘制关节角、足端接触、根部速度和每个 reward term；如果标称任务都不稳定，不要直接扩大随机范围。

#### Step 4：校准质量、PD、阻尼和接触基准

用真实轨迹匹配根部加速度、关节响应和接触时刻，先调质量、惯量、PD、阻尼，再调摩擦。把真实估计值和不确定性写成表格，作为随机化中心和上下界。

#### Step 5：加入接触动态/静摩擦随机化

先用 `startup` 和少量 bucket 验证当前 PhysX 配置；重复 reset 后检查每个环境的材质是否改变。通过后才尝试 `mode="reset"`，并测量吞吐和内存。

#### Step 6：加入关节传动摩擦

使用 `mdp.randomize_joint_parameters` 或自定义 actuator。确认写入的是关节摩擦而不是 geom 摩擦；检查静、动态和黏性项的后端差异。先从 `U(0, 0.1)` 开始，再扩展到 `U(0, 0.3)`。

#### Step 7：启用三阶段课程

阶段一只做窄摩擦范围和基本行走；阶段二加入常见滑移、质量变化和小推力；阶段三才加入极端摩擦、关节摩擦上限、噪声和延迟。每个阶段保存独立 checkpoint 和随机化日志。

#### Step 8：加入观测噪声、偏置和延迟

根据传感器数据设定每种量的噪声，不要给所有观测使用同一个数。先在仿真中做关闭/标称/偏高三档对比，确认策略没有因为噪声学成静止。

#### Step 9：训练 3--5 个随机种子

固定代码、配置和评估矩阵，只改变随机种子。比较平均值和方差，而不是挑表现最好的一次；保存每次训练的 git/配置快照、随机化分布和 checkpoint。

#### Step 10：做仿真固定点和随机分布双重评估

固定点评估回答“在某个已知地面上能否稳定”；随机分布评估回答“策略是否泛化”。两者都要包含不带推力和带推力的测试，避免只在平均场景上报高分。

#### Step 11：低速台架和短时实机测试

先不追求完整步态：站立、重心转移、单步、两步、低速直行逐级放开。限幅和 watchdog 独立于策略；策略只输出目标，不得绕过硬件安全层。

#### Step 12：用真实失败反向校准

把“起步打滑、换向抖动、落脚延迟、过早终止”等失败按类型标注，再更新对应参数分布。优先缩小明显不可能的范围或增加缺失的参数，不要无差别扩大全部随机化。

### 5.2 常见问题与解决方案

| 现象 | 常见原因 | 排查和处理 |
| --- | --- | --- |
| 加随机化后完全不收敛 | 范围一次扩大太多，或样本不物理 | 回到标称/阶段一；逐个参数族加入，检查 reward 方差和 `NaN` |
| 训练 100 轮后一直站立 | 前进奖励未触发、命令被 standing 覆盖、步态 mask/接触传感器错误 | 打印命令、`fixed_forward_walk`、gait mode 和接触力；确认第一阶段 `0.2 m/s` |
| 仿真成功、真实打滑 | 漏了静摩擦，或脚底实际材质与 URDF 不同 | 测量起滑力；相关采样 static/dynamic；核对最终 PhysX 材质 |
| 脚像粘在地面上 | 动态摩擦过高、脚底碰撞网格或 URDF `mu` 未被覆盖 | 查看实际 shape properties，先固定中等摩擦再调碰撞网格 |
| 静摩擦小于动态摩擦 | 两个参数独立采样或裁剪顺序错误 | 先采 dynamic，再采 gap；写入前断言 `static >= dynamic` |
| reset 变得很慢 | PhysX CPU 材质/关节属性写入或 bucket 太多 | 小环境压测，减少 bucket，改为 startup/资产变体或自定义 GPU-safe 事件 |
| 机器人真实抖动 | PD 太高、延迟未建模、速度/加速度噪声过大 | 降低增益和动作尺度，加入延迟模型，使用独立安全限幅 |
| 真实机器人比仿真慢 | 质量、阻尼、关节摩擦或电机饱和未校准 | 先做 Real2Sim 参数拟合，再扩大随机范围 |
| 噪声一加就不走 | 噪声单位过大，所有观测使用同一标准差 | 按角度/速度/IMU/接触分组，逐档增加并检查归一化 |
| 训练和评估结果矛盾 | 训练与评估的摩擦范围、命令或 reset 逻辑不同 | 固定评估配置，打印每 episode 实际参数和命令 |
| 关节摩擦随机化没有效果 | 把 geom 摩擦当成 joint 摩擦，或写入后端不支持 | 使用 `randomize_joint_parameters`/actuator，读取仿真中的最终 joint property |

### 5.3 推荐参考实现

- **legged_gym（Isaac Gym 生态，常被泛称为 IsaacGymEnvs 示例）**：可参考其 [`legged_robot_config.py`](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py) 中的 `domain_rand` 组织方式，重点学习参数范围、reset 事件和日志记录；不要直接复制旧版 API 名称。
- **DROPO**：可检索其公开实现 [gabrieletiboni/dropo](https://github.com/gabrieletiboni/dropo)，重点看如何用少量真实轨迹估计参数分布，而不是只抄一个摩擦区间。
- **Sim2real-gym**：按用户清单给出的项目名检索其 seeded domain randomization、per-link mass、joint friction 和 PD gain 示例；公开仓库名称可能有多个同名项目，使用前应核对作者、版本和许可证，不要把搜索到的任意仓库当成唯一官方实现。
- **本项目代码**：以 `source/isaaclab_assets/envs/my_exoskeleton_env_cfg.py` 的 `MyExoskeletonEventsCfg`、`MyExoskeletonCurriculumCfg` 和 `MyExoskeletonRewardsCfg` 为唯一实际接口来源；先核对本地 Isaac Lab 版本，再修改事件。

### 5.4 真实硬件安全注意事项

随机化训练不会自动保证实机安全。部署时必须保留关节软限位、速度/加速度/力矩限制、通信 watchdog、姿态跌倒检测和物理急停；低速台架测试时应有机械支撑和人员监督。任何论文或仿真成功率都不能替代风险评估、硬件验收和逐级放开测试。

## 附录：可追溯论文和版本说明

以下链接用于核对版本，不把论文报告的数字当作本项目结果：

1. [DROPO: Sim-to-Real Transfer with Offline Domain Randomization（正式版 2023；预印本 2022）](https://doi.org/10.1016/j.robot.2023.104432)
2. [Impact of Static Friction on Sim2Real in Robotic Reinforcement Learning（IROS 2025）](https://doi.org/10.1109/IROS60139.2025.11246554)
3. [PolySim: Bridging the Sim-to-Real Gap for Humanoid Control via Multi-Simulator Dynamics Randomization（2025 预印本）](https://arxiv.org/abs/2510.01708)
4. [A Survey of Sim-to-Real Methods in RL: Progress, Prospects and Challenges with Foundation Models（2025 预印本）](https://arxiv.org/abs/2502.13187)
5. [Robot Learning From Randomized Simulations: A Review（正式版 2022；预印本 2021）](https://doi.org/10.3389/frobt.2022.799893)
6. [Learning Agility and Adaptive Legged Locomotion via Curricular Hindsight Reinforcement Learning（预印本 2023；正式期刊版 2024）](https://doi.org/10.1038/s41598-024-79292-4)
7. [Enhancing Sim2Real Transfer for Torque-Controlled Robots through Real2Sim Dynamics Estimation and Reinforcement Learning（IEEE/ASME AIM 正式版 2026；用户清单标为 2025）](https://doi.org/10.1109/AIM65483.2026.11658256)
8. [RMA: Rapid Motor Adaptation for Legged Robots（RSS 2021，补充阅读）](https://doi.org/10.15607/rss.2021.xvii.011)

## 最终执行检查表

- [ ] 倒地会终止当前行走 episode，并清理历史状态。
- [ ] 静摩擦、动态摩擦、关节摩擦分别建模，没有把 geom 摩擦当成 joint 摩擦。
- [ ] 每个 episode 的随机参数和随机种子可以追溯。
- [ ] 课程三阶段按成功率/步数推进，且范围扩大前有 smoke test。
- [ ] 质量随机化使用相对缩放（例如 `0.85--1.15`），而不是误把绝对加法称为 ±15%。
- [ ] 观测噪声按物理量分组，并在真实传感器数据上校准。
- [ ] 至少 3--5 个随机种子、固定参数矩阵和随机参数矩阵都完成评估。
- [ ] 实机测试保留限位、watchdog、急停和人工监督。
