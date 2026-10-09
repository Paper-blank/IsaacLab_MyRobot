# `my_exoskeleton.urdf` 零基础详解

这份文档解释外骨骼模型源文件
[my_exoskeleton.urdf](my_exoskeleton.urdf)。它适合第一次接触机器人模型的读者：先说明
每个 XML 标签“是什么”，再说明本项目中它“为什么这样写”，最后说明修改后还要检查
哪些代码。

> 重要：`my_exoskeleton.urdf` 是结构和基础物理参数的源文件。
> `urdf/my_exoskeleton/` 目录中的 `.usd`、`.usda` 是导入器生成的缓存，不能手工修改。
> 环境当前设置了 `force_usd_conversion=True`，启动时会依据 URDF 重新转换模型。

## 1. 先把 URDF 理解成“机器人装配说明书”

URDF（Unified Robot Description Format，统一机器人描述格式）是一种 XML 文件。它不
包含强化学习算法，而是告诉仿真器：机器人由哪些刚体组成、刚体如何连接、每块有多重、
关节能转多少，以及视觉和碰撞外形在哪里。

最重要的两个概念是：

- `link`（连杆）：一块不能自行变形的刚体，例如左大腿、右小腿或脚板。
- `joint`（关节）：把父连杆和子连杆连接起来的运动关系，例如膝关节。

可以把 `link` 想成积木，把 `joint` 想成积木之间的铰链。URDF 把所有积木和铰链连成
一棵树；最上面的 `base_link` 是根，其他连杆都能沿唯一一条路径追溯到它。

本模型的连杆树如下：

```text
base_link
├── fixed: base_to_top_mount -> top_mount
├── base_to_Rturn -> R_turn
│   └── Rturn_to_Rthigh -> Rthigh
│       └── Rthigh_to_Rleg -> Rleg
│           └── Rleg_to_Rfoot -> Rfoot
└── base_to_Lturn -> Lturn
    └── Lturn_to_Lthigh -> Lthigh
        └── Lthigh_to_Lleg -> Lleg
            └── Lleg_to_Lfoot -> Lfoot
```

模型共有 10 个 `link`、1 个固定关节和 8 个可转动的主动关节。名称中的 `R` 表示右侧
（Right），`L` 表示左侧（Left），`thigh`、`leg`、`foot` 分别表示大腿、小腿和脚。

## 2. 单位和坐标系

URDF 不会在每个数字后面写单位，因此必须按统一约定理解。把毫米值误当成米，或者把
角度值误当成弧度，都会让模型严重失真。

| 字段 | 单位 | 示例含义 |
| --- | --- | --- |
| `xyz` | 米（m） | `0.1` 表示 10 cm |
| `rpy` | 弧度（rad） | `1.5708` 约等于 90° |
| 关节 `lower`、`upper` | 弧度（rad） | 旋转关节的位置范围 |
| `mass` | 千克（kg） | 刚体质量 |
| `inertia` | 千克·平方米（kg·m²） | 质量相对转轴的分布 |
| `effort` | 牛·米（N·m） | 旋转关节允许的力矩上限 |
| `velocity` | 弧度每秒（rad/s） | 旋转关节速度上限 |
| `axis` | 无量纲方向向量 | 关节局部坐标系中的旋转轴 |

### 2.1 `xyz` 和 `rpy`

通俗理解：`xyz` 决定“放在哪里”，`rpy` 决定“朝向哪里”。`rpy` 是 roll、pitch、yaw
（滚转、俯仰、偏航）三个旋转量。

技术细节：`<origin xyz="x y z" rpy="r p y"/>` 描述一个坐标系相对另一个坐标系的
变换。`rpy` 不是角度制；例如 90° 必须写成约 `1.5708`，不能写 `90`。

### 2.2 为什么不同位置都有 `<origin>`

同样名为 `<origin>` 的标签，所在位置不同，含义也不同：

| 所在位置 | 它相对谁 | 它控制什么 |
| --- | --- | --- |
| `<joint><origin>` | 父连杆坐标系 | 子关节坐标系的位置和方向 |
| `<inertial><origin>` | 当前 link 坐标系 | 质心和惯量坐标系 |
| `<visual><origin>` | 当前 link 坐标系 | 屏幕上看到的网格位置 |
| `<collision><origin>` | 当前 link 坐标系 | 物理碰撞外形的位置 |
| Gazebo `<sensor><pose>` | 所属 reference link | 传感器安装位姿 |

因此，调好画面中的网格并不代表质心或碰撞体也正确。改动前要先确认正在编辑哪一种
坐标变换。

## 3. 怎样读一个 `<link>`

一个 `link` 通常包含 `inertial`、`visual` 和 `collision` 三部分：

```xml
<link name="Rfoot">
  <inertial>...</inertial>
  <visual>...</visual>
  <collision>...</collision>
</link>
```

### 3.1 `<inertial>`：质量和惯量

通俗理解：它告诉物理引擎这块零件有多重，以及重量集中在哪里。两个外形相同的物体，
如果一个质量集中在中心、另一个质量集中在边缘，转动起来会完全不同。

技术细节：

```xml
<inertial>
  <origin xyz="..." rpy="..."/>
  <mass value="..."/>
  <inertia ixx="..." ixy="..." ixz="..."
           iyy="..." iyz="..." izz="..."/>
</inertial>
```

- `mass` 是质量 [kg]。
- `origin` 通常放在质心处。
- 六个 `inertia` 数值组成对称的 3×3 惯量矩阵 [kg·m²]。
- 惯量必须与质量、几何尺寸和惯量坐标系一致；不能只凭感觉随意填写。

错误的质量或惯量会使训练策略学到错误的加速、刹车和平衡方式。实体机迁移前，应使用
CAD 数据或实测结果核对，并把测量不确定性纳入域随机化。

### 3.2 `<visual>`：只负责“看起来像什么”

通俗理解：`visual` 是渲染器画给人看的外壳。颜色或视觉网格出错可能很难看，但它本身
不一定改变碰撞结果。

技术细节：`geometry/mesh` 引用 STL 网格，`material/color rgba` 定义红、绿、蓝和透明度。
本文件的视觉网格位于 `../meshes/`。

### 3.3 `<collision>`：物理引擎实际拿来碰撞的外形

通俗理解：`collision` 是机器人在物理世界中的“硬壳”。画面上两块零件没碰到，但碰撞
外形重叠，仿真仍会产生很大的接触力。

技术细节：本模型的 `visual` 和 `collision` 当前引用相同 STL。Isaac Lab 导入时使用
`Convex Decomposition`（凸分解）生成更稳定的碰撞近似，而不是直接让复杂凹形三角网格
参与动态碰撞。环境同时启用了 self-collision（自碰撞），左右腿或腿与机架互相穿入时
会产生接触响应。

### 3.4 `<surface>`：Gazebo 接触参数

文件中的 `<friction><ode><mu>...`、`<contact><ode>...` 等属于 Gazebo/ODE 扩展：

- 普通连杆的 `mu` 和 `mu2` 当前为 `1.0`；
- 两只脚的 `mu` 和 `mu2` 当前为 `50.0`；
- `kp`、`kd`、`soft_cfm`、`soft_erp` 调整 Gazebo 接触的刚度与数值稳定性。

这些值不是电机 PD 参数，也不能假设所有仿真器都会用完全相同的方式解释它们。Isaac
Lab 中的材料摩擦和域随机化应在对应的环境配置、事件项中继续核对。

## 4. 怎样读一个 `<joint>`

以一个旋转关节为例：

```xml
<joint name="Rthigh_to_Rleg" type="revolute">
  <origin xyz="..." rpy="..."/>
  <parent link="Rthigh"/>
  <child link="Rleg"/>
  <axis xyz="0 0 1"/>
  <limit lower="-2.25" upper="0.05" effort="60" velocity="3.14"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

### 4.1 `type`

- `fixed`：父子连杆固定在一起，没有可控制自由度。本模型的
  `base_to_top_mount` 属于这种类型。
- `revolute`：绕一根轴旋转，并有上下限。本模型的 8 个主动关节都属于这种类型。

### 4.2 `parent` 和 `child`

通俗理解：父连杆是铰链的上游，子连杆挂在铰链下游。父子关系决定机器人树的方向，
不等于人体中的长幼或左右。

技术细节：关节位姿先相对 `parent` 定义；关节运动再带动整个 `child` 子树。交换父子
会改变坐标关系、关节符号和动力学，不能仅通过改两行名称实现“方向取反”。

### 4.3 `axis`

通俗理解：`axis` 是铰链轴的朝向。右手拇指指向轴的正方向时，其余四指弯曲方向就是
正角度方向，这叫右手定则。

技术细节：`axis` 在关节自己的局部坐标系中表达，而不是直接在世界坐标系中表达。
本模型 8 个主动关节都写成 `0 0 1`，但各关节的 `origin rpy` 不同，所以它们在世界中
的实际轴向并不相同。不能只看到相同的 `axis` 就认定所有关节朝同一方向转。

### 4.4 `limit`

- `lower`、`upper`：允许的最小和最大关节角 [rad]；
- `effort`：URDF 中的最大关节力矩 [N·m]；
- `velocity`：最大关节速度 [rad/s]。

这些是仿真模型约束，不自动等于实体机械的认证安全范围。实体机还需要独立、保守的
软件软限位和驱动器/硬件限位。

### 4.5 `dynamics`

`damping="0.1"` 表示 URDF 关节自身的被动粘性阻尼，`friction="0.0"` 表示被动关节
摩擦参数。它们描述机械关节，不是控制器。

环境配置中的隐式 PD 执行器另设 `stiffness=80.0`、`damping=4.0`。其中 `4.0` 是控制
目标跟踪的微分增益，而 URDF 的 `0.1` 是机械模型阻尼；两个字段名字都叫 damping，
但作用层级完全不同，不能互相替代。

## 5. 当前 8 个主动关节

策略、环境和数据文件使用下面的标准顺序，即先右腿，再左腿。这个顺序定义在
[robot_contract.py](../../../tasks/robot_contract.py) 的 `ACTUATED_JOINT_NAMES` 中。

| 策略下标 | 关节名 | 父 link → 子 link | 功能 | URDF 范围 [rad] |
| ---: | --- | --- | --- | --- |
| 0 | `base_to_Rturn` | `base_link` → `R_turn` | 右侧转向 | `[-1.5708, 1.5708]` |
| 1 | `Rturn_to_Rthigh` | `R_turn` → `Rthigh` | 右髋 | `[-1.5708, 1.5708]` |
| 2 | `Rthigh_to_Rleg` | `Rthigh` → `Rleg` | 右膝 | `[-2.25, 0.05]` |
| 3 | `Rleg_to_Rfoot` | `Rleg` → `Rfoot` | 右踝 | `[-1.5708, 1.5708]` |
| 4 | `base_to_Lturn` | `base_link` → `Lturn` | 左侧转向 | `[-1.5708, 1.5708]` |
| 5 | `Lturn_to_Lthigh` | `Lturn` → `Lthigh` | 左髋 | `[-1.5708, 1.5708]` |
| 6 | `Lthigh_to_Lleg` | `Lthigh` → `Lleg` | 左膝 | `[-0.05, 2.25]` |
| 7 | `Lleg_to_Lfoot` | `Lleg` → `Lfoot` | 左踝 | `[-1.5708, 1.5708]` |

8 个主动关节的 `effort` 都是 `60 N·m`，`velocity` 都是 `3.14 rad/s`，URDF 被动
`damping` 都是 `0.1`，被动 `friction` 都是 `0.0`。

### 5.1 为什么左右膝的正负号相反

模型左右结构镜像。当前右膝原始关节角 `q < 0` 表示人体屈膝，而左膝 `q > 0` 表示
人体屈膝。公共契约用下面的符号把两侧统一为“人体屈曲角为正”：

```python
KNEE_ARTICULATION_SIGNS = {
    "Rthigh_to_Rleg": -1.0,
    "Lthigh_to_Lleg": 1.0,
}
anatomical_flexion = KNEE_ARTICULATION_SIGNS[joint_name] * urdf_joint_angle
```

因此，修改膝关节轴、父子关系或正方向后，必须同步修改这个符号映射、人体膝限位、
数据集 `joint_sign` 和实体电机方向标定。只改 URDF 会让奖励函数把屈膝误判成过伸。

## 6. 策略顺序不等于电机编号顺序

强化学习策略按“右腿 4 个、左腿 4 个”排列，55AA 通讯帧则按实体电机编号 1 到 8
排列。协议编码器会通过关节名进行重排。

| 电机槽位 | 实体关节 | URDF 关节名 |
| ---: | --- | --- |
| 1 | 左踝 | `Lleg_to_Lfoot` |
| 2 | 右踝 | `Rleg_to_Rfoot` |
| 3 | 左膝 | `Lthigh_to_Lleg` |
| 4 | 右膝 | `Rthigh_to_Rleg` |
| 5 | 左髋 | `Lturn_to_Lthigh` |
| 6 | 右髋 | `Rturn_to_Rthigh` |
| 7 | 左转向 | `base_to_Lturn` |
| 8 | 右转向 | `base_to_Rturn` |

绝对不能把策略数组下标直接当成电机编号，否则多个关节会收到属于其他关节的命令。
具体协议见 [communication_protocol.md](../../../../../communication_protocol.md) 和
[`protocol_55aa.py`](../../../tasks/protocol_55aa.py)。

## 7. Gazebo 传感器扩展

URDF 标准主要描述机械结构；文件后半部分的 `<gazebo reference="...">` 是 Gazebo
扩展。`reference` 指明设置属于哪个 link。

### 7.1 IMU

IMU（Inertial Measurement Unit，惯性测量单元）测量角速度和线加速度。当前定义 6 个：

| 所属 link | 名称 | 话题 | 更新率 |
| --- | --- | --- | ---: |
| `Rthigh` | `imu_rthigh` | `/my_exoskeleton/imu/rthigh` | 200 Hz |
| `Rleg` | `imu_rleg` | `/my_exoskeleton/imu/rleg` | 200 Hz |
| `Rfoot` | `imu_rfoot` | `/my_exoskeleton/imu/rfoot` | 200 Hz |
| `Lthigh` | `imu_lthigh` | `/my_exoskeleton/imu/lthigh` | 200 Hz |
| `Lleg` | `imu_lleg` | `/my_exoskeleton/imu/lleg` | 200 Hz |
| `Lfoot` | `imu_lfoot` | `/my_exoskeleton/imu/lfoot` | 200 Hz |

`<pose>` 是传感器相对所属 link 的安装位姿。移动它等于移动虚拟传感器，读取结果的
坐标含义也会改变，因此还要同步核对传感器映射和实体 IMU 安装标定。

### 7.2 足底接触传感器

左右脚各定义 1 个接触传感器，更新率为 20 Hz：

- `right_foot_contact_sensor` 发布 `/right_foot_contact`；
- `left_foot_contact_sensor` 发布 `/left_foot_contact`。

每只脚设置 `max_contacts=10`。文件注释说明接触点后续会分箱成 `10×6` 压力网格；
URDF 本身只产生接触信息，真正的分箱和观测拼接逻辑仍在传感器/任务代码中。

### 7.3 `selfCollision`

每个 link 的 Gazebo 块当前都设置了 `<selfCollision>true</selfCollision>`，允许同一台
机器人内部零件发生碰撞。Isaac Lab 环境也单独启用了自碰撞；两者属于不同仿真路径，
修改其中一处不能默认另一处自动同步。

## 8. `ros2_control` 是什么

通俗理解：`ros2_control` 像一张“控制器可以向哪些关节写什么、又能读回什么”的接口
清单。它不等于实体机驱动程序，也不会自动知道 CAN、EtherCAT 或串口协议。

技术细节：当前 8 个主动关节都声明了：

```xml
<command_interface name="position"/>
<state_interface name="position"/>
<state_interface name="velocity"/>
```

也就是位置命令接口，以及位置、速度状态反馈接口。若改成速度或力矩控制，需要同时
修改控制器、硬件接口、策略动作解释和安全限制，不能只改这个 XML 标签。

文件末尾 Gazebo 插件的 `controllers.yaml` 路径目前硬编码为：

```text
/home/black_paper/my_exoskeleton/install/my_exoskeleton/share/my_exoskeleton/config/controllers.yaml
```

这不是当前项目的可移植路径。在另一台电脑使用 Gazebo/ROS 2 前，必须改为实际安装
位置或使用包共享目录解析；Isaac Lab 的训练入口不依赖这条 Gazebo 控制器路径。

## 9. URDF 进入 Isaac Lab 后发生什么

[环境配置](../../../envs/my_exoskeleton_env_cfg.py) 通过 `UrdfFileCfg` 导入本文件。当前
关键设置包括：

- `fix_base=False`：根节点不是固定在世界上；
- `merge_fixed_joints=True`：导入时允许合并固定关节；
- `force_usd_conversion=True`：依据当前 URDF 重新生成 USD 缓存；
- `collision_type="Convex Decomposition"`：复杂 STL 被凸分解后参与碰撞；
- `self_collision=True` 且 articulation 也启用自碰撞；
- `soft_joint_pos_limit_factor=1.0`：不再围绕 URDF 限位中点额外收缩范围；
- 隐式 PD 为 `stiffness=80.0`、`damping=4.0`；
- actuator 的力矩上限为 `60 N·m`、速度上限为 `3.14 rad/s`。

环境中的传感器对象、奖励、终止条件和动作映射由 Python 配置建立。Gazebo 扩展标签并
不等于 Isaac Lab 场景中同名传感器已经自动正确连接；要以环境配置和运行时检查为准。

## 10. 修改一处时还要同步检查什么

URDF 是多个模块共同依赖的“接口”。下面的联动表比单纯让模型成功加载更重要。

| 你修改的内容 | 必须同步检查 |
| --- | --- |
| joint/link 名称 | `robot_contract.py`、环境中的关节/刚体路径、奖励和终止项、IMU 映射、数据集名称与符号映射、55AA 协议配置 |
| 膝关节轴、父子关系或正方向 | `KNEE_ARTICULATION_SIGNS`、人体膝限位、模仿数据 `joint_sign`、协议每电机的 `direction` 与 `zero_offset` |
| 位置、速度或力矩限位 | 环境 actuator 限制、动作尺度、奖励边界、训练数据校验、实体机软件软限位和驱动器限位 |
| 质量或惯量 | 域随机化的标称值/范围、初始站姿稳定性、单环境自由响应和接触测试 |
| 碰撞网格或碰撞位姿 | 重新转换 USD，检查自碰撞、脚底接地、零姿态是否穿模，以及 PhysX 是否出现异常冲量 |
| 摩擦或接触参数 | 确认目标仿真器实际读取哪个参数，重新检查地面材料和域随机化范围 |
| 传感器名称、所属 link 或 pose | 环境 scene 的传感器路径、IMU mapping、数据坐标变换和实体传感器标定 |
| `ros2_control` 接口 | ROS 2 controller YAML、hardware interface、实体驱动模式、反馈和安全状态机 |

尤其注意：协议的 `direction` 和 `zero_offset` 是实体电机标定，URDF 关节正方向只是仿真
坐标约定。两者需要经过实机低速、卸载或悬空测试确认，不能仅靠名称猜测。

## 11. 推荐的安全修改流程

1. 先复制并记录当前 URDF 参数，以及本次修改的机械依据。
2. 每次只改一类内容，例如只改一个关节的轴或只改一块连杆的质量。
3. 搜索被修改的 joint/link 名称，按上一节联动表同步更新项目代码。
4. 重新导入 URDF；不要直接编辑 `my_exoskeleton/*.usd(a)`。
5. 只启动 1 个环境，不加载策略，检查零姿态、连杆树、关节轴和限位。
6. 逐个关节发送很小的正负目标，确认旋转方向和左右镜像关系。
7. 检查碰撞可视化、自碰撞、两脚是否平整接地，以及静止时是否出现异常抖动。
8. 检查 6 个 IMU 与 2 个足底传感器的运行时路径和数据方向。
9. 再进行短时间回放，观察跟踪误差、力矩、速度、接触力和终止原因。
10. 只有仿真检查通过后，才更新实体机标定；实体测试从低速度、低力矩和急停就绪开始。

## 12. 常见误区

### “STL 看起来正确，模型就正确吗？”

不一定。视觉网格、碰撞网格、质心、惯量和关节坐标是五类不同信息，必须分别检查。

### “把 `<axis>` 从 `0 0 1` 改成 `0 0 -1` 就结束了吗？”

不行。关节角和速度符号会反转，已有 checkpoint、奖励、参考数据和电机方向映射都可能
随之失效。旧 checkpoint 学到的是旧坐标约定，通常不能直接当作同一任务继续使用。

### “URDF 的 `effort=60` 会自动保护实体电机吗？”

不会。它首先是模型/导入器使用的限制。实体机必须在工控机、驱动器和必要的硬件层
分别设置独立限位、急停和看门狗。

### “URDF 的摩擦值就是训练时最终摩擦值吗？”

不一定。仿真器导入规则、环境材料配置和域随机化都可能覆盖或缩放它。应在运行时读取
实际物理属性，并用接触实验验证。

### “为什么修改后还看到旧模型？”

先确认改的是 [my_exoskeleton.urdf](my_exoskeleton.urdf)，不是生成的 USD；再确认环境
仍启用 `force_usd_conversion=True`，并检查启动日志中的实际 `asset_path`。最后用单环境
查看关节、碰撞和传感器路径，而不是只凭渲染颜色判断。

