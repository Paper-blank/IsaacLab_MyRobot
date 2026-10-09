"""多模态外骨骼模仿奖励与紧凑传感器观测。

本模块由 ObservationManager 和 RewardManager 在每个控制步调用。输入一部分来自当前
仿真场景（关节、IMU、RGB、事件），另一部分来自环境预先放入 ``_imitation_targets``
的实测目标和 ``valid`` 掩码；输出是固定 shape 的策略观测或每环境一个奖励值。

所有相似度均为 ``exp(-MSE/sigma^2)`` 形式的 ``[0,1]`` 正分数：越接近实测目标越接近
1。奖励管理器给这些相似度正权重；安全、能耗等成本仍在 locomotion 配置中使用负
权重。无效通道不进入分子和分母，整项无有效数据时得 0，绝不会把“缺失”当作完美匹配。

核心计算只使用 Torch 张量，底部的小型适配器才读取 Isaac Lab 环境。固定契约为：
IMU ``[N,6,9]``（姿态 rad、角速度 rad/s、加速度 m/s^2），RGB 紧凑特征
``[N,12]``（左右相机 RGB 均值与标准差），Event ``[N,32]``（左右/正负四平面各
2x4 空间池化后 ``log1p``）。这种压缩避免把大图像直接拼入 PPO 观测。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_assets.tasks.robot_contract import ACTUATED_JOINT_NAMES

IMU_FEATURES = 9  # roll/pitch/yaw + 陀螺仪 xyz + 加速度计 xyz
IMU_COUNT = 6
RGB_FEATURES = 12  # 两相机 x（RGB 三通道均值 + RGB 三通道标准差）
EVENT_FEATURES = 32  # 两相机 x 两极性 x 八个空间池化格


def _as_tensor(value: Any) -> torch.Tensor:
    """把 Isaac Lab 代理数组和普通张量统一为 Torch 张量。

    转换不复制数据，也不改变 dtype、device、shape 或物理单位。
    """

    return value.torch if hasattr(value, "torch") else value


def masked_exponential_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    sigma: float,
    valid: torch.Tensor | None = None,
    dim: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """计算可选有效掩码下的 ``exp(-MSE/sigma^2)``。

    Args:
        prediction: 仿真值，第一维通常是并行环境 ``N``。
        target: 目标值；会与 ``prediction`` 按 Torch 广播规则对齐。
        sigma: 正的误差尺度，与输入使用相同物理单位。
        valid: 可广播到误差 shape 的布尔/0-1 掩码。无效元素不计误差也不计分母。
        dim: 求均方误差的维度；``None`` 且无掩码时仅压缩最后一维，有掩码时压缩
            除环境维之外的全部维度。

    Returns:
        压缩指定维度后的 ``[0,1]`` 张量。某环境分母为零时该位置严格返回 0。

    Raises:
        ValueError: ``sigma`` 不为正数。
        RuntimeError: prediction、target 或 valid 的 shape 无法广播。
    """

    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    prediction, target = torch.broadcast_tensors(prediction, target)
    error = torch.square(prediction - target)
    if valid is None:
        if dim is None:
            mse = error.mean(dim=-1)
        else:
            mse = error.mean(dim=dim)
        return torch.exp(-mse / (sigma * sigma))
    mask = valid.to(dtype=error.dtype, device=error.device)
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    mask = torch.broadcast_to(mask, error.shape)
    if dim is None:
        reduce_dims = tuple(range(1, error.ndim))
    elif isinstance(dim, int):
        reduce_dims = (dim if dim >= 0 else error.ndim + dim,)
    else:
        reduce_dims = tuple(d if d >= 0 else error.ndim + d for d in dim)
    numerator = (error * mask).sum(dim=reduce_dims)
    denominator = mask.sum(dim=reduce_dims)
    score = torch.exp(-numerator / denominator.clamp_min(1.0) / (sigma * sigma))
    return torch.where(denominator > 0.0, score, torch.zeros_like(score))


def imu_orientation_similarity(
    simulation: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    sigma: float = 0.35,
) -> torch.Tensor:
    """比较六路 IMU 的前三个欧拉姿态分量 [rad]。"""

    return masked_exponential_similarity(simulation[..., :3], target[..., :3], sigma=sigma, valid=valid, dim=(-1, -2))


def gyro_similarity(
    simulation: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    sigma: float = 1.0,
) -> torch.Tensor:
    """比较六路 IMU 的角速度通道 [rad/s]。"""

    return masked_exponential_similarity(simulation[..., 3:6], target[..., 3:6], sigma=sigma, valid=valid, dim=(-1, -2))


def accelerometer_similarity(
    simulation: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    sigma: float = 2.0,
) -> torch.Tensor:
    """比较六路 IMU 的线加速度通道 [m/s^2]。"""

    return masked_exponential_similarity(simulation[..., 6:9], target[..., 6:9], sigma=sigma, valid=valid, dim=(-1, -2))


def joint_similarity(
    simulation: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    sigma: float = 0.35,
) -> torch.Tensor:
    """按 canonical 顺序比较八个关节角 [rad]。"""

    return masked_exponential_similarity(simulation, target, sigma=sigma, valid=valid, dim=-1)


def velocity_similarity(
    simulation: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    sigma: float = 0.5,
) -> torch.Tensor:
    """比较机身线速度与目标速度向量 [m/s]。"""

    return masked_exponential_similarity(simulation, target, sigma=sigma, valid=valid, dim=-1)


def feature_similarity(
    simulation: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    sigma: float = 0.25,
) -> torch.Tensor:
    """比较无物理单位的紧凑 RGB 或事件特征。"""

    return masked_exponential_similarity(simulation, target, sigma=sigma, valid=valid, dim=-1)


def weighted_imitation_reward(
    components: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    valid_components: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """按有效通道逐环境归一化多个正相似度。

    权重为零或负数的项被关闭。分母按环境从“权重大于零且当前有效”的项计算，因此
    某个 trial 缺少 motion/event 不会产生 NaN，也不会稀释仍然有效的 IMU 奖励。

    Args:
        components: 名称到相似度 ``[N]`` 张量的映射。
        weights: 名称到非负权重的映射；缺失名称等价于权重 0。
        valid_components: 可选名称到 ``bool[N]`` 有效掩码的映射。

    Returns:
        ``float[N]`` 归一化奖励；某环境没有任何有效正权重项时为 0。

    Raises:
        ValueError: ``components`` 为空。
    """

    if not components:
        raise ValueError("at least one imitation component is required")
    first = next(iter(components.values()))
    numerator = torch.zeros_like(first)
    denominator = torch.zeros_like(first)
    for name, value in components.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0.0:
            continue
        active = torch.ones_like(value)
        if valid_components is not None and name in valid_components:
            active = valid_components[name].to(dtype=value.dtype, device=value.device)
        numerator = numerator + weight * value * active
        denominator = denominator + weight * active
    return torch.where(denominator > 0.0, numerator / denominator.clamp_min(1.0e-8), torch.zeros_like(numerator))


def _target_cache(env) -> dict[str, torch.Tensor]:
    """读取环境的目标缓存；未初始化或类型错误时返回空字典。"""

    cache = getattr(env, "_imitation_targets", None)
    if not isinstance(cache, dict):
        return {}
    return cache


def _target(env, name: str, shape: tuple[int, ...], *, fill: float = 0.0) -> torch.Tensor:
    """取得浮点目标，并在缺失时构造固定 shape 默认值。

    Args:
        env: 模仿环境，提供 ``num_envs``、``device`` 和目标缓存。
        name: 缓存键。
        shape: 不含环境维的单环境 shape。
        fill: 缺失时的填充值。

    Returns:
        位于环境 device 的 ``float32[num_envs,*shape]`` 张量。
    """

    value = _target_cache(env).get(name)
    if value is None:
        return torch.full((env.num_envs, *shape), fill, device=env.device, dtype=torch.float32)
    return value.to(device=env.device, dtype=torch.float32)


def _target_valid(env, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    """取得目标有效掩码；缺失时返回全 ``False``，而不是猜测数据有效。"""

    value = _target_cache(env).get(name)
    if value is None:
        return torch.zeros((env.num_envs, *shape), device=env.device, dtype=torch.bool)
    return value.to(device=env.device, dtype=torch.bool)


def _sensor_enabled(env: Any, name: str) -> bool:
    """读取可选传感器开关；旧配置没有该字段时为兼容而默认启用。"""

    cfg = getattr(env, "cfg", None)
    return bool(getattr(cfg, f"use_{name}", True))


def _zero_reward(env: Any) -> torch.Tensor:
    """为禁用通道返回 device/dtype 正确的 ``float32[N]`` 零奖励。"""

    return torch.zeros((int(env.num_envs),), device=env.device, dtype=torch.float32)


def _scene_entity(env, name: str) -> Any | None:
    """安全取得场景实体，不假设场景容器实现字典 ``get`` 方法。"""

    scene = getattr(env, "scene", None)
    if scene is None:
        return None
    try:
        if name not in scene.keys():
            return None
        return scene[name]
    except (AttributeError, KeyError, TypeError):
        return None


def _robot_joint_tensor(env, field: str) -> torch.Tensor:
    """按 canonical 八关节顺序抽取机器人状态字段。

    第一次调用会从 articulation 的 ``joint_names`` 建立索引并缓存到环境，之后每步
    直接 ``index_select``，避免重复字符串查找。

    Returns:
        shape ``[N,8]`` 的关节状态；单位由 ``field`` 决定，例如 ``joint_pos`` 为 rad。

    Raises:
        ValueError: 机器人缺少 canonical 关节。
    """

    robot = env.scene["robot"]
    value = _as_tensor(getattr(robot.data, field))
    names = tuple(getattr(robot, "joint_names", ()))
    ids = getattr(env, "_imitation_joint_ids", None)
    if ids is None:
        missing = [name for name in ACTUATED_JOINT_NAMES if name not in names]
        if missing:
            raise ValueError(f"robot is missing canonical joints: {missing}")
        ids = torch.tensor([names.index(name) for name in ACTUATED_JOINT_NAMES], device=value.device)
        setattr(env, "_imitation_joint_ids", ids)
    return value.index_select(-1, ids)


def _sensor_tensor(sensor: Any, field: str, env_count: int, device: torch.device | str) -> torch.Tensor:
    """把可选传感器三轴字段规范为 ``float32[N,3]``。

    传感器或字段缺失时返回同 device 的零张量；“是否有效”由单独掩码表示，零值本身
    不能用来判断缺失。
    """

    if sensor is None:
        return torch.zeros((env_count, 3), device=device)
    data = sensor.data
    value = getattr(data, field, None)
    if value is None:
        return torch.zeros((env_count, 3), device=device)
    value = _as_tensor(value).to(device=device, dtype=torch.float32)
    if value.ndim == 1:
        value = value.reshape(env_count, -1)
    return value.reshape(env_count, -1)[..., :3]


def _imu_sensor_names(env) -> tuple[str, ...]:
    """返回环境配置的六路 IMU 名称；未配置时使用记录器标准名称。"""

    return tuple(
        getattr(env, "imitation_imu_sensor_names", ("imu_01", "imu_02", "imu_03", "imu_04", "imu_05", "imu_06"))
    )


def _body_orientation(env, body_name: str) -> torch.Tensor:
    """返回指定刚体的 world-frame XYZ 欧拉角 ``[N,3]`` [rad]。

    若名称无法在 link 列表中解析，会回退到 articulation 根四元数。
    """

    robot = env.scene["robot"]
    quat = _as_tensor(robot.data.root_quat_w)
    names = tuple(getattr(robot, "body_names", ()))
    if body_name in names and hasattr(robot.data, "body_link_quat_w"):
        quat = _as_tensor(robot.data.body_link_quat_w)[:, names.index(body_name)]
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    return torch.stack((roll, pitch, yaw), dim=-1)


def imu_observation(env, *, inspect: bool = False) -> torch.Tensor:
    """构造 PPO 使用的六路 IMU 扁平观测 ``[N,54]``。

    Args:
        env: 含六个 IMU 场景实体和已解析肢段名称的模仿环境。
        inspect: ObservationManager 探测 shape 时设为 ``True``，此时不读传感器。

    Returns:
        ``float[N,54]``，每路九维顺序为欧拉角 [rad]、角速度 [rad/s]、线加速度
        [m/s^2]。同时缓存原始 ``[N,6,9]`` 和 ``bool[N,6]`` 供诊断/奖励使用。
    """

    if inspect or not _sensor_enabled(env, "imu"):
        env._latest_imu = torch.zeros((env.num_envs, IMU_COUNT, IMU_FEATURES), device=env.device)
        env._latest_imu_valid = torch.zeros((env.num_envs, IMU_COUNT), dtype=torch.bool, device=env.device)
        return torch.zeros((env.num_envs, IMU_COUNT * IMU_FEATURES), device=env.device)
    body_names = tuple(getattr(env, "_resolved_imu_links", ("Lfoot", "Rfoot", "Lleg", "Rleg", "Lthigh", "Rthigh")))
    rows = []
    for sensor_name, body_name in zip(_imu_sensor_names(env), body_names, strict=True):
        orientation = _body_orientation(env, body_name)
        sensor = _scene_entity(env, sensor_name)
        gyro = _sensor_tensor(sensor, "ang_vel_b", env.num_envs, env.device)
        accel = _sensor_tensor(sensor, "lin_acc_b", env.num_envs, env.device)
        rows.append(torch.cat((orientation, gyro, accel), dim=-1))
    stacked = torch.stack(rows, dim=1)
    # 保留原生 ``[N, 6, 9]`` 张量供 play/viser 诊断；交给 manager 的 term 仍展平，便于
    # rl_games 拼接。
    env._latest_imu = stacked
    env._latest_imu_valid = torch.stack(
        [
            torch.full(
                (env.num_envs,),
                _scene_entity(env, sensor_name) is not None,
                dtype=torch.bool,
                device=env.device,
            )
            for sensor_name in _imu_sensor_names(env)
        ],
        dim=1,
    )
    return stacked.reshape(env.num_envs, -1)


def camera_rgb_observation(env, *, inspect: bool = False) -> torch.Tensor:
    """把左右 RGB 图像压缩为每环境 12 维矩特征。

    输入相机缓冲应为 ``[N,H,W,C]``，uint8 范围会除以 255 归一化。输出顺序是左/右
    相机三通道均值，再接左/右三通道总体标准差，共 ``[N,12]``。最多四个环境或显式
    开启 ``store_raw_camera_frames`` 时还会保留原图供回放诊断。

    Args:
        env: 含左右 RGB 相机的模仿环境。
        inspect: 为 ``True`` 或 RGB 被禁用时只建立零 shape，不访问 RTX 缓冲。

    Returns:
        归一化紧凑特征 ``float[N,12]``。任一路相机不可用时全环境有效掩码为
        ``False``，shape 仍保持不变。
    """

    if inspect or not _sensor_enabled(env, "rgb"):
        env._latest_rgb_valid = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        env._latest_rgb_feature = torch.zeros((env.num_envs, RGB_FEATURES), device=env.device)
        env._latest_camera_rgb = {}
        return torch.zeros((env.num_envs, RGB_FEATURES), device=env.device)
    means = []
    stds = []
    raw = {}
    available = torch.ones((env.num_envs,), dtype=torch.bool, device=env.device)
    for name in ("left_rgb_camera", "right_rgb_camera"):
        camera = _scene_entity(env, name)
        image = None
        if camera is not None:
            try:
                image = _as_tensor(camera.data.output["rgb"]).to(device=env.device, dtype=torch.float32)
            except (KeyError, AttributeError, RuntimeError):
                image = None
        # ObservationManager 首次探测 shape 时（第一次渲染前）RTX 相机缓冲可能为空；应把
        # 此状态视为帧不可用，而不是对无效 CUDA view 做归约运算。
        if (
            image is None
            or image.ndim != 4
            or image.shape[0] != env.num_envs
            or image.shape[1] < 1
            or image.shape[2] < 1
            or image.shape[3] < 3
        ):
            available.zero_()
            image = torch.zeros((env.num_envs, 256, 256, 3), device=env.device)
        elif image.shape[-1] > 3:
            image = image[..., :3]
        # Isaac Lab 渲染器返回 uint8 范围的 RGB；数据集加载器和特征奖励使用 [0, 1]
        # 浮点值，因此这里只归一化一次。
        try:
            if image.numel() and float(image.detach().amax()) > 1.5:
                image = image / 255.0
            means.append(image.mean(dim=(1, 2)))
            stds.append(image.std(dim=(1, 2), unbiased=False))
        except (RuntimeError, TypeError, ValueError):
            # headless stage reset 时渲染器可能使缓冲失效；这里保持 PPO 观测 shape 稳定。
            available.zero_()
            image = torch.zeros((env.num_envs, 256, 256, 3), device=env.device)
            means.append(image.mean(dim=(1, 2)))
            stds.append(image.std(dim=(1, 2), unbiased=False))
        if bool(getattr(env.cfg, "store_raw_camera_frames", False)) or env.num_envs <= 4:
            raw[name] = image
    feature = torch.cat((*means, *stds), dim=-1)
    env._latest_rgb_feature = feature
    env._latest_camera_rgb = raw
    env._latest_rgb_valid = available
    return feature


def event_camera_observation(env, *, inspect: bool = False) -> torch.Tensor:
    """取得 EVIS 的固定 32 维事件特征，缺失时返回无效零值。

    环境的 ``update_event_observation`` 负责把左右相机正/负极性四平面从 ``[N,4,H,W]``
    池化为 ``[N,4,2,4]``，再经 ``log1p`` 得到 ``[N,32]``。本函数只调用该适配器并
    保存有效掩码，不使用 RGB 帧差作为后备方案。
    """

    if inspect or not _sensor_enabled(env, "event"):
        env._latest_event_hist = torch.zeros((env.num_envs, EVENT_FEATURES), device=env.device)
        env._latest_event_valid = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        return torch.zeros((env.num_envs, EVENT_FEATURES), device=env.device)
    update = getattr(env, "update_event_observation", None)
    if callable(update):
        value, valid = update()
        env._latest_event_valid = valid
        return value
    env._latest_event_valid = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    return torch.zeros((env.num_envs, EVENT_FEATURES), device=env.device)


def imitation_imu_reward(env, orientation_sigma: float = 0.35, gyro_sigma: float = 1.0, accel_sigma: float = 2.0):
    """返回每环境 IMU 姿态、角速度和线加速度的加权模仿奖励 ``[N]``。"""

    if not _sensor_enabled(env, "imu"):
        return _zero_reward(env)
    simulation = imu_observation(env)
    simulation = simulation.reshape(env.num_envs, IMU_COUNT, IMU_FEATURES)
    target = _target(env, "target_imu", (IMU_COUNT, IMU_FEATURES))
    valid = _target_valid(env, "imu_valid", (IMU_COUNT,))
    pose = imu_orientation_similarity(simulation, target, valid, orientation_sigma)
    gyro = gyro_similarity(simulation, target, valid, gyro_sigma)
    accel = accelerometer_similarity(simulation, target, valid, accel_sigma)
    return (pose + 0.5 * gyro + 0.25 * accel) / 1.75


def imitation_gyro_reward(env, sigma: float = 1.0):
    """返回六路陀螺仪角速度模仿奖励 ``[N]``；禁用 IMU 时返回零。"""

    if not _sensor_enabled(env, "imu"):
        return _zero_reward(env)
    simulation = imu_observation(env).reshape(env.num_envs, IMU_COUNT, IMU_FEATURES)
    target = _target(env, "target_imu", (IMU_COUNT, IMU_FEATURES))
    valid = _target_valid(env, "imu_valid", (IMU_COUNT,))
    return gyro_similarity(simulation, target, valid, sigma)


def imitation_joint_reward(env, sigma: float = 0.35):
    """比较仿真与参考的 8 关节角 [rad]，返回指数相似度奖励 ``[N]``。"""

    simulation = _robot_joint_tensor(env, "joint_pos")
    target = _target(env, "target_motion", (len(ACTUATED_JOINT_NAMES),))
    valid = _target_valid(env, "motion_valid", ())
    return joint_similarity(simulation, target, valid, sigma)


def imitation_velocity_reward(env, sigma: float = 0.5):
    """比较机体水平速度与二维目标速度 [m/s]，返回模仿奖励 ``[N]``。"""

    simulation = _as_tensor(env.scene["robot"].data.root_lin_vel_b)[:, :2]
    target = _target(env, "target_velocity", (2,))
    valid = _target_valid(env, "velocity_valid", ())
    return velocity_similarity(simulation, target, valid, sigma)


def imitation_rgb_reward(env, sigma: float = 0.25):
    """比较 RGB 特征并返回奖励 ``[N]``；传感器禁用或目标缺失时不贡献奖励。"""

    if not _sensor_enabled(env, "rgb"):
        return _zero_reward(env)
    simulation = camera_rgb_observation(env)
    target = _target(env, "target_rgb_feature", (RGB_FEATURES,))
    valid = _target_valid(env, "rgb_valid", ()) & getattr(
        env, "_latest_rgb_valid", torch.ones((env.num_envs,), dtype=torch.bool, device=env.device)
    )
    return feature_similarity(simulation, target, valid, sigma)


def imitation_event_reward(env, sigma: float = 1.0):
    """比较事件相机特征并返回奖励 ``[N]``，同时应用目标与运行时有效掩码。"""

    if not _sensor_enabled(env, "event"):
        return _zero_reward(env)
    simulation = event_camera_observation(env)
    target = _target(env, "target_event_feature", (EVENT_FEATURES,))
    target_valid = _target_valid(env, "event_valid", ())
    runtime_valid = getattr(
        env,
        "_latest_event_valid",
        torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
    )
    valid = target_valid & runtime_valid.to(device=env.device, dtype=torch.bool)
    return feature_similarity(simulation, target, valid, sigma)


def imitation_total_reward(
    env,
    weights: Mapping[str, float] | None = None,
    orientation_sigma: float = 0.35,
    gyro_sigma: float = 1.0,
    joint_sigma: float = 0.35,
    velocity_sigma: float = 0.5,
    rgb_sigma: float = 0.25,
    event_sigma: float = 1.0,
):
    """计算按有效性归一化的六通道多模态奖励。

    六项为 IMU 综合、陀螺仪、关节角、机身速度、RGB 特征和事件特征，单项 shape 均为
    ``[N]``。传感器开关、目标文件有效性和运行时有效性共同决定该项是否进入每个环境
    的加权分母。

    Args:
        env: 当前模仿环境。
        weights: 可选六项权重；省略时使用函数内默认值。
        orientation_sigma: 姿态误差尺度 [rad]。
        gyro_sigma: 角速度误差尺度 [rad/s]。
        joint_sigma: 关节角误差尺度 [rad]。
        velocity_sigma: 线速度误差尺度 [m/s]。
        rgb_sigma: 归一化 RGB 矩特征误差尺度。
        event_sigma: ``log1p`` 事件池化特征误差尺度。

    Returns:
        每个并行环境一个 ``[0,1]`` 奖励的 ``float[N]`` 张量。
    """

    weights = weights or {
        "imu": 1.0,
        "gyro": 0.5,
        "joint": 0.5,
        "velocity": 0.5,
        "rgb": 0.25,
        "event": 0.25,
    }
    components = {
        "imu": imitation_imu_reward(env, orientation_sigma, gyro_sigma),
        "gyro": imitation_gyro_reward(env, gyro_sigma),
        "joint": imitation_joint_reward(env, joint_sigma),
        "velocity": imitation_velocity_reward(env, velocity_sigma),
        "rgb": imitation_rgb_reward(env, rgb_sigma),
        "event": imitation_event_reward(env, event_sigma),
    }
    valid = {
        "imu": _sensor_enabled(env, "imu") & _target_valid(env, "imu_valid", (IMU_COUNT,)).any(dim=-1),
        "gyro": _sensor_enabled(env, "imu") & _target_valid(env, "imu_valid", (IMU_COUNT,)).any(dim=-1),
        "joint": _target_valid(env, "motion_valid", ()).reshape(env.num_envs),
        "velocity": _target_valid(env, "velocity_valid", ()).reshape(env.num_envs),
        "rgb": (
            _sensor_enabled(env, "rgb")
            & _target_valid(env, "rgb_valid", ()).reshape(env.num_envs)
            & getattr(
                env,
                "_latest_rgb_valid",
                torch.ones((env.num_envs,), dtype=torch.bool, device=env.device),
            )
        ),
        "event": (
            _sensor_enabled(env, "event")
            & _target_valid(env, "event_valid", ()).reshape(env.num_envs)
            & getattr(
                env,
                "_latest_event_valid",
                torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
            ).to(device=env.device, dtype=torch.bool)
        ),
    }
    return weighted_imitation_reward(components, weights, valid)


__all__ = [
    "EVENT_FEATURES",
    "IMU_COUNT",
    "IMU_FEATURES",
    "RGB_FEATURES",
    "accelerometer_similarity",
    "camera_rgb_observation",
    "event_camera_observation",
    "feature_similarity",
    "gyro_similarity",
    "imu_observation",
    "imitation_event_reward",
    "imitation_gyro_reward",
    "imitation_imu_reward",
    "imitation_joint_reward",
    "imitation_rgb_reward",
    "imitation_total_reward",
    "imitation_velocity_reward",
    "joint_similarity",
    "masked_exponential_similarity",
    "velocity_similarity",
    "weighted_imitation_reward",
]
