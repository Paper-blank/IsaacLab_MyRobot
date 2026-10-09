"""外骨骼环境配置类的统一导出入口。

这里只重新导出配置类，不会创建仿真环境。真正的场景、观测、动作和奖励定义位于两个
my_exoskeleton 环境配置文件中。
"""

from .my_exoskeleton_env_cfg import MyExoskeletonEnvCfg, MyExoskeletonFlatEnvCfg
from .my_exoskeleton_imitation_env_cfg import MyExoskeletonImitationEnvCfg

__all__ = ["MyExoskeletonEnvCfg", "MyExoskeletonFlatEnvCfg", "MyExoskeletonImitationEnvCfg"]
