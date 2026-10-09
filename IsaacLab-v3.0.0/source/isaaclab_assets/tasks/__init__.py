"""外骨骼自定义任务包的公开入口。

基础环境可以直接导入；模仿环境采用延迟导入，避免配置发现阶段形成循环依赖。这里不
包含训练循环，只负责让外部代码用稳定名称找到环境类。
"""

from .my_exoskeleton_flat import MyExoskeletonFlatEnv

__all__ = ["MyExoskeletonFlatEnv", "MyExoskeletonImitationEnv"]


def __getattr__(name: str):
    """配置模块准备完成后再加载模仿环境。

    模仿任务先导入环境配置，而环境配置又会从本包引用共享任务函数。如果在包加载时
    同时立即导入双方，Python 会遇到尚未初始化完成的循环依赖。只有外部真正访问
    MyExoskeletonImitationEnv 时才导入，既打破循环，又保留稳定的公共访问路径。
    """

    if name == "MyExoskeletonImitationEnv":
        from .my_exoskeleton_imitation import MyExoskeletonImitationEnv

        return MyExoskeletonImitationEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
