"""
敏感性分析工具模块
"""

# 确保配置模块可以被正确导入
try:
    from ..configs.config import MODEL_CONFIGS, DATASET_CONFIGS
except ImportError:
    try:
        import sys
        from pathlib import Path
        # 添加项目根目录到路径
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from scripts.utils.config import MODEL_CONFIGS, DATASET_CONFIGS
    except ImportError:
        # 最后的备选方案
        sys.path.insert(0, str(Path(__file__).parent.parent / "configs"))
        from config import MODEL_CONFIGS, DATASET_CONFIGS