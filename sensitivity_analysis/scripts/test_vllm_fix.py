#!/usr/bin/env python3
"""
VLLM推理修复测试脚本
"""

import sys
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_config_import():
    """测试配置导入"""
    print("🧪 测试配置导入...")

    try:
        # 测试新路径
        from sensitivity_analysis.configs.config import MODEL_CONFIGS
        print("✅ 新路径导入成功: sensitivity_analysis.configs.config")
        return True
    except ImportError as e:
        print(f"❌ 新路径导入失败: {e}")

    try:
        # 测试旧路径
        from scripts.utils.config import MODEL_CONFIGS
        print("✅ 旧路径导入成功: scripts.utils.config")
        return True
    except ImportError as e:
        print(f"❌ 旧路径导入失败: {e}")

    try:
        # 测试直接导入
        sys.path.insert(0, str(Path(__file__).parent.parent / "configs"))
        from config import MODEL_CONFIGS
        print("✅ 直接导入成功: config")
        return True
    except ImportError as e:
        print(f"❌ 直接导入失败: {e}")

    return False

def test_vllm_config():
    """测试VLLM配置获取"""
    print("\n🧪 测试VLLM配置获取...")

    try:
        # 尝试多种导入方式
        MODEL_CONFIGS = None

        try:
            from sensitivity_analysis.configs.config import MODEL_CONFIGS
        except ImportError:
            try:
                from scripts.utils.config import MODEL_CONFIGS
            except ImportError:
                sys.path.insert(0, str(Path(__file__).parent.parent / "configs"))
                from config import MODEL_CONFIGS

        if MODEL_CONFIGS is not None:
            print(f"✅ 配置加载成功，共{len(MODEL_CONFIGS)}个模型配置")

            # 测试特定模型的配置
            test_model = "/root/autodl-tmp/models/Qwen1.5-7B"
            if test_model in MODEL_CONFIGS:
                template, trust_remote_code = MODEL_CONFIGS[test_model]
                print(f"✅ 找到{test_model}配置: template={template}, trust_remote_code={trust_remote_code}")
            else:
                print(f"⚠️ 未找到{test_model}配置")

            return True
        else:
            print("❌ 配置加载失败")
            return False

    except Exception as e:
        print(f"❌ VLLM配置测试失败: {e}")
        return False

def test_vllm_inference_import():
    """测试VLLM推理脚本导入"""
    print("\n🧪 测试VLLM推理脚本导入...")

    try:
        from scripts.vllm_infer import vllm_infer
        print("✅ VLLM推理脚本导入成功")
        return True
    except ImportError as e:
        print(f"❌ VLLM推理脚本导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始VLLM修复测试...\n")

    # 设置日志级别
    logging.basicConfig(level=logging.INFO)

    # 运行测试
    tests = [
        ("配置导入测试", test_config_import),
        ("VLLM配置测试", test_vllm_config),
        ("VLLM推理脚本导入测试", test_vllm_inference_import),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🔧 {test_name}")
        print('='*50)

        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}异常: {e}")
            results.append((test_name, False))

    # 输出总结
    print(f"\n{'='*50}")
    print("📊 测试结果总结")
    print('='*50)

    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n总计: {passed}/{len(results)} 测试通过")

    if passed == len(results):
        print("🎉 所有测试通过！VLLM修复成功！")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)