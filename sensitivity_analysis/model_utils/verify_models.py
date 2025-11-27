#!/usr/bin/env python3
"""
模型验证工具 - 整合所有模型验证功能
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any

# 添加项目路径
current_dir = Path(__file__).parent
sensitivity_analysis_root = current_dir.parent
project_root = sensitivity_analysis_root.parent

# 添加项目根目录到路径
sys.path.insert(0, str(project_root))

# 导入配置 - 优先使用新的路径
try:
    from sensitivity_analysis.configs.config import MODEL_CONFIGS
except ImportError:
    try:
        from scripts.utils.config import MODEL_CONFIGS
    except ImportError:
        # 如果都失败了，尝试直接导入
        sys.path.insert(0, str(sensitivity_analysis_root / "configs"))
        from config import MODEL_CONFIGS

class ModelVerifier:
    """模型验证器"""

    def __init__(self):
        self.logger = self.setup_logging()
        self.model_dir = Path("/root/autodl-tmp/models")

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        if not self.model_dir.exists():
            self.logger.error(f"❌ 模型目录不存在: {self.model_dir}")
            return []

        model_preferences = [
            "Qwen1.5-7B",
            "Meta-Llama-3.1-8B-Instruct",
            "Baichuan2-7B-Chat",
            "Mistral-7B-v0.1",
            "chatglm3-6b"
        ]

        available = []
        for model in model_preferences:
            model_path = self.model_dir / model
            if model_path.is_dir():
                available.append(model)

        return available

    def test_basic_loading(self, model_name: str) -> bool:
        """测试基础模型加载"""
        try:
            model_path = str(self.model_dir / model_name)

            # 简单的加载测试
            import transformers
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto"
            )

            # 简单推理测试
            inputs = tokenizer("Hello", return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=1)

            self.logger.info(f"✅ {model_name}: 基础加载成功")
            return True

        except Exception as e:
            self.logger.error(f"❌ {model_name}: 基础加载失败 - {e}")
            return False

    def test_llamafactory_compatibility(self, model_name: str) -> bool:
        """测试LLaMA-Factory兼容性"""
        try:
            # 创建测试配置
            import tempfile
            import yaml

            model_path = str(self.model_dir / model_name)
            template = MODEL_CONFIGS.get(model_path, ("qwen", True))[0]

            config = {
                "model_name_or_path": model_path,
                "template": template,
                "stage": "sft",
                "do_train": False,
                "dataset": "alpaca_en_demo",
                "max_samples": 1,
                "output_dir": tempfile.mkdtemp()
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                config_file = f.name

            # 运行测试
            cmd = ["llamafactory-cli", "train", config_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # 清理
            os.unlink(config_file)
            os.makedirs(config["output_dir"], exist_ok=True)

            if result.returncode == 0:
                self.logger.info(f"✅ {model_name}: LLaMA-Factory兼容")
                return True
            else:
                self.logger.error(f"❌ {model_name}: LLaMA-Factory不兼容 - {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"❌ {model_name}: LLaMA-Factory测试异常 - {e}")
            return False

    def test_vllm_compatibility(self, model_name: str) -> bool:
        """测试VLLM兼容性"""
        try:
            model_path = str(self.model_dir / model_name)
            template = MODEL_CONFIGS.get(model_path, ("qwen", True))[0]

            # 创建临时输出
            import tempfile
            temp_output = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
            temp_output.close()

            cmd = [
                "python", "scripts/vllm_infer.py",
                "--model_name_or_path", model_path,
                "--template", template,
                "--dataset", "alpaca_en_demo",
                "--save_name", temp_output.name,
                "--max_new_tokens", "5",
                "--max_samples", "1"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # 清理
            try:
                os.unlink(temp_output.name)
            except:
                pass

            if result.returncode == 0:
                self.logger.info(f"✅ {model_name}: VLLM兼容")
                return True
            else:
                self.logger.error(f"❌ {model_name}: VLLM不兼容 - {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"❌ {model_name}: VLLM测试异常 - {e}")
            return False

    def generate_report(self, results: Dict[str, Dict[str, bool]]) -> str:
        """生成验证报告"""
        report = ["# 模型验证报告\n"]
        report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 统计
        total = len(results)
        basic_ok = sum(1 for r in results.values() if r["basic_loading"])
        factory_ok = sum(1 for r in results.values() if r["llamafactory"])
        vllm_ok = sum(1 for r in results.values() if r["vllm"])

        report.append("## 统计摘要")
        report.append(f"- 总模型数: {total}")
        report.append(f"- 基础加载成功: {basic_ok}/{total}")
        report.append(f"- LLaMA-Factory兼容: {factory_ok}/{total}")
        report.append(f"- VLLM兼容: {vllm_ok}/{total}\n")

        # 详细结果
        report.append("## 详细结果\n")
        for model, result in results.items():
            status = "✅" if all(result.values()) else "⚠️"
            report.append(f"### {model} {status}")
            report.append(f"- 基础加载: {'✅' if result['basic_loading'] else '❌'}")
            report.append(f"- LLaMA-Factory: {'✅' if result['llamafactory'] else '❌'}")
            report.append(f"- VLLM: {'✅' if result['vllm'] else '❌'}\n")

        return "\n".join(report)

    def run_verification(self, tests: List[str] = None) -> bool:
        """运行模型验证"""
        if tests is None:
            tests = ["basic_loading", "llamafactory", "vllm"]

        models = self.get_available_models()
        if not models:
            self.logger.error("❌ 没有找到可用模型")
            return False

        self.logger.info(f"🔍 开始验证 {len(models)} 个模型...")

        results = {}
        for model in models:
            self.logger.info(f"🧪 验证模型: {model}")

            result = {}

            if "basic_loading" in tests:
                result["basic_loading"] = self.test_basic_loading(model)

            if "llamafactory" in tests:
                result["llamafactory"] = self.test_llamafactory_compatibility(model)

            if "vllm" in tests:
                result["vllm"] = self.test_vllm_compatibility(model)

            results[model] = result

        # 生成报告
        report = self.generate_report(results)

        # 保存报告
        report_path = "model_verification_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        self.logger.info(f"📄 验证报告已保存到: {report_path}")

        # 打印摘要
        all_ok = all(all(r.values()) for r in results.values())
        if all_ok:
            self.logger.info("🎉 所有模型验证通过！")
        else:
            self.logger.warning("⚠️ 部分模型验证失败，请查看详细报告")

        return all_ok

def main():
    parser = argparse.ArgumentParser(description="模型验证工具")
    parser.add_argument("--tests", nargs="+",
                       choices=["basic_loading", "llamafactory", "vllm"],
                       default=["basic_loading", "llamafactory", "vllm"],
                       help="选择要运行的测试")

    args = parser.parse_args()

    verifier = ModelVerifier()
    success = verifier.run_verification(args.tests)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    import time
    main()