# 敏感性分析文档索引

## 📚 文档导航

### 🚀 快速开始
- **[主文档 - README.md](README.md)** - 项目概述、快速开始、目录结构
- **[环境设置 - SETUP.md](SETUP.md)** - 安装指南、环境配置、依赖管理

### 🧪 实验流程
- **1. Mini Test100 链路**: 快速验证 (200样本)，确保模型/代码无误。
- **2. 正式大规模链路**: 1k~20k 全量实验，生成最终数据。
- **3. Analyze & Plot**: 结果分析与绘图，生成论文可用图表。

### 🔧 问题解决
- **[问题排查 - TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - 常见问题、调试技巧、解决方案

### 📋 项目文档
- **[产品需求 - PRD.md](PRD.md)** - 详细需求规范、技术架构、实施计划
- **[项目指南 - ../../CLAUDE.md](../../CLAUDE.md)** - 整体项目架构、开发指南

## 🎯 按需求查找文档

### 我是新用户
1. 📖 阅读 [README.md](README.md) 了解项目概况
2. ⚙️ 按照 [SETUP.md](SETUP.md) 设置环境
3. 🚀 运行 `python scripts/sa.py quick` 开始使用

### 我遇到问题
1. 🔍 查看 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 解决常见问题
2. 🧪 运行诊断脚本验证安装
3. 📞 收集错误信息寻求帮助

### 我是开发者
1. 📋 阅读 [PRD.md](PRD.md) 了解技术规范
2. 🏗️ 查看 [CLAUDE.md](../../CLAUDE.md) 了解项目架构
3. 🔧 参考 [SETUP.md](SETUP.md) 的开发环境配置

## 📁 文件结构

```
sensitivity_analysis/docs/
├── index.md                    # 📖 文档索引 (本文件)
├── README.md                   # 🚀 主文档
├── SETUP.md                    # ⚙️ 环境设置指南
└── TROUBLESHOOTING.md          # 🔧 问题排查指南

../docs/prd/
└── llm_sensitivity_analysis_prd.md  # 📋 产品需求文档

../../
└── CLAUDE.md                   # 🏗️ 项目主文档
```

## 🔍 快速导航

| 我想要... | 应该阅读... | 相关命令 |
|-----------|--------------|----------|
| 快速开始使用项目 | [README.md](README.md) | `python scripts/sa.py quick` |
| 设置开发环境 | [SETUP.md](SETUP.md) | `pip install -e ".[torch,metrics]"` |
| 解决常见问题 | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | `python sensitivity_analysis/scripts/test_vllm_fix.py` |
| 了解技术架构 | [PRD.md](PRD.md) | 查看项目规范 |
| 验证安装 | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | `python scripts/sa.py test` |

## 📊 文档统计

- **总文档数**: 5个核心文档
- **主要内容**: 快速开始、环境设置、问题排查、技术规范
- **维护状态**: ✅ 活跃维护中
- **最后更新**: 2025-11-17

## 🆕 文档更新日志

### v2.0 (2025-11-17)
- ✅ 整合4个分散文档为3个核心文档
- ✅ 创建统一的文档索引结构
- ✅ 添加快速导航和搜索功能
- ✅ 优化文档组织结构

### v1.0 (2025-11-15)
- ✅ 创建基础PRD文档
- ✅ 添加环境设置指南
- ✅ 建立问题排查机制

---

**文档版本**: v2.0
**最后更新**: 2025-11-17
**维护状态**: ✅ 活跃维护中
