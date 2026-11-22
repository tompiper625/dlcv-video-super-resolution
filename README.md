# 三帧视频超分辨率 · 课程项目

仓库里的项目资料包含研究问题、模型对照、成员分工、课程说明和 CPU 预实验。A 的可运行源码、数据清单、汇报图及 600 组受控样本均已打包。[项目总指南](01_项目资料/第13题-视频超分辨率-项目总指南.md) · [A 数据交接 Release](https://github.com/tompiper625/dlcv-video-super-resolution/releases/tag/v1.0-a-data)

## 交接文件

| 成员 | 下载 | 开始步骤 |
|---|---|---|
| B · 网络与统一训练 | Release 中的 `vsr_A-data.zip`，以及 REDS 的 `val_sharp.zip` | 将 REDS 原始帧 ZIP 放到交接包解压目录的 `downloads/` 下，按包内 README 第 1 节解压和验收。 |
| C · 对齐与评估 | Release 中的 `vsr_A-data.zip` | 按包内 README 读取联调样例，核对光流方向和掩码。包内另有全部 600 组受控样本。 |

在发布页的 **Assets** 中点击 `vsr_A-data.zip`。B 需要的原始帧 ZIP 约 4.27 GB，可从作者提供的 Hugging Face 或 Google Drive 链接下载。[REDS 清晰帧 ZIP](https://huggingface.co/datasets/snah/REDS/resolve/main/val_sharp.zip) · [REDS 作者页](https://seungjunnah.github.io/Datasets/reds.html)

## 参考资料

- 查评分和提交要求。[课程说明](<01_项目资料/DLCV Project Overview.pdf>)
- 查 REDS 数据来源及引用。[REDS 数据集论文](01_项目资料/REDS-NTIRE-2019.pdf)
- 查看早期线性模型的脚本和测量结果。[CPU 预实验](01_项目资料/vsr_pilot/)
