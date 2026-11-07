# 作业三文件说明

对应 `2026-秋季-大语言模型与应用实践-L2-part2.pptx` 第68页。正文为 [报告.md](报告.md)。代码实现中文BERT标点恢复微调，以及长文顺序滑窗、批量重叠滑窗的评测。

本次已在Apple M4、16GB内存、MPS设备上实际运行3轮端到端微调，完成553篇测试文章评测、23篇长文的5组速度对比和11项回归测试。

## 文件

| 文件 | 内容 |
| --- | --- |
| `download_model.py` | 下载固定版本的预训练模型，校验权重SHA256 |
| `prepare.py` | 9801文章合并、清理、字符标注和日期划分 |
| `modeling.py` | tokenizer标签对齐、动态padding、设备和指标函数 |
| `train.py` | BERT端到端微调、逐轮验证、保存最佳模型 |
| `calibrate.py` | 在验证片段上选择O类logit偏置 |
| `inference.py` | 顺序解码、固定滑窗、batch推理和重叠概率融合 |
| `evaluate.py` | 融合选型、整篇测试、基线、速度实验与示例 |
| `predict.py` | 加载微调模型恢复新文本标点 |
| `test_hw3.py` | 11项回归测试 |
| `analyze.py` | 从预测复算指标、加载核对、生成图表 |
| `run.py` | 执行单个步骤，把实际输出保存在日志 |
| `data/` | 三组片段数据、整篇文章和段落数据 |
| `models/pretrained/` | UER原始权重、配置、词表和模型卡 |
| `models/punctuation/` | 本次微调后的模型和tokenizer |
| `results/*.json` | 数据统计、训练、校准、融合选择、测试和计时结果 |
| `results/test_predictions.jsonl` | 553篇测试文章的真实与预测标签 |
| `results/*.png` | 训练曲线、逐标点F1和吞吐量图 |
| `results/logs/` | 各步骤实际执行日志 |
| `tools/` | 阅读的模型说明及Transformers序列标注文档 |

## 本机运行

在项目根目录执行。独立环境为 `hw3/.venv/`，Python 3.10.18、PyTorch 2.8.0、Transformers 4.57.1。现有模型和数据已齐，下面的下载步骤只校验本地文件。

```bash
hw3/.venv/bin/python hw3/run.py download
hw3/.venv/bin/python hw3/run.py prepare
hw3/.venv/bin/python hw3/run.py train --device mps
hw3/.venv/bin/python hw3/run.py calibrate --device mps
hw3/.venv/bin/python hw3/run.py evaluate --device mps
hw3/.venv/bin/python hw3/run.py test
hw3/.venv/bin/python hw3/run.py analyze --device mps
```

重新训练会覆盖微调模型和训练记录。校准、评测及分析依次运行后更新相应结果，报告正文保持原样。运行日志由 `run.py` 自动写入 `results/logs/`。GPU同步计时在推理代码中完成。

`--device auto` 会依次选择CUDA、MPS、CPU，也可显式指定设备。普通CPU可以运行整个流程，耗时通常更长。本机受限执行环境中的MPS访问需要相应设备权限；这次正式训练和评测使用了MPS。

## 单独预测

```bash
hw3/.venv/bin/python hw3/predict.py \
  '全国篮球联赛结束球队获得冠军大家向运动员表示祝贺' --device mps
```

读取UTF-8文件并写出恢复结果：

```bash
hw3/.venv/bin/python hw3/predict.py --file hw3/input.txt \
  --output hw3/output.txt --batch-size 8 --device auto
```

默认读取 `results/calibration.json` 的O类偏置，以及 `results/merge_selection.json` 的融合规则。`--merge mean` 或 `--merge center` 可指定融合方式。输出保留预处理后的字符顺序，恢复逗号、句号、顿号、冒号、分号、问号和叹号。输入空白、引号、括号等排版符号按照训练时的规则清理。

现代新闻和古文示例见 [demo.json](results/demo.json)，一篇超过1500字的原输入、参考和实际输出见 [long_example.json](results/long_example.json)。模型训练数据是现代新闻，这次古文示例中存在较多错误。

## 其他机器

使用Python 3.10建立独立环境：

```bash
python3.10 -m venv hw3/.venv
hw3/.venv/bin/python -m pip install -r hw3/requirements.txt
hw3/.venv/bin/python hw3/run.py download
hw3/.venv/bin/python hw3/run.py prepare
hw3/.venv/bin/python hw3/run.py train --device auto
hw3/.venv/bin/python hw3/run.py calibrate --device auto
hw3/.venv/bin/python hw3/run.py evaluate --device auto
hw3/.venv/bin/python hw3/run.py test
hw3/.venv/bin/python hw3/run.py analyze --device auto
```

下载脚本依赖 `curl`，默认从 `hf-mirror.com` 下载固定revision；可用 `download_model.py --endpoint https://huggingface.co` 切换主站。权重大小35,183,943字节，SHA256固定为 `c60f6fde023a817b263dab52a4d0767ae139be98971da4a1a070c695ad24612e`。

课程语料路径为 `人民日报语料/1998-上半年-带音/1998-01-2003版-带音.txt`。路径不同可用 `prepare.py --source 路径`。CUDA平台安装PyTorch时可选与本机CUDA对应的官方发行包，其他代码保持相同。训练默认batch 32、3轮、学习率5e-5，参数可通过命令行调整。

本次完整安装版本保存在 [checks.json](results/checks.json) 的 `packages` 项。随机种子固定为2026；设备、数值后端及依赖差异可能带来数值或时间变化。

## 结果约定

训练使用1月1—20日文章，验证为21—24日，测试为25—31日。同一篇文章的段落保持在同一个集合。最佳轮次、O类偏置和概率融合方式都在验证集上选择。

速度实验使用测试集中长度至少1500字的23篇文章，共54,129字，每个配置重复3次。主测试指标按整篇恢复结果计算；`training_metrics.json` 中另保留了未校准的256字片段评测。

`.gitignore` 排除了虚拟环境、大型数据、模型和阅读材料，这些文件仍在本机 `hw3/` 中。报告、代码、逐篇预测、图表、统计和日志可一并提交。新闻语料用于课程实验，保留原提供方的使用约定。
