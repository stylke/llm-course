# 作业一文件说明

提交正文为 [报告.md](报告.md)。实验已经实际运行，所有结果位于本目录。

课程作业仓库：[llm-course](https://github.com/stylke/llm-course)，本次作业位于仓库的 [`hw1/`](https://github.com/stylke/llm-course/tree/main/hw1) 目录。

仓库提交报告、Python代码、实验结果及日志。下文的 `data/`、`models/`、`.venv/` 和本机工具安装包保留在本地，不随 Git 提交；从仓库下载后请按“环境重建与其他机器”安装依赖，并将原始语料放到项目根目录的 `人民日报语料/1998-上半年-带音/` 后运行实验。

| 文件或目录 | 内容 |
| --- | --- |
| `prepare.py` | 编码检查、标注清理、切句去重、按月份划分和训练词表建立 |
| `run_experiment.py` | 三种阶数各训练3次、验证选阶、测试评测和全部续写 |
| `generate.py` | 独立续写入口，支持温度、top-k、top-p、贪心和随机种子 |
| `analyze.py` | 未见组合比例、概率归一化、评分一致性、有限上下文和生成复现检查 |
| `results/metrics.csv` | 可直接查看的训练时间、大小、困惑度汇总 |
| `results/generations.md` | 17条完整原始续写，含停止原因 |
| `results/*.json` | 数据统计、源文件哈希、参数、训练测量、评测和补充分析 |
| `results/logs/` | 9份训练日志及9份二进制转换日志，保留具体命令 |
| `data/` | 清理后的原词文本、映射后文本及训练词表 |
| `models/` | 各阶ARPA及trie二进制模型 |
| `tools/` | 本机可用KenLM工具、Python安装包和源码包、许可文件 |

## 在本机复现

在项目根目录执行。已经准备了本目录的独立Python环境，不依赖临时目录中的环境：

```bash
hw1/.venv/bin/python hw1/run_experiment.py
hw1/.venv/bin/python hw1/analyze.py
```

全流程会重新清理数据、顺序训练各模型3次并覆盖结果。耗时受机器状态影响，训练过程不需要联网。

只重新评测和生成，不重新训练：

```bash
hw1/.venv/bin/python hw1/run_experiment.py --skip-prepare --skip-training
```

单独续写：

```bash
hw1/.venv/bin/python hw1/generate.py --model hw1/models/4gram.bin --temperature 0.7 --top-k 20 --seed 2026
hw1/.venv/bin/python hw1/generate.py --model hw1/models/4gram.bin --temperature 1 --top-k 0 --top-p 0.9 --seed 42
```

清理程序仅依赖Python标准库，也可直接执行 `python3 hw1/prepare.py`。所有路径均以脚本所在目录为基准。

## 环境重建与其他机器

实际环境为Python 3.9.6、KenLM 0.3.0、macOS ARM64。系统默认的Python 3.13不能直接编译这个版本的KenLM Python扩展，因此本次改用了系统Python 3.9。若本机独立环境被删除，可离线重建：

```bash
/usr/bin/python3 -m venv hw1/.venv
hw1/.venv/bin/python -m pip install --no-index hw1/tools/kenlm-0.3.0-cp39-cp39-macosx_26_0_universal2.whl
```

这个安装包由本机已有的KenLM 0.3.0安装文件重新打包，保留原有平台标记和许可文件，只适合相应macOS/Python环境。`tools/lmplz`、`tools/build_binary`是本机已有工具的副本，仍依赖本机 `/opt/anaconda3/lib` 中的Boost库，不是跨平台发行包。

其他机器建议用Python 3.9—3.11新建环境，通过 `python -m pip install -r hw1/requirements.txt` 安装Python接口，并另外编译KenLM命令行工具。本地已保留上游0.3.0源码包；从仓库下载的副本需要先从 [KenLM 0.3.0](https://pypi.org/project/kenlm/0.3.0/#files) 下载源码包到 `hw1/tools/`，然后解压构建：

```bash
tar -xzf hw1/tools/kenlm-0.3.0.tar.gz -C /tmp
cmake -S /tmp/kenlm-0.3.0 -B /tmp/kenlm-hw1-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/kenlm-hw1-build --parallel 4
python hw1/run_experiment.py --kenlm-bin /tmp/kenlm-hw1-build/bin
```

需要C++编译器、CMake以及Boost等开发依赖；较新CMake若拒绝旧版本项目，可在配置时加 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`。Python接口的安装本身不保证提供 `lmplz`。

## 阅读代码时的几个要点

- 词表和低频词阈值仅用训练集确定。`__RARE__`是普通的可学习词类，KenLM内部的 `<unk>`另行保留，避免二者冲突。
- 预处理按完整分词句子去重，先训练月份，再验证、测试月份。只去除完全相同的句子，没有检测近似转载。
- 三种阶数均采用默认插值改进Kneser–Ney、512M排序预算、不剪枝；没有启用折扣估计失败时的兜底选项。
- 困惑度计入EOS，不计BOS，在映射后的词序列上计算。其含义受分词与词表约束。
- `BaseScore`返回log10概率，温度采样先乘 `ln(10)` 再进行指数换算。每一步都给完整输出词表打分；top-k为0表示不截断词数。
- 生成时排除BOS及两类未知词占位符，再归一化。最大长度是60个新增词，EOS结束时不自动添加句号，也不强制最短长度。
- 贪心可能循环；代码保留原始行为，不额外加入重复惩罚。随机种子是预设值，未根据输出内容筛选。

提交或分享时，正文、4个Python脚本、`requirements.txt`、`README.md`和`results/`即可说明过程；`models/`和`data/`较大，保留在本地供复现使用。
