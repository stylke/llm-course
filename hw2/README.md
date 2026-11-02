# 作业二文件说明

正文为 [报告.md](报告.md)，补充阅读记录为 [源码阅读笔记.md](源码阅读笔记.md)。本次对应 `2026-秋季-大语言模型与应用实践-L2.part1.pptx` 第137页，并参考第123—136、143—149、154—170页。

实验已经在本机运行。表示学习使用独立的 THUCNews 中文新闻标题子集；分类使用课程提供的人民日报9801带音语料，不调用 `hw1/` 的脚本或模型。

| 文件 | 内容 |
| --- | --- |
| `download_data.py` | 下载外部新闻标题文件，校验本次使用的 SHA256 |
| `prepare.py` | 标注清理、合并文章、版面代理标签、日期划分、外部语料分词 |
| `run_experiment.py` | Gensim TF-IDF、LSI、Word2Vec、FastText 及逻辑回归分类 |
| `supervised.py` | 官方 fastText 监督分类，比较词unigram和bigram |
| `analyze.py` | 指标复算、加载一致性、OOV/子词/哈希分析、随机种子重复 |
| `predict.py` | 独立使用训练好的 FastText + IDF 四分类模型 |
| `test_hw2.py` | 预处理、划分和向量处理的回归测试 |
| `results/metrics.csv` | 30组表示学习分类结果 |
| `results/*.json` | 数据统计、参数、词相似度、预测、误差分析、验证选型 |
| `results/*.png` | 模型比较图与四分类混淆矩阵 |
| `results/logs/` | 实际执行的预处理、训练、监督分类、检查日志 |
| `data/` | 下载的语料和预处理文件 |
| `models/` | 本机训练得到的 Gensim、逻辑回归及监督式fastText模型 |
| `tools/` | 本地阅读的论文页面、数据说明及fastText源码包 |

## 本机复现

在项目根目录运行。最终环境为 Python 3.10.18、Gensim 4.3.3、NumPy 1.26.4、SciPy 1.13.1，独立虚拟环境位于 `hw2/.venv310/`：

```bash
hw2/.venv310/bin/python hw2/download_data.py
hw2/.venv310/bin/python hw2/prepare.py
hw2/.venv310/bin/python hw2/run_experiment.py
hw2/.venv310/bin/python hw2/supervised.py
hw2/.venv310/bin/python hw2/analyze.py --repeat-seeds
hw2/.venv310/bin/python -m unittest discover -s hw2 -p 'test_*.py' -v
```

本机语料已齐，下载步骤只会校验已有文件，不需要再次联网。后续步骤全部离线运行。训练与评测通常为几分钟以内，具体时间取决于机器负载。重跑会覆盖模型和对应结果。

仅使用现有词向量重新分类评测：

```bash
hw2/.venv310/bin/python hw2/run_experiment.py --skip-training
```

这个选项加载已有的Word2Vec/FastText模型，并重建TF-IDF、LSI和逻辑回归。结果文件会更新，报告正文保持原样，其中的训练时间取自正式运行记录。

单独预测：

```bash
hw2/.venv310/bin/python hw2/predict.py '全国篮球联赛结束，球队获得冠军。'
hw2/.venv310/bin/python hw2/predict.py --segmented '全国 篮球 联赛 结束 球队 获得 冠军'
```

默认用 jieba 分词，也可用 `--segmented` 传入已经分好的词。新文本的分词可能与人民日报原分词不同。输出为分类器的原始概率，尚未做概率校准。

## 其他机器

使用 Python 3.10 或3.11建立环境；固定版本的 NumPy 1.26.4 不支持 Python 3.13：

```bash
python3.10 -m venv hw2/.venv
hw2/.venv/bin/python -m pip install -r hw2/requirements.txt
hw2/.venv/bin/python hw2/download_data.py
hw2/.venv/bin/python hw2/prepare.py
hw2/.venv/bin/python hw2/run_experiment.py
hw2/.venv/bin/python hw2/supervised.py
hw2/.venv/bin/python hw2/analyze.py --repeat-seeds
```

另需将课程语料放在项目根目录的 `人民日报语料/1998-上半年-带音/1998-01-2003版-带音.txt`。官方fastText Python扩展若没有对应平台的wheel，需要本机C++编译器。网络不通时，也可以直接取得同一外部文件放到 `hw2/data/raw/thucnews_train.txt`，再执行下载脚本做哈希校验。

`.gitignore` 排除了虚拟环境、大语料、模型和下载的上游源码；这些文件仍保留在本机 `hw2/` 中。代码、报告、小型结果和日志可直接提交。数据遵循原提供方条款，仅用于课程实验；不重新发布完整原始新闻。

## 数据与评测约定

9801的“国际、体育、国内经济、科教文卫”标签根据工作日版面编号自动生成。实验限定1月5—27日中的工作日，训练1月5—16日，验证1月19—21日，测试1月22、23、26、27日。一篇报道的全部段落只进入一个集合。分类器根据文章内容预测版面标签。

TF-IDF、LSI、IDF和逻辑回归只拟合9801训练集。外部语料的类别只用于控制抽样数量，模型使用清理后的文本。`ft_adapted`从头在“外部标题 + 9801训练句子”的并集上训练。二、三、四分类沿用同一套标签和日期划分。

所有分类器参数由验证集选定，测试集用于最终比较和误差分析。`selection.json`记录Gensim表示管线按验证Macro-F1选出的模型，官方fastText单独记录。随着类别增加，任务中的文章数量和类别构成也随之变化。
