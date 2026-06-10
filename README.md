# ReviewGuard（评盾）：虚假评论/水军识别系统

ReviewGuard（评盾）用于“大数据原理与技术”课程最终大作业：模仿垃圾文本检测流程，完成虚假评论/水军评论识别。

## 项目目标

给定评论文本，判断其是否为虚假评论/水军评论。工程包含：

- 数据预处理：清洗 HTML、URL、数字、特殊符号，中文分词，停用词过滤。
- 特征提取：TF-IDF、字/词 n-gram。
- 模型训练：逻辑回归、朴素贝叶斯、线性 SVM。
- 结果评估：准确率、精确率、召回率、F1、混淆矩阵、错误案例分析。
- 可视化展示：Streamlit 页面支持输入评论并展示预测结果。

## 目录结构

```text
.
├── app.py                         # 视频展示用的可视化演示页面
├── data/
│   ├── raw/                        # 原始 train/dev/test txt
│   ├── train.csv                   # 训练集（label,text）
│   ├── dev.csv                     # 验证集
│   ├── test.csv                    # 测试集
│   ├── hit_stopwords.txt           # 中文停用词表
│   └── sample_reviews.csv          # 小规模演示样例
├── docs/
│   ├── data_statistics.md          # 数据统计报告（脚本生成）
│   ├── figures/                    # 统计图输出目录
│   ├── report_template.md          # 实验报告模板
│   ├── presentation_outline.md     # 课堂展示提纲
│   └── team_plan.md                # 四人分工建议
├── models/                         # 训练后模型输出目录
├── requirements.txt
└── src/
    ├── config.py
    ├── preprocess.py
    ├── data_io.py
    ├── convert_data.py
    ├── data_stats.py
    ├── train.py
    └── predict.py
```

## 快速开始

安装依赖：

```powershell
pip install -r requirements.txt
```

数据转换（从 `data/raw/*.txt` 生成 CSV，若已存在`data/*.csv`，可跳过这一步）：

```powershell
python -m src.convert_data
```

生成数据统计报告与图表：

```powershell
python -m src.data_stats
```

训练模型：

```powershell
python -m src.train --data data/train.csv --model logistic_regression
```

单条预测：

```powershell
python -m src.predict --text "这家店太好了，姐妹们冲，五星好评返现"
```

启动演示页面：

```powershell
streamlit run app.py
```

## 数据格式

数据文件使用 CSV，至少包含两列：

```text
label,text
0或1,评论文本
```

其中（京东数据集标签规则）：

- `0`：虚假/水军评论
- `1`：真实用户评论

预处理由 `src/preprocess.py` 完成，训练时 `train.py` 会自动调用，不需要单独生成清洗后的 CSV。

## 后续建议

1. 收集或构造更真实的数据集，例如电商评论、酒店评论、餐饮评论。
2. 增加水军行为特征，例如重复评论比例、账号评论频率、时间间隔、评分与文本情感不一致。
3. 对比多种模型和特征组合，并把错误案例放进报告和课堂展示。
