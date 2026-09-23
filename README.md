# 机器学习：从直觉到推导

打开 `index.html` 开始学习。网站是静态文件；学习进度保存在当前浏览器中。

## 内容

- `roadmap.html`：课程知识地图、章节关系、建议顺序。
- 每个章节的 `theory.html`、`derivation.html`、`visualization.html`、`exercises.html`、`code.html`：直觉、公式、SVG 图、手算与问题、代码阅读。
- `code.py`：可下载运行的 Python 示例，代码附中文注释。
- `errata.html`：原讲义中需注意的符号、条件和笔误。
- 原讲义 PDF 未随公开网站发布。

公式使用 MathJax 在线资源。首次打开公式页需要网络；若资源不可用，页面仍保留 TeX 公式文本，可在自己的讲义副本中对照。

Python 示例需要 NumPy。库对照示例还需要 scikit-learn；约束优化代码需要 SciPy。可在自己的 Python 环境中安装：

```text
python -m pip install numpy scipy scikit-learn
```

## 学习建议

每章按“先懂思想 → 逐步推导 → 几何图解 → 手算与训练 → 代码实现”学习。先独立算小例子，再运行代码核对。进度标记只记录你自己确认学完的单元。

本网站覆盖讲义六章的主干知识并逐个解释所选核心公式；原讲义中的课外证明、部分引理和全部编号公式并未逐式重排。需要完整公式序列时请对照原讲义。

