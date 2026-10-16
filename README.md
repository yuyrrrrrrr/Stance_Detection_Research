# 中文零样本立场检测：领域泛化与数据增强（C-STANCE）

> 基于预训练语言模型的中文**零样本立场检测（ZSSD）**研究 —— 让模型面对**没见过的目标**、**没见过的领域**时，依然能判断文本对某目标的立场是 **支持 / 反对 / 中立**。
>
> 方法主线：**预训练语言模型 + 数据增强（DA）+ 领域泛化（DG）**。

**项目状态：🚧 早期阶段** —— 数据勘察与 EDA 已完成，零样本划分与建模代码开发中。

---

## 目录

- [一、模型是做什么的](#一模型是做什么的)
- [二、数据来源与致敬](#二数据来源与致敬)
- [三、模型来源与致敬](#三模型来源与致敬)
- [四、方法思想来源与致敬](#四方法思想来源与致敬)
- [五、已完成的工作](#五已完成的工作)
- [六、正在进行与计划中的工作](#六正在进行与计划中的工作)
- [七、目录结构](#七目录结构)
- [八、快速开始](#八快速开始)
- [九、评测协议约定](#九评测协议约定)
- [十、引用](#十引用)
- [十一、致谢与免责声明](#十一致谢与免责声明)

---

## 一、模型是做什么的

**输入**：一条中文微博文本 + 一个立场目标（target，可以是名词短语如「垃圾分类」，也可以是一个主张如「垃圾分类问题一直受到人们的关注」）

**输出**：三分类立场

| 标签 | 含义 |
|---|---|
| **支持** | 文本作者对目标持赞成、认同、维护态度 |
| **反对** | 文本作者对目标持批评、质疑、否定态度 |
| **中立** | 文本提及了目标，但未表达明确倾向，或倾向模糊／仅为客观陈述 |

**与普通三分类情感分析的区别 —— 零样本（Zero-Shot）**：

- **Subtask A（目标级零样本）**：测试集中出现的立场目标，在训练集中**完全未出现**。
- **Subtask B（领域级零样本）**：测试样本来自训练中**完全未出现的领域**（如训练时没有「环保」领域，测试全为「环保」）。

**为什么难（本仓库 EDA 实测得到的事实）**：

| 现象 | 实测数值 | 对建模的要求 |
|---|---|---|
| 目标极端稀疏 | 训练集 25,787 个目标中 **94.2% 只出现 1 次** | 必须做**目标级**增强，而非简单复制样本 |
| 中立类跨域漂移 | 中立占比 **28.6%（权益）~ 40.6%（世界事件）**，差 12pp | 需要域不变表示；中立 F1 是核心观察指标 |
| 目标词表跨域几乎不重叠 | 两两重叠仅 **0.25% ~ 2.29%** | target 字面是最强的领域信号，模型易学「认目标」而非「读立场」 |
| 目标粒度差异大 | claim（长句）约占 60%，noun_phrase（短词）约占 40% | 两类目标需分别分析与增强 |

---

## 二、数据来源与致敬

本项目使用 **C-STANCE** 作为唯一的数据基础。

| 项目 | 说明 |
|---|---|
| 数据集 | **C-STANCE**：首个大规模中文零样本立场检测数据集 |
| 规模 | **48,126** 条微博-目标标注对，约 **40k** 个不同目标（含名词短语与主张两类） |
| 领域 | 7 个：新冠疫情、世界事件、文化教育、娱乐和消费、体育、权益、环保 |
| 子任务 | Subtask A（基于目标）、Subtask B（基于领域） |
| 论文 | Zhao, Li & Caragea. *C-STANCE: A Large Dataset for Chinese Zero-Shot Stance Detection.* ACL 2023 (Long Papers), pp. 13369–13385 |
| 论文链接 | https://aclanthology.org/2023.acl-long.747/ |
| 数据来源 | 作者公开的上游数据仓库（本地保存为 `C-STANCE-main/`，内容未做任何修改）；具体地址请以论文页面给出的链接为准 |

**🙏 致敬**：我们由衷感谢 **Chenye Zhao、Yingjie Li、Cornelia Caragea** 三位作者构建并公开了 C-STANCE。中文立场检测长期缺少大规模、多目标、多领域的公开基准，C-STANCE 的发布（比 VAST 大 2.5 倍以上、比此前中文立场数据集大 16 倍以上）直接为中文 ZSSD 研究提供了可能。本仓库的研究问题、数据划分方式、甚至研究中要面对的困难，都源自这份数据集的贡献。

**数据使用说明**：

- 本仓库中的 `c_stance_dataset/` 为**上游数据的原样副本，未做任何修改**（含 Subtask B 各划分的 `In Use` 标记）。
- 数据版权归原作者团队所有，本项目仅将其用于学术研究。**任何使用者请遵守上游仓库的许可与使用条款，并引用上述论文。**
- 本仓库对数据的二次产物（预处理 parquet、增强样本、划分清单）会标注出处，且不会改变原始文件。

---

## 三、模型来源与致敬

本项目**不做预训练**，只在下述开源预训练模型之上做立场检测适配与领域泛化改造。

| 底座模型 | 来源 | 在本项目中的角色 | 链接 |
|---|---|---|---|
| `hfl/chinese-roberta-wwm-ext` | 哈工大讯飞联合实验室（HFL） | **主底座** | https://huggingface.co/hfl/chinese-roberta-wwm-ext |
| `bert-base-chinese` | Google Research | **对照底座**（消融「全词掩码是否有效」） | https://huggingface.co/bert-base-chinese |
| `hfl/chinese-macbert-base` | HFL | 备选底座（阶段 5 的候选之一） | https://huggingface.co/hfl/chinese-macbert-base |
| 中文大语言模型（Qwen / GLM 等）| 各开源团队 | 仅作**数据增强的生成器**（可选组件，可替换为任意 LLM API） | https://huggingface.co/Qwen ・ https://huggingface.co/THUDM |
| 翻译模型 `Helsinki-NLP/opus-mt-zh-en` / `opus-mt-en-zh` | University of Helsinki NLP | 回译增强（可选组件） | https://huggingface.co/Helsinki-NLP |

**🙏 致敬**：特别感谢 **HFL 团队**长期开放中文预训练模型（`chinese-roberta-wwm-ext` / `chinese-macbert` / `chinese-bert-wwm` 等），以及其在 *Revisiting Pre-trained Models for Chinese Natural Language Processing*（Findings of EMNLP 2020）与 CLUE 基准中的系统性工作 —— 中文 NLP 的绝大多数下游任务都直接受益于此，本项目也不例外。同样感谢 **Google Research** 公开 BERT 及其**中文版 bert-base-chinese**，奠定了本任务基线的可比性。感谢 **Helsinki-NLP** 与各开源大模型团队开放模型权重，使数据增强无需从零训练生成器。

**我们不对以上任何模型的预训练成果做任何主张**；本项目只报告在其之上的适配、增强与领域泛化实验结果。

---

## 四、方法思想来源与致敬

本项目是**方法适配与组合设计**，不是方法发明。下文每一项都是已发表工作的思想借用，代码与论文中都会明确标注出处。

| 方法 | 出处 | 在本项目中的作用 |
|---|---|---|
| 零样本立场检测设定 / VAST | Allaway & McKeown, *Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representations*, EMNLP 2020 | 提供了 ZSSD 这一任务范式（英文对应数据集），C-STANCE 的中文设定与之呼应 |
| 监督对比学习（SupCon） | Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020 | 拉近「同立场、不同目标」的样本，让表示不依赖目标字面 |
| 领域对抗训练（DANN / GRL） | Ganin et al., *Domain-Adversarial Training of Neural Networks*, JMLR 2016 | 通过梯度反转层抹去领域特有信号（本项目扩展为「真实域 + 增强风格」联合判别） |
| 原型网络（Prototypical Networks） | Snell, Swersky & Zemel, NeurIPS 2017 | 以类原型做跨域对齐，稳定小样本/零样本场景下的类别边界 |
| 不变风险最小化（IRM）（相关工作） | Arjovsky et al., 2019 | 为「跨域不变性」提供理论视角，作为方法动机的支撑 |
| EDA（简易数据增强） | Wei & Zou, *EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks*, EMNLP-IJCNLP 2019 | 同义词替换／插入／交换／删除四种文本增强操作（中文版需额外做极性词保护） |
| 回译增强（Back-Translation） | Sennrich, Haddow & Birch, ACL 2016 | 以「中→英→中」生成改写文本，增加表层多样性 |
| EDDA（对抗式/表征级数据增强） | Zhang et al.（对抗式数据增强工作）<!-- TODO: 引用前请核对原文标题与出处 --> | 以表征扰动做更「难」的增强，与文本级增强形成互补 |

**🙏 致敬**：本项目提出的任何组合设计（例如把「增强来源」当作风格域标签与真实领域一起送入域判别器，使数据增强与领域泛化真正协同而非简单拼接），都建立在上述工作之上。我们感谢这些作者公开方法、开源实现与实验细节 —— 没有它们，中文 ZSSD 上的领域泛化探索无从下手。同时感谢 **VAST** 等工作在英文 ZSSD 上的先行探索，为中文数据集的设计提供了参照。

---

## 五、已完成的工作

- ✅ **数据勘察与 EDA**：基础分析脚本 + 完整分析脚本 + 分析报告 + 10 张图表
- ✅ **关键事实的量化确认**（见[第一章](#一模型是做什么的)的表格），其中两条对研究设计影响最大：
  - 训练集 **94.2%** 的目标只出现一次 → 数据增强的着力点是**目标级增强**，而不是类别再平衡（三类标签整体已接近 1:1:1）。
  - 跨域中立占比漂移 **12pp**、跨域目标词表重叠 **< 2.3%** → 领域泛化有明确的现实动机。
- ✅ **零样本协议的一致性核查**，并发现两处需要在论文中说清楚的问题：
  - Subtask A 的 train/val/test 目标**并非严格不相交**（存在少量高度模板化的重叠主张目标），因此本项目默认采用「严格协议」并同时报告原始划分结果作对照。
  - Subtask B 的 `In Use == 0` 样本（LexsimTopics，与测试目标字面高度相似）**必须**从训练/验证中过滤，否则零样本假设被破坏。
- ✅ **完整工作流设计**：EDA → 数据预处理 → 严格零样本划分 → 数据增强 → 模型选择 → 训练 → 领域泛化微调 → 评估 → 调优 → 最终评估 → 部署。
- ✅ **硬件与可行性评估**：本地仅 RTX 3050 Laptop（4 GB 显存），可完成调试与评估基建；全量多随机种子训练计划上云执行。

---

## 六、正在进行与计划中的工作

**核心研究主张（一句话）**：把数据增强的**来源**（原始／回译／LLM 生成／EDDA）当作额外的「风格域」标签，与真实领域标签**一起**送入域判别器，使模型对「增强风格」也保持不变性 —— 从而让数据增强与领域泛化**协同**，而不是让模型学会「识别出这是增强样本」。

**总损失形式**：

```
L_total = L_CE + λ_s · L_SupCon + λ_p · L_Proto + λ_d · L_Domain
```

其中 `L_SupCon` 的正样本对限定为「**同立场 + 不同目标**」，`L_Domain` 同时判别「真实领域」与「增强风格」。

**路线图**：

- [x] EDA 与数据事实确认
- [x] 研究协议与评测口径设计
- [ ] **严格零样本划分**：Subtask A 迭代剔除与测试集重叠的目标；Subtask B 过滤 `In Use == 0` 并断言测试域未出现在训练中
- [ ] **数据预处理**：统一 23 个 CSV 为单一 schema，产出可复现的划分清单（含文件校验值）
- [ ] **数据增强**：目标改写／LLM 条件生成／EDDA／回译，配「标签一致性 + 去重 + 泄漏扫描」三级质量闸门，并人工抽检标签一致性
- [ ] **建模与训练**：`chinese-roberta-wwm-ext` 主底座 + `bert-base-chinese` 对照，多随机种子训练
- [ ] **领域泛化微调**：监督对比学习 + 域对抗（真实域 × 增强风格）+ 原型对齐，逐模块消融
- [ ] **评估与统计**：以 Macro-F1 为主指标、**中立类 F1 为关键分析指标**，至少 5 个随机种子报告均值±标准差，与基线做 Wilcoxon 符号秩检验（Holm 校正）
- [ ] **部署**：FastAPI 服务，提供**支持／反对／中立**三分类 API（文本 + 目标 → 立场与概率），并保证线上推理与训练使用同一输入模板
- [ ] **论文／技术报告**

**预期交付物**：

| 类别 | 交付内容 |
|---|---|
| 数据 | C-STANCE 的 7 个领域划分清单 + 通过质量闸门的增强训练集 |
| 模型 | 领域泛化版中文 RoBERTa（含对比学习与域对抗模块）权重 |
| 评估 | Macro-F1 主表（均值±标准差）、中立类 F1 分析、逐领域/逐目标类型明细、显著性检验结果 |
| 服务 | FastAPI 三分类推理接口 + 接口文档 + 上线一致性回归测试 |

---

## 七、目录结构

```
Stance_Detection_Research/
├─ README.md
└─ C-STANCE-main/C-STANCE-main/           # 上游 C-STANCE 数据（原样保留，未修改）
   ├─ c_stance_dataset/
   │  ├─ subtaskA/                        # 目标级零样本：raw_{train,val,test}_all_onecol.csv
   │  └─ subtaskB/                        # 领域级零样本：7 个留一域文件夹，各含 train/val/test
   └─ analysis/                           # ✅ 本项目新增：EDA
      ├─ eda.py                           # 基础 EDA
      ├─ eda_full.py                      # 完整 EDA（含跨域漂移、目标重叠、In Use 过滤量）
      ├─ EDA_report.md                    # 分析报告（面向 DG + DA）
      ├─ EDA_report_outline.md            # 报告大纲
      └─ output/figs/                     # 10 张分析图表

# 以下目录为规划中，将随开发逐步落地：
# src/       训练、增强、评估、服务代码（含统一输入模板模块）
# configs/   实验配置（模板、超参、增强预算、损失权重）
# data/      预处理产物与划分清单（不进版本库的由 .gitignore 排除）
# scripts/   协议检查、批量跑种子、一键复现
# outputs/   日志、checkpoint、结果表与图
# docs/      工作流文档、误差分析、API 文档
```

---

## 八、快速开始

> 当前可直接运行的是 **EDA 部分**（训练、增强与服务的代码开发中，尚未提交）。

```bash
# 1) 环境（当前 EDA 仅需以下依赖）
pip install pandas numpy matplotlib scikit-learn

# 2) 基础 EDA：打印数据规模、标签分布、目标稀疏度、跨域差异等统计
python C-STANCE-main/C-STANCE-main/analysis/eda.py

# 3) 完整 EDA：额外输出 10 张图表到 analysis/output/figs/
python C-STANCE-main/C-STANCE-main/analysis/eda_full.py
```

**读完 EDA 报告后你会知道的三件事**：

1. 为什么本项目的增强重点在**目标**而不是类别平衡；
2. 为什么必须做**领域泛化**（中立占比跨域漂移 12pp、目标词表跨域几乎不重叠）；
3. 为什么零样本划分需要**额外的严格化处理**（Subtask A 存在少量重叠目标；Subtask B 需要过滤 `In Use == 0`）。

---

## 九、评测协议约定

为避免「事后改口径」，以下规则一经确定即冻结，所有结果均按此报告：

| 约定 | 内容 |
|---|---|
| 主指标 | **Macro-F1** |
| 关键分析指标 | **中立类 F1**（隐式表达、反讽、多目标冲突最集中体现在这一类） |
| 随机性 | 至少 **5 个随机种子**，报告**均值 ± 标准差** |
| 显著性 | 与基线做 **Wilcoxon 符号秩检验**（配对）+ 配对 t 检验，多重比较用 **Holm 校正** |
| 零样本严格性 | Subtask A：训练/验证/测试目标**互不相交**；Subtask B：测试领域在训练中**完全未出现**，且训练/验证**必须**过滤 `In Use == 0` |
| 数据增强 | 增强样本**只能进入训练集**；不得包含任何测试目标或测试领域的信息；需通过标签一致性、去重、泄漏扫描三级闸门并人工抽检 |
| 选模型 | 测试集**绝不**用于选择 checkpoint 或调整阈值；跨域场景下明确区分「固定轮数」与「验证集早停」两种口径 |

---

## 十、引用

如果本仓库的代码、划分或分析对你的研究有帮助，请同时引用数据集的原始论文：

```bibtex
@inproceedings{zhao-etal-2023-c,
    title = "{C}-{STANCE}: A Large Dataset for {C}hinese Zero-Shot Stance Detection",
    author = "Zhao, Chenye  and
      Li, Yingjie  and
      Caragea, Cornelia",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.747/",
    pages = "13369--13385",
}
```

使用到的预训练模型，请按各模型页面给出的方式引用，例如：

- `hfl/chinese-roberta-wwm-ext`：Cui et al., *Revisiting Pre-trained Models for Chinese Natural Language Processing*, Findings of EMNLP 2020
- `bert-base-chinese`：Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019
- 回译所用翻译模型：Helsinki-NLP OPUS-MT 系列（见模型页面）
- 数据增强使用的大语言模型（如 Qwen / GLM 等）：请按对应模型页面引用

---

## 十一、致谢与免责声明

**再次致敬**

- 感谢 **Chenye Zhao、Yingjie Li、Cornelia Caragea** 与 **ACL 2023** 公开 C-STANCE —— 没有这份数据集，本项目的研究问题本身就不成立。
- 感谢 **哈工大讯飞联合实验室（HFL）**、**Google Research**、**University of Helsinki NLP** 及所有开源大模型团队公开预训练模型与工具，使中文立场检测的研究不必从零开始。
- 感谢 **SupCon、DANN、Prototypical Networks、EDA、回译、EDDA** 等方法工作的作者公开思路与实现，本项目的一切组合设计都建立在这些基础之上。
- 感谢开源社区（PyTorch、Transformers、scikit-learn、pandas 等）提供的工程基础设施。

**免责声明**

- 本项目为**学术研究项目**，与 C-STANCE 原作者团队、以及任何预训练模型提供方**无隶属或合作关系**；本仓库的观点与结论仅代表本项目作者。
- 数据集与预训练模型的**版权归其各自作者/机构所有**，本项目仅作学术研究使用，不用于任何商业用途。
- 立场检测结果可能包含偏差与错误，**不应作为内容审核、舆情定性的唯一依据**。
- 各上游组件遵循其原始许可；本项目自主新增的实现部分若未来单列许可协议，将以 `LICENSE` 文件为准。