# 公开文件白名单与发布前检查

本文件是候选清单，不表示已经上传；最终add/commit/push由主代理审核执行。

| 目录/文件 | 公开内容 | 排除内容 |
|---|---|---|
| README/.gitignore/LICENSE/THIRD_PARTY/PUBLIC_FILES | 项目入口、许可边界与公开规则 | 无 |
| research/RenderFormer-精读记录.md | 原创中文精读、引用、PDF哈希 | PDF、全文摘录txt、论文原图 |
| research/input-output-training.md、research/training-results.md | 原创输入输出教学说明、实际训练方法/曲线解读与留出评估结论 | 权重、原始日志、个人信息 |
| report/ 的明确白名单 | 最终 RenderFormer-期末报告.pdf/.docx、report-draft.md、build_equations.py、build_pdf.py、build_report.py、build_report_figures.py、equations.json、equations/、figures/、result-table.json（空白个人信息封面） | 学校原模板、QA缓存、渲染日志、个人信息、旧工作稿、private/、.deps/、template-check.md |
| scripts/*.py, *.ps1 | 原创复现实验工具 | 环境、密钥、模型、第三方网格 |
| scripts/training/*.py、*.md | 原创数据准备、审计、缓存、容量探测、DPT微调与评价脚本/说明 | __pycache__、特征缓存、数据集原图、模型和输出头权重 |
| unity-exporter/ | package.json、Editor源码/asmdef、README、Tests~ | 任何原Unity工程或Sponza衍生资产 |
| docs/ | 网页HTML/CSS/JS、状态JSON、脱敏training.json（真实历史/停止/全量测试指标）、已核实可公开的本机展示图 | 未运行冒充结果、原论文截图、未注明来源的第三方展示图 |
| public-data/（如存在） | 清理后的版本/指标/实验配置、io-guide.json、训练划分与参数摘要 | 本机绝对路径、用户目录、原资产路径、秘密或原始日志 |

整个runtime、vendor、experiments默认不公开。实验选定指标或配置应复制并清理到public-data，网页展示图复制到docs/assets后重新核验状态。所有H5、EXR、Blender、NPZ、FBX、OBJ、权重文件都被忽略。

训练公开图片限已审核的本机 A/B/C 展示 PNG、固定色标误差图与真实曲线；A=Cycles 参考、B=预训练、C=微调。原始 EXR/H5、缓存张量与Sponza源资产保留本地。原创训练脚本目前含固定 F 盘工作目录，异机复现须按脚本说明修改路径并自行准备获许可的资产；脚本路径不能被复制进 public-data 的脱敏记录。

`.gitignore`不是发布授权，不保证docs中新加入文件自动安全。最终需审查实际暂存列表与diff、检查报告状态、权重revision、图像是否确为本机运行输出及第三方许可；勿用`git add -f`绕过边界。

