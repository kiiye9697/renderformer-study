# RenderFormer课程实验

Transformer学习方法及其在三角网格神经渲染中的应用。

[学习入口](https://www.kiiye9697.cn/renderformer-study/) · [方法笔记](https://www.kiiye9697.cn/renderformer-study/learn.html) · [实验记录](https://www.kiiye9697.cn/renderformer-study/experiments.html) · [参考图标准](https://www.kiiye9697.cn/renderformer-study/reference.html)

## 实验内容

- 官方Cornell Box基线校验，明确参考、预训练预测与误差图。
- Cornell受限场景族训练：192组配置，按配置划分训练、验证与测试集。
- 原生Sponza参考与输入能力分析：保留网格、贴图、法线和透明材质，检查GI与表示限制。
- 历史相机、光源边界测量独立保留，不与新基线合并。

官方Cornell示例推理达到PSNR 38.30 dB、SSIM 0.9771。场景族训练在10000步验证集上达到21.46 dB、0.8163，目前暂停，尚未进行最终测试。两组实验条件不同，不直接横比。[训练结果](https://www.kiiye9697.cn/renderformer-study/course-training.html)保留完整验证记录。

## 文件

`docs/`为GitHub Pages内容；`scripts/build_learning_entry.py`生成入口、方法与结果页。课程报告共6页，见[Word报告](https://www.kiiye9697.cn/renderformer-study/assets/course-report.docx)，生成方式见`report/README.md`。本地输出为相邻的`../report/RenderFormer-课程报告.docx`；原有“期末报告”PDF/Word属于历史实验。

未验收的新参考图只留在本地审核材料中，不作为已完成结果发布。公开页面不包含原始资产、模型、HDR或缓存。

## 复现

当前作者代码固定为`iamNCJ/RenderFormer-Studio@b128481a48a8e77b8dbbd1d4fe68e72615f68ae9`。官方基线权重为`microsoft/renderformer-v1-base@8aa0009563789dbb6c7785a26aac5da2c9b6d117`。历史测量使用旧仓库`microsoft/renderformer@eec6a9b8d592303def45e458dd4ebd09d6572abc`；版本分别记录。第三方资产须按原许可自行取得。

网页只发布PNG、测量JSON及脚本，原始EXR、H5、模型与缓存留在实验目录。报告封面的姓名、班级、学号和日期需要提交者填写。
