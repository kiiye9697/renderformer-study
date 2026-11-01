# RenderFormer课程实验

Transformer学习方法及其在三角网格神经渲染中的应用。

[实验网页](https://www.kiiye9697.cn/renderformer-study/) · [报告PDF](report/RenderFormer-期末报告.pdf) · [报告Word](report/RenderFormer-期末报告.docx)

## 实验内容

- 官方康奈尔盒与茶壶材质参数复现。
- Bunny与简化Sponza的DPT解码器微调：256像素，18/6/6帧划分，DPT100轮加联合适配60轮，按最低验证损失选择权重。
- 论文边界条件：内部相机、内部光源、大面积光源、9光源直接预测与逐光源相加。

本轮固定几何中的粗糙度留出结果，Bunny显示PSNR约19.61→35.55 dB，Sponza约3.08→16.47 dB。Sponza归一化边缘误差下降25.3%，但仍有明显模糊；9光源分解的PSNR低于直接预测。逐视图指标、图像及训练曲线位于`docs/assets/server256full/`。

## 文件

`report/`保存8页课程报告和报告图；`scripts/server_study/`保存本轮训练与导出脚本；`scripts/prepare_paper_badcases.py`生成边界条件配置；`scripts/build_course_site.py`从已测数据生成网页和图表。

新网页保留方法、结果、边界实验和下载入口，已移除实时材质预览及重复面板。历史512与256像素测量仍保留为版本记录，不与本轮256像素结果合并。

## 复现

官方代码固定为`eec6a9b8d592303def45e458dd4ebd09d6572abc`；模型为`microsoft/renderformer-v1-base`，revision为`8aa0009563789dbb6c7785a26aac5da2c9b6d117`。环境与步骤见[scripts/server_study](scripts/server_study/README.md)。第三方源资产须按原许可自行取得。

网页只发布PNG、测量JSON及脚本，原始EXR、H5、模型与缓存留在实验目录。报告封面的姓名、班级、学号和日期需要提交者填写。
