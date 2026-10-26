# Transformer 与 RenderFormer：数据科学期末学习项目

以 Transformer 为学习方法，以 SIGGRAPH 2025 RenderFormer 为图形学应用实例，解释输入表示、注意力、监督学习和神经渲染，并记录当前机器上的真实推理实验。**优先完成课程报告；不把运行预训练模型称为复现完整训练。**

## 阅读入口

- [期末报告 PDF](report/RenderFormer-期末报告.pdf) / [DOCX](report/RenderFormer-期末报告.docx)：公开版封面不含个人信息。
- [报告源文稿](report/report-draft.md)与构建脚本、公式源位于report/。
- [中文论文精读](research/RenderFormer-精读记录.md)：含方法与训练细节、页码、原论文/发布版本差异、局限和证据级别。
- [学习网页](docs/index.html)：静态展示，不提供在线GPU推理。实验状态以网页数据中的实际记录为准。
- [Unity导出器](unity-exporter/README.md)：独立UPM工具，严格限定静态网格/常量URP材质/方形透视相机。
- [第三方与公开边界](THIRD_PARTY.md)、[公开清单](PUBLIC_FILES.md)。

## 实验设计与可信度

A. 官方案例检查转换、推理与Cycles参考流程。B. 康奈尔盒＋茶壶分别改变粗糙度和侧墙颜色，保留误差与失败。C. 来自本地Unity工程的几何片段仅在独立副本做受控材质测试，不能代表原游戏外观；相关第三方资产默认不公开。

两边共享转换后的三角形、法线、相机和材质定义。保留线性HDR原始结果，展示统一曝光与色调映射；指标声明变换与范围。参考采样收敛单独检查。加载、转换和稳态推理计时分开。本项目没有执行论文规模重新训练，也未依据单一示例宣称严格分布外泛化。

## 官方来源与固定版本

- [论文](https://arxiv.org/abs/2505.21925)，[项目](https://microsoft.github.io/renderformer/)，[官方代码](https://github.com/microsoft/renderformer)。
- 本次核验代码commit：`eec6a9b8d592303def45e458dd4ebd09d6572abc`。
- 主方法是原论文205M base；发布模型修复训练数据shader问题后重训，因此实测并非论文原checkpoint数值复现。
- 实际模型ID、权重revision、硬件、软件与测量信息需读取实验版本记录；模型文件不入库。

```sh
git clone https://github.com/microsoft/renderformer vendor/renderformer
git -C vendor/renderformer checkout eec6a9b8d592303def45e458dd4ebd09d6572abc
```

根据官方README在独立Python环境安装依赖；Windows可用PyTorch SDPA，不强制安装Flash Attention。下载固定权重snapshot到本地目录。`scripts/`包含转换后的参考渲染、测量推理与指标工具；运行前查看各脚本参数和本机路径配置。EXR/H5/Blender工程及大模型保留在本地实验目录。不要直接在原Unity工程中测试导出插件。

## 已完成结果与证据

官方康奈尔盒、金属茶壶五档参数、Sponza中庭建筑骨架均已完成真实A/B与误差评估。Sponza保留明显失败结果，金属强高光另有8192spp独立参考收敛诊断；不以显示域高PSNR掩盖线性HDR误差。真实模型hook记录了输入/输出shape与激活统计，明确区别原论文10维材质表示与发布base的13×32×32纹理编码。

[在线学习页](https://kiiye9697.github.io/renderformer-study/) · [公开测量记录](public-data/) · [源代码仓库](https://github.com/kiiye9697/renderformer-study)

论文表格数字属于作者结果，本地测量只从本次运行记录填入；训练未执行。教学动画、结构图与真实测量分别标注。
