# 第三方来源、许可与公开边界

## RenderFormer

论文作者：Chong Zeng、Yue Dong、Pieter Peers、Hongzhi Wu、Xin Tong；SIGGRAPH 2025 Conference Papers；DOI 10.1145/3721238.3730595。原创中文学习笔记引用其方法与数据。论文PDF、提取全文与原图只留本地research，不作为本仓库公开附件。

官方代码来自 https://github.com/microsoft/renderformer ，本地固定commit见README；该上游仓库标示MIT许可。vendor不入库，以来源和commit复建。上游模型权重许可与使用条件以对应模型页为准，本仓库不重新分发权重，也不把本仓库LICENSE套用于模型。

## 场景资产

官方康奈尔盒/茶壶实验的几何来源与处理应随实验配置记录；第三方原始或转换网格默认不发布。公开展示图属于本机实验输出而非转载论文图，但仍须检查其场景资产许可。

本地Unity工程涉及Classic Sponza及NTC相关内容。仅在独立工程中导入Sponza_Modular.FBX的小片段并替换常量材质验证导出。用户已要求展示本次建筑骨架A/B渲染；公开生成图片须标注来源与材质/几何删减，**原FBX、衍生OBJ/H5、贴图与原工程内容不纳入公开包**。源目录copyright.txt记录Frank Meinl/Crytek、Marko Dabrovic、Morgan McGuire和Kristijonas Jalnionis的来源/改作关系及radiosity公开使用说明；不据此声称第三方资产为本项目原创。这不是重新给第三方资产授权。原工程未作为插件开发工作区。

## 作业模板与个人信息

教师提供的DOC模板仅用于核对格式，不公开模板原件。报告提交前填写姓名/学号等；公开版应删除不希望公开的个人信息。报告构建源和公开成稿须单独检查本机绝对路径、资产引用和输出状态。

## 本项目代码

LICENSE仅覆盖本仓库原创scripts与unity-exporter代码，不覆盖第三方依赖、模型、论文、场景资产、学校模板或报告所引用材料。学习报告与笔记保留作者权利，第三方引文依其来源归属。
