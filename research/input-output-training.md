# RenderFormer：输入、输出与训练计数

证据状态：**CODE_AUDITED / CHECKPOINT_RUN（推理）**。本文说明已锁定发布模型的真实接口，以及小规模教学微调的设计。目前已完成 128 像素三步训练 smoke test，未保存其权重；这只验证训练路径可执行，不证明收敛。正式训练仍需完整日志与独立评估；已有八组渲染结果是预训练权重推理结果。

代码固定为 `eec6a9b8d592303def45e458dd4ebd09d6572abc`；模型为 `microsoft/renderformer-v1-base`，权重 revision 为 `8aa0009563789dbb6c7785a26aac5da2c9b6d117`。论文方法见[精读记录](RenderFormer-精读记录.md)，实际激活形状见[本机推理记录](../docs/shapes.json)。以下 B 表示场景批量、N 表示补齐后的三角形数、V 表示相机数；它们不是图片对照中 A/B 的标签。

## 1. 网络到底吃什么

| 数据 | 加入批维后的形状 | 含义 |
|---|---|---|
| triangles | B×N×3×3 | 每个三角形三个顶点的场景坐标；进入底层模型时展平为 B×N×9 |
| vn | B×N×3×3 | 三个顶点的法线；底层展平为 B×N×9 |
| texture | B×N×13×32×32 | 每个三角形的材质属性小块，并非渲染参考图 |
| mask | B×N | 布尔有效性；本实现 **True 表示有效三角形**，补齐项应为 False |
| c2w | B×V×4×4 | 相机到场景坐标的刚体变换 |
| fov | B×V×1 | 透视视场角，pipeline 接受度，射线生成器接受弧度 |

官方 H5 保存上述数据但不带 B 维；fov 原存为 V，调用时补最后一维。加载器将几何、材质、相机均转为 float32；H5 中 texture 原存 float16。autocast 下中间特征可为 float16，不代表全部输入都必须预先转半精度。

13 通道的**实际次序**为 diffuse RGB（0–2）、specular RGB（3–5）、roughness（6）、normal-map RGB（7–9）、emissive RGB（10–12）。转换脚本把最后三通道的变量命名为 `irradiance`，其来源实际是 `material_config.emissive`；它不是参考渲染得到的间接照度标签。无纹理常量材质在三角形小块有效区域内重复，区域外清零；官方掩码条件为整数网格 `x+y<=32`。法线贴图常量为 (0.5,0.5,1)，不同于独立输入 vn 的几何法线。展平后每个材质向量长 13×32×32=13312，再投影到 768 维。**不能把当前发布版说成论文主线的十个常量直接投影。**

场景转换与网络归一化是两件事。官方 `normalize_to_unit_sphere` 在启用时先减去网格顶点均值，再除以最大顶点距离的两倍，所得最大半径为 0.5，随后应用物体旋转、缩放、平移。它按物体处理，不是自动把最终全场景压入单位球。已准备好的统一场景必须按自身导出规则核对，不可再对模型一边单独归一化。法线和材质编码的 RMSNorm 则是网络内部特征操作。

## 2. 相机怎样变成查询，输出是什么

第一阶段在场景坐标处理 N 个三角形，并添加 16 个 register token。顶点位置通过空间 RoPE 参与注意力，法线与材质编码进入 token。当前发布配置中法线位置编码将 9 维变为 117 维，再投影至 768 维。

`turn_to_cam_coord=true` 时，pipeline 为第二阶段把三角形变换到每个相机坐标，并将对应 c2w 换为单位矩阵；第一阶段的场景输入不因此改成相机坐标。射线采用像素中心采样、相机前方 −Z、图像向上 +Y，方向归一化。射线原点为 B×V×3，方向为 B×V×H×W×3。每个 8×8 像素方向束展平为 192 维，经投影得到一个 768 维查询。

例如已实测 Cornell 场景 N=5633、B=V=1、H=W=512：场景序列为 5649×768，图像查询为 4096×768。第二阶段 cross-attention 读取场景特征，self-attention 交换查询信息，DPT 解码器生成图像。解码器内部激活不是可直接解释的照度图，也不是注意力矩阵。

底层模型输出形状为 **B×V×3×H×W**，当前 HDR 配置的输出表示为对数域值 z。pipeline 改为 **B×V×H×W×3** 并执行：

\[
I_{\mathrm{linear}}=10^{z}-1,\qquad z=\log_{10}(1+I_{\mathrm{linear}}).
\]

这不是 sRGB，也不是 PNG 显示值。末端使用 ELU(alpha=0.001)，因此对数预测允许很小的负数，逆变换也可能产生小负值；不能宣称网络输出天然严格非负。现有评估只在显示变换时截断负值，线性域误差保留原值。pipeline 同时会对输入 texture 的最后三个发光通道原地执行 log10(1+x)。重复调用必须复制原始 texture，避免二次取对数。训练用的标签应是与输入网格、材质、光源、相机严格对应的线性 HDR 参考图，转换到同一对数表示；负值、非有限值必须先检查并明确处理策略，不能悄悄更换标签。

网页图片另经固定 EV0、逐通道 Reinhard 和 sRGB 显示变换。显示域 PSNR/SSIM、线性 HDR RMSE、训练对数域损失回答不同问题，不能相互替代。低粗糙度茶壶已经显示：少量强高光的线性误差可很大，而显示域整图仍较接近。

## 3. 为什么不能直接给推理 pipeline 加 backward

官方 pipeline 加载后调用 model.eval，并用 torch.no_grad 包围前向；直接对其输出求梯度不能完成训练。教学训练需要显式准备底层 model.forward 所需的材质、射线与相机坐标三角形，设定可训练参数，保留可训练部分的计算图，再计算损失、反传和更新。固定几何的坐标预处理不需要几何梯度，但不能把可训练解码器一起放进 no_grad。

候选节省显存方案是冻结主干，只微调 `view_transformer.out_dpt`，microbatch=1、梯度累积=4，先以低分辨率探测。**128 像素三步 smoke test 已完成；这些候选超参数仍须以正式运行配置确认，不能视为完整论文训练复现。**冻结主干时可缓存同一输入的确定性特征，但必须保证特征对应正确的场景、相机、分辨率、权重和预处理；更换任一条件应使缓存失效。解码器微调能够改变输出映射，不能自动解决主干场景表示缺失。

教学损失可先采用对数域 L1，以容易解释的有限目标确认梯度与参数更新；具体最终损失以训练配置为准。原论文使用对数域 L1 加 0.05 倍色调映射 LPIPS，且训练规模远大于本项目。若省略 LPIPS，必须称作教学微调目标，不能声称复现了原论文完整训练。

## 4. epoch、step 和 batch 到底数什么

先定义一个样本是一个“场景配置与相机对应的图像监督对”，不能把三角形或像素直接当成独立训练样本。若每项含多视角，需同时记录场景数与监督图像数。

- microbatch：一次前向和反向装入设备的样本数 m，直接影响峰值显存。
- 梯度累积次数 a：经过 a 次 microbatch 反向后才更新一次参数。损失需按实际累积样本数正确缩放，最后不足一组不能盲目仍除以 a。
- 有效 batch：等长样本、单卡时为 m×a；数据并行 w 卡且均正确平均时为 m×a×w。
- optimizer step：一次 optimizer.step，参数实际更新一次。学习率计划若按 step，应按更新次数推进，而不是每个 microbatch 推进。
- epoch：完整遍历一次训练集。若训练集有 D 个样本，单卡、无丢尾时每 epoch 有 ceil(D/m) 个 microbatch；在每 epoch 末结算余量时有 ceil(ceil(D/m)/a) 次参数更新。

例如 D=12、m=1、a=4，一轮是 12 次前后向、3 次参数更新。训练 20 轮只有 60 次更新。该数字是教学算例，不是本项目训练日志。增大 epoch 不保证收敛；同一组少量图像反复见到很多次也不等于扩大数据覆盖。累积可降低同等有效 batch 的显存压力，但不会产生新数据，也不能保证与一次大 batch 数值完全等价。

## 5. 怎样有证据地说“训练有效”

先冻结训练/验证/测试划分，再选择学习率、batch 和停止规则。相近视角与重复配置容易泄漏；同一几何的留出视角最多说明有限的视角适应，不能称陌生场景泛化。既有八组展示图不能不经划分直接一边训练一边当独立测试。

最小证据链是：单步 loss 有限、可训练参数梯度非零且更新；记录固定验证集的 step=0 预训练基线；持续记录训练损失与固定验证损失；只按验证集选最优 checkpoint；最后对未参与选型的测试配置测量。除主要训练目标外，保留显示域和 HDR 指标，检查局部高光与其他条件是否退步。

早停需预先记录验证间隔、最小改进量 min_delta 和 patience（以验证次数计），连续若干次未改善后停止并恢复最佳验证权重。有限步数内曲线趋平只说明这个小数据设置下趋于稳定，不等于全局最优。训练下降而验证上升提示过拟合；两者不降需检查学习率、梯度、标签和输入，而非只加 epoch。损失非有限或输出异常应停止诊断。

日志至少保存 split、随机种子、数据与基础权重版本、可训练参数列表、分辨率、m/a/有效 batch、学习率及调度、epoch、microstep、optimizer step、训练/验证曲线、最佳 checkpoint 和独立测试结果。没有这些证据时只报告“准备中”“单步通过”或“有限步微调”，不提前写“已收敛”。

## 6. 源码核查入口

以下链接全部固定到本次代码 commit：

- [H5 转换：材质次序、三角形小块掩码、相机矩阵](https://github.com/microsoft/renderformer/blob/eec6a9b8d592303def45e458dd4ebd09d6572abc/scene_processor/to_h5.py)
- [物体归一化与变换](https://github.com/microsoft/renderformer/blob/eec6a9b8d592303def45e458dd4ebd09d6572abc/scene_processor/scene_mesh.py)
- [H5 数据加载](https://github.com/microsoft/renderformer/blob/eec6a9b8d592303def45e458dd4ebd09d6572abc/infer.py)
- [pipeline：发光取对数、坐标转换、no_grad 与输出逆变换](https://github.com/microsoft/renderformer/blob/eec6a9b8d592303def45e458dd4ebd09d6572abc/renderformer/pipelines/rendering_pipeline.py)
- [模型输入与有效 mask](https://github.com/microsoft/renderformer/blob/eec6a9b8d592303def45e458dd4ebd09d6572abc/renderformer/models/renderformer.py)
- [射线生成](https://github.com/microsoft/renderformer/blob/eec6a9b8d592303def45e458dd4ebd09d6572abc/renderformer/utils/ray_generator.py)

本文的接口结论依据发布版源码和实际 hook；训练部分是方法与验证协议，执行状态由单独训练记录更新。请勿把协议中的候选值读取为已测结果。
