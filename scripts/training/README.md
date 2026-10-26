# 受限场景 DPT 输出头微调

本目录实现本地教学实验，不是官方训练代码，也不是从零重现论文训练规模。
预训练 RenderFormer 主干全部冻结，仅训练 `view_transformer.out_dpt` 的 10,595,203 个参数。
原始模型文件只读；初始、最佳和最后输出头分别保存，旧 8 例推理结果不覆盖。

## 数据与输入输出

`prepare_dataset.py` 在看到训练结果前锁定 seed=20261027 的 40 个完整配置组，每组 2 个相机。
两种固定场景族各 20 组，按 12/4/4 组划分：48 train、16 validation、16 test。
同组的几何、材质和光源组合不会跨 split；相机扇区也隔离。这是同场景族适应，不能声称新场景泛化。
金属茶壶变动粗糙度、旋转、墙色和灯光；Sponza 变动材质、灯光和相机，建筑几何保持固定。

每帧输入为官方转换生成的 `scene.h5`，监督为同转换网格的 256×256 Cycles 线性 HDR，2048 spp、seed 11、无降噪。
实际网格/法线和文件 SHA256 由 `audit_dataset.py` 检查。所有网格、H5、EXR、权重和特征缓存保存在 F 盘本地。
Sponza 资产不随仓库发布，复现该部分需自行提供许可允许的本地源资产。

`cache_features.py` 只计算冻结主干的四层 DPT 输入，保留实际 tensor dtype。
首次 train 帧验证缓存输出头与完整官方推理 HDR 输出数值一致。
缓存 test 输入允许，但此时不读取 test 参考像素、不计算 test 预测或评价。
目标为 `log10(1+max(HDR,0))`；负值数量与最小值记录于训练日志，预测逆变换为 `10**y-1`。

## 优化与停止

损失是 log HDR 的逐像素 RGB MSE；这是本教学微调选择，不冒称原论文完整训练损失。
AdamW，lr=1e-5，weight_decay=1e-4，梯度范数上限 1，AMP/GradScaler。
`cached_capacity.py` 在 train 输入上仅比较 microbatch 4 和 8，丢弃探测权重；有效 batch 等于 microbatch，累积为 1。
正式运行从原始预训练输出头重新开始，最大 100 epochs。
最佳 checkpoint 始终取最低 validation loss；patience 使用独立锚点，连续 12 epochs 未改善至少 0.2% 即停止。
到最大 epoch 但未 plateau 不称为收敛，也不开 test，先根据 validation 曲线决定是否继续。
这里的收敛只表示固定实验下的操作性验证平台期，不是数学最优保证。

每个 epoch 末以 eval 模式重测 train/val，曲线口径一致；更新中的 online loss 另存 steps.jsonl。
`initial.safetensors`、`best.safetensors`、`last.safetensors` 分离。输出头约占整个 base 模型参数的 5.16%。

## 顺序运行

在仓库根目录使用隔离环境 `F:/RenderFormer-work/venv/Scripts/python.exe`：

```powershell
python scripts/training/prepare_dataset.py
python scripts/training/lock_showcase.py
python scripts/training/render_dataset_batch.py
python scripts/training/audit_dataset.py
python scripts/training/cache_features.py
python scripts/training/cached_capacity.py
python scripts/training/train_head.py --microbatch 8 --output F:/RenderFormer-work/training-pilot/run-v1
python scripts/training/evaluate_final.py --run F:/RenderFormer-work/training-pilot/run-v1
```

batch 8 仅为命令示例，实际采用容量探测确认的值。prepare/lock 拒绝覆盖锁定 manifest；正式训练拒绝覆盖已有 run。
数据生成可恢复；Blender 每帧恢复 factory settings，避免对象/材质跨帧污染。
最终 test 一次逻辑评价固定验证选出的 checkpoint，全部 16 帧保留，不能根据结果再选模型。
同 checkpoint 和 manifest 的中断恢复允许补齐产物。所有显示图统一 EV0、Reinhard、sRGB，指标在同尺寸浮点域计算。

额外 12 帧 camera-only 展示在训练前锁定，仅在 test 后执行，不参与选模，也不冒充独立泛化测试。

## 本次实测结果与后处理

本次容量比较选用 microbatch 8、梯度累积 1，每 epoch 6 次更新。正式运行在第 25 epoch（150 steps）达到预定早停条件，最低验证损失位于第 13 epoch，随后才执行全部 16 帧留出评价。最大 100 epochs 并未耗尽。
整体测试 PSNR 从 14.99 dB 增至 17.67 dB，但金属茶壶 SSIM 从 0.835 降至 0.682；Sponza 图像仍明显模糊。历史 512px 回归中，康奈尔盒、金属茶壶及 Sponza 外部视角均有退步。这是受限适应及其代价，不是通用渲染质量提升。

以下后处理在最终 checkpoint 固定且 test 完成后顺序执行，不重选模型：

```powershell
python scripts/training/export_summary.py
python scripts/training/historical_regression.py
python scripts/training/render_dataset_batch.py F:/RenderFormer-work/training-pilot/showcase-v1
python scripts/training/predict_showcase.py
```

`run-v1/publication/` 下的三份 summary 分别记录 256px 留出测试、512px 历史回归、256px 离散多视角展示；三种用途不能混合计算泛化结论。原始 HDR、checkpoint、运行日志和输入资产保留本地，公开展示文件由发布步骤单独拷贝。
