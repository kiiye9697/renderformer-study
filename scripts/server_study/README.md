# Bunny与简化Sponza的256像素适配

固定官方代码eec6a9b8、base权重revision 8aa00095。训练、验证与材质留出对照分别为18/6/6帧，训练/验证粗糙度为0.15、0.5、0.9 / 0.3；本轮对照为0.65。几何与相近视角共享，这些配置用于迭代检查，不用于声称未知场景泛化。

设置`RENDERFORMER_WORKSPACE`，该目录包含`repos/`、`datasets/`、`models/`、`runs/`、`results/`与`archives/`。官方代码放在`repos/renderformer-eec6a9b8d592303def45e458dd4ebd09d6572abc`，固定权重放在`models/renderformer-v1-base-8aa00095`。将官方代码加入`PYTHONPATH`，设置`ATTN_IMPL=sdpa`与FreeImage库路径。

依次执行：

1. `prepare_training.py`：准备Bunny与经许可取得的Sponza源资产，构造原始场景配置。Sponza资产打包位置为`archives/sponza-gltf.zip`，不随仓库分发。
2. `prepare_detail256.py`：生成256像素、2048 spp参考。
3. `fix_detail_inputs.py`：转换0.65对照配置的独立H5，确保输入材质与新参考完全一致。
4. `train_detail256.py`：冻结Transformer，训练DPT100轮，目标为对数HDR L1加0.15倍归一化空间梯度L1。
5. `train_viewfull256.py`：从上述最佳DPT出发，冻结场景Transformer，适配视角Transformer与DPT60个迭代轮；每轮每配置轮换1个视图，共360次更新，等于20次完整视图覆盖。使用激活检查点和逐视图更新，可训练参数81,560,713。验证最低目标出现在第59轮。
6. `export_viewfull256.py`：导出原始预训练与最终模型的同条件PNG和指标，不导出权重、缓存、资产或原始HDR。

执行场景转换与参考脚本时，工作目录应为固定官方仓库。预适配学习率5e-6，联合阶段视角网络1e-5、DPT 2e-6。原128像素结果与本轮不同，不合并分数。Sponza归一化边缘误差下降25.3%，细柱和梁架仍有明显模糊。
