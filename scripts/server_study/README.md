# Bunny与简化Sponza实验

本轮使用RTX 2080 Ti、PyTorch 2.5.1、官方代码eec6a9b8与base权重revision 8aa00095。18/6/6帧分别用于训练、验证和测试。128像素、30轮，仅更新FP32 DPT解码器。

设置环境变量`RENDERFORMER_WORKSPACE`为独立实验目录。该目录包含`repos/`、`datasets/`、`models/`、`runs/`、`results/`与`archives/`。官方代码放在`repos/renderformer-eec6a9b8d592303def45e458dd4ebd09d6572abc`，固定权重放在`models/renderformer-v1-base-8aa00095`。

运行顺序：

1. 按官方依赖安装PyTorch、场景转换和Blender 4.0组件，准备FreeImage HDR读写库。
2. 自行取得Khronos glTF Sample Assets的Sponza模型与配套资源，保留其许可说明，打包为`archives/sponza-gltf.zip`。源资产不随仓库发布。
3. 执行`prepare_training.py`。该脚本产生固定的材质划分和简化后的共享几何。
4. 按生成的`datasets/training/manifest.json`逐配置运行官方`scene_processor/convert_scene.py`和`scene_processor/to_blend.py`，参考设置为128像素、512 spp、`--save_img`，输出到manifest指定位置。
5. 将官方代码加入`PYTHONPATH`，设置`ATTN_IMPL=sdpa`，执行`train_decoder_fp32.py`。首次运行缓存冻结主干特征；随后在FP32中更新DPT，按验证损失选择权重。
6. 使用`prepare_paper_badcases.py`生成相机与光源对照。参考设置为512像素、2048 spp；模型使用原始预训练权重。9次单光源预测在线性HDR中相加，再统一显示变换。

`export_server_results.py`用于现有结果目录的最小展示导出。它只生成PNG与JSON，不上传模型、资产、训练缓存或原始HDR。

报告中的微调结果限定为共享几何、相近视角的材质插值。30轮训练仍未出现稳定验证平台。论文边界实验的9光源分解结果未改善直接预测，原样保留。
