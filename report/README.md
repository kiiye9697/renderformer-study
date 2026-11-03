# 课程报告源码

当前稿件以官方模型推理为主要展示，辅以Cornell场景族训练和原生Sponza输入边界分析。数据快照为2026年11月3日，训练结果截至10000步，不含最终测试。正文分5节，含封面共6页。

- `report-content.md`：人工编辑的正文，指标位置使用占位符。
- `build_course_report.py`：生成入口，调用`build_cornell_report.py`。
- `report-draft.md`：生成后的可读正文，数值由本地实测JSON填入。
- 本地输出：`../../report/RenderFormer-课程报告.docx`；公开下载：`../docs/assets/course-report.docx`。
- 提交版可通过`--identity <本地JSON> --output <独立DOCX路径>`填写学院、专业、班级、姓名、学号；身份JSON与填写后的报告保存在站点仓库外，不进入公开下载。
- 生成依据与校验：`../../report/qa/cornell-stage/`。

生成器读取相邻实验目录和原模板，核对7个检查点的128张验证图、逐图汇总、模板哈希及保留部件。需要python-docx与lxml，使用Codex工作区依赖提供的Python运行。可从任意工作目录执行本目录的`build_course_report.py`。不连接服务器、不训练、不访问测试集、不发布网页。

重新生成会将排版校验状态设为待检查；交付前必须导出并检查所有页面，确认6—10页、黑体小四标题、宋体单倍行距和图片面积不超过1/6。Windows本机可用Word只读导出PDF，再用Poppler生成页面图；QA导出不是最终报告。

本目录原有`RenderFormer-期末报告.docx`属于历史适配实验，不能当作当前稿。旧草稿和生成脚本已存入`../../report/archive/retired-adaptation/`。当前报告单独命名，避免覆盖历史文件。

后续补齐实际续训批量与结束步数，冻结验证集选择，再执行最终测试。更新数据快照时同步修改正文、固定阶段图片及生成器的快照约束，不只替换表格数字。
