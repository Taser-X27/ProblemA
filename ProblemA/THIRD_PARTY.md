# 第三方来源

- `upstream/videomaev2/`：本工作区已固定的VideoMAEv2官方模型实现，保留`LICENSE`与源文件头。新增中文注释通过AST等价校验。
- `upstream/cnn/`：Noah Oksuz 的 CUHK-X 公开方案的模型和数据处理实现。原下载 URL、来源文件 SHA256 见 `source_manifest_original.json`，中文注释与原实现 AST 一致。
- `vendor/timm*`、`vendor/ultralytics*`、`vendor/thop*`及`vendor/cv2*`：复制当前实际使用的依赖版本，保留包内许可证与dist-info元数据。
- PyTorch、torchvision、NumPy、Pillow等基础依赖由现有`pt_qwen`环境提供，版本记录在`configs/environment.json`。

初始化权重来源与SHA256见`configs/assets.json`，源码校验见`upstream/sources.json`。训练与验证只从本项目相对路径加载。`vendor/`还包含matplotlib及其绘图依赖，保留原包的许可证和元数据。
