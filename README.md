# DF-Box

Android 图片标注极简测试 App（Kotlin）。选一张图 → 端上跑 yolo11n_dfv15 → 把 enemy / teammate 的框和置信度画在图上。

所有构建都在 GitHub Actions 上完成，本地不装 Android SDK。

## 功能

- 图库选图（`ACTION_GET_CONTENT`），最长边超 2048 像素时自动降采样
- 双后端可切换：`ONNX Runtime`（onnxruntime-android 1.19.2）/ `TFLite`（tensorflow-lite 2.16.1）
- letterbox 到 640×640，输入布局自动识别（NHWC / NCHW），坐标单位自动识别（像素 / 归一化）
- 置信度阈值滑杆（0.05–0.95，默认 0.25），类别内 NMS（IoU 0.45）
- 状态栏显示：目标数、分类计数、预处理 / 推理 / 后处理耗时、实际输入布局
- 「保存标注」把画好框的图写到 `Android/data/cn.dfbox.annotator/files/Pictures/DFBox/`

## 模型

`app/src/main/assets/` 下内置两个导出，均来自 `dfv15_v2` best.pt（ultralytics 8.4.203，imgsz 640）：

| 文件 | 输入 | 输出 |
| --- | --- | --- |
| `yolo11n_dfv15.onnx` | `images` [1,640,640,3] float32 NHWC /255 | `output0` [1,6,8400]，cx,cy,w,h 为像素坐标 |
| `yolo11n_dfv15.tflite` | `serving_default_args_0` [1,3,640,640] float32 NCHW /255 | [1,6,8400]，cx,cy,w,h 已归一化到 0–1 |

类别：`{0: enemy, 1: teammate}`。两种导出都是 xywh + 已过 sigmoid 的类别分，所以后处理只做单位还原、置信度过滤和 NMS。

## 构建

推送到 `main` 触发 [`.github/workflows/android.yml`](.github/workflows/android.yml)：

1. JDK 17 + Android SDK + Gradle 8.7
2. `gradle :app:assembleDebug`
3. APK 上传为 artifact，同时发布到 [Releases](https://github.com/JeanWenzelCN/DF-Box/releases)（tag `latest`）
4. 完整构建日志强制推送到 `ci-log` 分支（`build.log.txt`），构建失败也能直接看日志

APK 只打 `arm64-v8a`（ORT 的 `libonnxruntime.so` 单个 ABI 就 17 MB）。

## 目录

```
app/src/main/java/cn/dfbox/annotator/
├── MainActivity.kt      # 选图 / 推理协程 / 保存
├── Detector.kt          # Detection、letterbox、后处理 + NMS、后端工厂
├── OrtDetector.kt       # ONNX Runtime 后端
├── TfliteDetector.kt    # TFLite 后端
├── BoxRenderer.kt       # 画框与标签
└── BoxOverlay.kt        # fitCenter 坐标换算的覆盖层
```

## 说明

测试用途，未做混淆、未做 EXIF 方向校正；日志与耗时以真机实测为准。
