# DF-Box

Android 图片标注测试 App（Kotlin + Material 3）。选一张图 → 端上跑 `yolo11n_dfv15` w8a16 量化模型 → 画 enemy / teammate 框 → 导出标注图。

构建全部在 GitHub Actions 上完成，本地不装 Android SDK。

## 这一版做了什么

- 模型换成 `exports/w8a16/yolo11n_dfv15_w8a16.tflite`（int8 权重 + fp16 激活，3.0 MB；输入输出仍为 float32）
- **NPU 优先**：默认 NNAPI delegate（`setUseNnapiCpu(false)`），创建失败或不支持时自动回落 XNNPACK CPU，状态栏显示实际生效的后端
- **后端测速**：每个后端各跑 12 轮纯推理，出 avg / p50 / p95 / fps / 首次耗时，可一键采用更快者
- **输入处理分辨率**：320 / 480 / 640 / 960（长边），影响解码后的缩放与预处理开销；模型输入固定 640×640
- **首次使用引导**：说明用法 + 引导去设置后端与分辨率，配置写进 SharedPreferences
- Material 3：MaterialToolbar + 菜单、MaterialCardView、Material Slider、MaterialAlertDialog
- APK 只打 `arm64-v8a`，体积从 49 MB 降到约 14 MB

## 代码结构

```
app/src/main/java/cn/dfbox/annotator/
├── MainActivity.kt      # 选图 / 推理 / 设置 / 测速 / 保存
├── Detector.kt          # Detection、ModelAssets、letterbox、后处理 + NMS
├── TfliteDetector.kt    # LiteRT 后端（NNAPI 优先，失败回落 CPU）
├── Benchmark.kt         # 双后端测速
├── Prefs.kt             # 设置项 + Backend 枚举 + 测速用测试图
├── BoxRenderer.kt       # 画框与标签
└── BoxOverlay.kt        # fitCenter 坐标换算的覆盖层
```

## 模型 IO（已在本机用 Python 实测核对）

| 项 | 值 |
| --- | --- |
| 文件 | `yolo11n_dfv15_w8a16.tflite`（ultralytics 8.4.204，imgsz 640） |
| 输入 | `serving_default_args_0` [1,3,640,640] float32，NCHW，/255 |
| 输出 | [1,6,8400] float32，cx,cy,w,h **已归一化到 0–1**，后接已过 sigmoid 的类别分 |
| 类别 | `{0: enemy, 1: teammate}` |

`exports/w8a16/yolo11n_dfv15_qnn.onnx` 没有打进 APK：它是 ONNX Runtime QNN EP 的 **EPContext** 预编译模型（整张图只有一个 `EPContext` 节点，元数据标记 QNN EP v2.51 / HTP），必须配 QNN runtime 才能执行；标准 `onnxruntime-android` 的 CPU/NNAPI EP 跑不了它。要上 HTP 得走 Qualcomm QNN SDK 自己编 runtime + 传 context binary，不在这一版范围。

## 构建

推 `main` 触发 [`.github/workflows/android.yml`](.github/workflows/android.yml)：

1. JDK 17 + Gradle 8.7（Android SDK 用 runner 自带的）
2. `gradle :app:assembleDebug`
3. APK 上传 artifact + 发布到 [Releases](https://github.com/JeanWenzelCN/DF-Box/releases)（tag `latest`）
4. 完整构建日志强推到 `ci-log` 分支（`build.log.txt`），失败也能直接拉日志

## 说明

测试用途，未做混淆、未做 EXIF 方向校正。NNAPI 是否真下沉到 NPU 取决于 SoC 驱动与算子覆盖，以「后端测速」实测为准。
