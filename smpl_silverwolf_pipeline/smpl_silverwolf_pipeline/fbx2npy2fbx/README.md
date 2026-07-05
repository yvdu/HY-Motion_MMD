# fbx2npy2fbx

这个文件夹整理了 FBX -> NPY，以及 NPY/NPZ -> FBX 的来回转换脚本。Python 代码已从原项目中原样复制，运行时不依赖本文件夹外的项目代码。

## 目录结构

```text
fbx2npy2fbx/
  ascii_fbx_smplx_io.py
  batch_fbx_to_smplx_npy.py
  fbx_to_smplx_npz.py
  fbx_to_npy/
    run.py
    config.yaml
  npy_to_fbx/
    run.py
    config.yaml
  templates/
    body_check_001__A296.fbx
```

## Python 依赖

需要 Python 环境中安装：

```bash
pip install numpy pyyaml
```

当前脚本本身不调用 `pyfbsdk`，不需要把其它项目 Python 文件放到 `PYTHONPATH`。生成的 FBX 可以再用 MotionBuilder 打开或处理。

## FBX 转 NPY

1. 把待转换的 `.fbx` 放到 `input_fbx/`，或修改 `fbx_to_npy/config.yaml` 里的 `input_path`。
2. 在本目录运行：

```bash
python fbx_to_npy/run.py
```

默认输出到 `output_npy/`。脚本会递归处理子目录，并保留相对子目录结构。

## NPY/NPZ 转 FBX

1. 确认 `npy_to_fbx/config.yaml` 中：
   - `input_path` 指向 `.npy` 或 `.npz` 输入目录，默认是 `./output_npy`
   - `output_dir` 指向 FBX 输出目录，默认是 `./output_fbx`
   - `template_fbx` 指向包内模板，默认是 `./templates/body_check_001__A296.fbx`
2. 在本目录运行：

```bash
python npy_to_fbx/run.py
```

默认输出 ASCII FBX 到 `output_fbx/`。

## 模板说明

`npy_to_fbx` 会把 NPY/NPZ 的动画曲线写回模板 FBX。模板需要和输入 motion 的骨架、关节命名、动画帧数匹配；如果帧数不一致，脚本会报错：

```text
模板帧数 ... 与 npy 帧数 ... 不一致
```

如果每个 motion 都有对应模板，可以把 `template_dir` 配成模板目录。脚本会优先按输入相对路径查找同名 `.fbx` 模板，例如 `a/b/motion.npy` 会匹配 `template_dir/a/b/motion.fbx`。

## 配置项

- `recursive`: 是否递归处理子文件夹。
- `preserve_subdirs`: 输出时是否保留输入目录的相对子目录结构。
- `overwrite`: 输出已存在时是否覆盖。
- `fps_override`: FBX -> NPY 时覆盖帧率；`null` 表示使用 FBX 内部帧率。
- `workers`: 并行进程数。`fbx_to_npy` 支持 `auto`，`npy_to_fbx` 使用整数。

## 注意事项

- 请从 `fbx2npy2fbx` 目录下运行命令，这样默认相对路径会正确解析。
- `fbx_to_npy` 支持二进制 FBX 和 ASCII FBX。
- `npy_to_fbx` 输出的是 ASCII FBX。
- `.npy` 格式是一个 dict object array，包含 `poses`、`trans`、`betas`、`gender`、`mocap_framerate`。
