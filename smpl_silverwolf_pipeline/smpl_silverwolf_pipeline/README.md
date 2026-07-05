# SMPL ↔ 银狼 重定向 / MMD 工具包

这是一个聚焦的小工具包，覆盖三件事：

1. **SMPL 动作格式互转**：`.npy/.npz` ↔ SMPL FBX（`fbx2npy2fbx/`）。
2. **SMPL ↔ 银狼(silver_wolf) 互相重定向**（MotionBuilder / mobupy，`retarget/`）。
3. **MMD 相关脚本**：PMX → FBX、FBX → VMD（Blender + mmd_tools，`mmd/`）。

完整链路（SMPL 动作 → 在 MMD 里播放银狼）：

```
SMPL .npy ──(fbx2npy2fbx)──▶ SMPL .fbx ──(retarget)──▶ 银狼 .fbx(带动作)
                                                          │
                              ┌──(fbx_ascii_to_binary)────┘
                              ▼
                         银狼 二进制 .fbx ──(fbx_to_vmd_custom)──▶ 银狼 .vmd
                                                                     │
                              银狼 .pmx(mmd/model/) + .vmd ──────────┘──▶ 在 MMD 里播放
```

反向（银狼动作 → SMPL .npy）同理：`retarget`（银狼→SMPL）→ `fbx2npy2fbx`（fbx→npy）。

---

## 🚀 一体化 pipeline（run_pipeline.py）

`run_pipeline.py` 把 **HY-Motion 文本生成** 和本工具包的 **npy→FBX→重定向→VMD** 串成一条流水线，
通过子进程分别调用各阶段脚本（普通 Python / mobupy / Blender 各自独立环境，互不污染）。

阶段划分：

| 阶段 | 输入 → 输出 | 运行环境 | 底层脚本 |
|:----:|-------------|----------|----------|
| 0 | 文本 → SMPL `.npz` | Python (+GPU) | `local_infer.py`（HY-Motion） |
| 1 | `.npz/.npy` → SMPL ASCII FBX | Python | `fbx2npy2fbx/npy_to_fbx/run_reframe.py` |
| 2 | SMPL FBX → 银狼 ASCII FBX | mobupy | `retarget/retarget.py` |
| 3 | 银狼 ASCII FBX → 二进制 FBX | mobupy | `mmd/fbx_ascii_to_binary.py` |
| 4 | 银狼二进制 FBX → `.vmd` | Blender + mmd_tools | `mmd/fbx_to_vmd_custom.py` |

> 阶段 1 用的是 `run_reframe.py`（**支持任意帧数**），不再要求模板帧数与动作帧数一致，
> 因此可直接吃 HY-Motion 生成的变长 `.npz`。

### 配置

编辑 `pipeline_config.yaml`，至少改三处可执行文件路径：

```yaml
executables:
  python: "python"                                                  # 跑 HY-Motion / 阶段1
  mobupy: "C:/Program Files/Autodesk/MotionBuilder 2027/bin/x64/mobupy.exe"
  blender: "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"
```

各阶段产物落在 `work_dir` 下的子目录：`00_npz / 01_smpl_fbx / 02_silverwolf_fbx / 03_silverwolf_bin / 04_vmd`。
配置里的相对路径都相对**仓库根目录**（`HY-Motion-1.0-master/`）解析。

### 运行

```powershell
cd smpl_silverwolf_pipeline/smpl_silverwolf_pipeline

# A) 已有 SMPL 动作 .npz/.npy：放进 work_dir/00_npz，跑 1→4
python run_pipeline.py --stages 1,2,3,4

# B) 从文本开始（先把 pipeline_config.yaml 里 hymotion.enabled 设为 true，并配好 model_path）
python run_pipeline.py --stages 0,1,2,3,4
# 或简写
python run_pipeline.py --stages all

# C) 只跑单个阶段（调试用）
python run_pipeline.py --stages 2
```

任一阶段子进程返回非 0 即中断并报错，方便定位是哪一步失败。
跑完后在 MMD 里加载 `mmd/model/silver_wolf_lv999.pmx` + `work_dir/04_vmd/*.vmd` 即可播放。

---

## 目录结构

```
smpl_silverwolf_pipeline/
├── README.md
├── requirements.txt
├── run_pipeline.py               # ★ 一体化编排器（HY-Motion → 银狼 → VMD）
├── pipeline_config.yaml          #   一体化 pipeline 配置（可执行文件 / 目录 / 各阶段参数）
│
├── fbx2npy2fbx/                  # 1) SMPL 动作格式互转（自带 README，纯 Python，自包含）
│   ├── README.md
│   ├── fbx_to_smplx_npz.py       #   FBX(二进制) → SMPL NPZ 解析
│   ├── ascii_fbx_smplx_io.py     #   ASCII FBX ↔ npy
│   ├── batch_fbx_to_smplx_npy.py #   批量 FBX → npy
│   ├── fbx_to_npy/  (run.py, config.yaml)   # FBX → NPY 入口
│   ├── npy_to_fbx/  (run.py, run_reframe.py, config.yaml)  # NPY/NPZ → FBX 入口
│   └── templates/body_check_001__A296.fbx   # 写回 FBX 用的 SMPL 模板
│
├── retarget/                     # 2) SMPL ↔ 银狼 互相重定向（mobupy）
│   ├── mobu_retareting.py        #   ★ 重定向引擎（HIK 角色化 + 烘焙 + 导出 + 朝向后处理）
│   ├── retarget.py               #   ★ 通用驱动：--source / --target 任意方向
│   ├── retarget_silver_wolf.py   #   便捷驱动：SMPL → 银狼
│   ├── configs/characters_cfg.json          # 角色配置（SMPLX-lh-neutral + silver_wolf）
│   └── data/templates/
│       ├── SMPLX-lh-neutral/std.fbx         # SMPL 的 HIK 模板
│       └── silver_wolf/std.fbx              # 银狼的 HIK 模板（已修 T-Pose）
│
└── mmd/                          # 3) MMD 模型/动作脚本（Blender + mmd_tools）
    ├── pmx_to_fbx.py             #   PMX → FBX（把 MMD 模型转成可重定向的 FBX）
    ├── make_silver_wolf_template.py  # 给银狼 FBX 做 HIK 角色化，生成 std.fbx 模板（mobupy）
    ├── fbx_ascii_to_binary.py    #   ASCII FBX → 二进制 FBX（Blender 不能读 ASCII FBX）（mobupy）
    ├── fbx_to_vmd_custom.py      #   ★ FBX → VMD 高保真导出（含尺度/接地修正）
    ├── verify_standing.py        #   校验：VMD 驱动后模型是否正常站立、踩地
    ├── verify_vmd.py             #   校验：VMD 与源 FBX 逐帧对照
    ├── mmd_tools.zip             #   开源 Blender 插件（用于读写 PMX/VMD）
    └── model/                    #   银狼模型与贴图（silver_wolf_lv999.pmx/.fbx + 纹理）
```

★ = 核心脚本。

---

## 环境准备

| 用途 | 运行环境 |
|------|----------|
| `fbx2npy2fbx`（npy↔fbx） | 普通 Python 3：`pip install -r requirements.txt`（numpy / PyYAML） |
| `retarget`（重定向） | **Autodesk MotionBuilder** 的 `mobupy.exe` |
| `mmd`（PMX→FBX、FBX→VMD） | **Blender** + `mmd_tools` 插件 |
| `mmd/fbx_ascii_to_binary.py`、`make_silver_wolf_template.py` | **MotionBuilder** 的 `mobupy.exe` |

安装 Blender 插件：Blender → Edit → Preferences → Add-ons → Install，选 `mmd/mmd_tools.zip` 并启用。
（脚本里用的模块名是 `bl_ext.user_default.mmd_tools`。）

下文命令里的可执行文件请替换成你机器上的实际路径，例如：

- `mobupy = "C:\Program Files\Autodesk\MotionBuilder 2027\bin\x64\mobupy.exe"`
- `blender = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"`

---

## 用法

### 1. SMPL 动作格式互转（fbx2npy2fbx）

该子目录完全自包含，详见 `fbx2npy2fbx/README.md`。常用：

```bash
cd fbx2npy2fbx
pip install numpy pyyaml

# NPY/NPZ → SMPL FBX（写回模板，输出 ASCII FBX）
python npy_to_fbx/run.py        # 配置见 npy_to_fbx/config.yaml

# SMPL FBX → NPY
python fbx_to_npy/run.py        # 配置见 fbx_to_npy/config.yaml
```

`.npy` 是一个 dict（object array），字段：`poses` / `trans` / `betas` / `gender` / `mocap_framerate`。

### 2. SMPL ↔ 银狼 互相重定向（retarget）

用 `mobupy` 运行通用驱动 `retarget.py`，`--input` 可传单个 FBX 或目录：

```powershell
# SMPL → 银狼
& $mobupy retarget\retarget.py --source SMPLX-lh-neutral --target silver_wolf `
    --input  <SMPL动作.fbx 或 目录> --output <输出目录>

# 银狼 → SMPL（反向）
& $mobupy retarget\retarget.py --source silver_wolf --target SMPLX-lh-neutral `
    --input  <银狼动作.fbx 或 目录> --output <输出目录>
```

说明：
- 角色定义在 `retarget/configs/characters_cfg.json`，模板在 `retarget/data/templates/<角色>/std.fbx`。
- `skel_templates_root` 已写成相对路径 `data/templates`，并在 `mobu_retareting.py` 里相对脚本目录自动解析，整包可任意位置运行。
- 银狼项带 `post_fix`：重定向后把根骨 `全ての親` 的朝向静态修正为 `[0,0,0]`，避免面朝地面。
- 也可用便捷脚本 `retarget_silver_wolf.py <源FBX> <输出目录>`（固定 SMPL → 银狼）。

### 3. MMD：PMX → FBX（接入新角色时）

把 MMD 模型转成可重定向的 FBX，再做 HIK 角色化生成 `std.fbx` 模板：

```powershell
# PMX → FBX（Blender；首次可传 mmd_tools.zip 让脚本自行安装插件）
& $blender --background --factory-startup --python mmd\pmx_to_fbx.py -- `
    mmd\model\silver_wolf_lv999.pmx  <输出.fbx>  mmd\mmd_tools.zip

# FBX → HIK 角色化模板 std.fbx（mobupy）
& $mobupy mmd\make_silver_wolf_template.py  <上一步.fbx>  retarget\data\templates\silver_wolf\std.fbx
```

### 4. MMD：FBX → VMD（把重定向结果带回 MMD）

> 关于「FBX 转 PMX」：**模型（PMX）不需要再转**——MMD 播放 = 现成的 PMX 模型 + 导出的 VMD 动作。
> 所以这里只需把重定向得到的银狼 FBX 动作转成 VMD，配合 `mmd/model/silver_wolf_lv999.pmx` 即可。

```powershell
# (a) 若重定向输出是 ASCII FBX，先转二进制（Blender 不能读 ASCII FBX）
& $mobupy mmd\fbx_ascii_to_binary.py  <重定向ascii.fbx>  <bin.fbx>

# (b) 二进制 FBX → VMD（Blender + mmd_tools）
& $blender --background --python mmd\fbx_to_vmd_custom.py -- `
    mmd\model\silver_wolf_lv999.pmx  <bin.fbx>  <输出.vmd>
```

然后在 MMD 里加载 `mmd\model\silver_wolf_lv999.pmx`，再载入导出的 `.vmd` 即可播放。

校验（可选）：

```powershell
# 看模型是否正常站立、踩地（髋部在 rest 高度、双脚在地面、比例不变）
& $blender --background --python mmd\verify_standing.py -- `
    mmd\model\silver_wolf_lv999.pmx  <输出.vmd>
```

---

## VMD 导出的关键修正（fbx_to_vmd_custom.py）

直接用 mmd_tools 往复转 VMD 会因其转换器的「反射矩阵」缺陷而旋转往返不可逆；本脚本做了三处处理保证保真与正确站立：

1. **解析逆变换**：`vmd_rot = q.conj() @ local_rot @ q`，与导入精确互逆；FK 骨只传旋转（MMD 旋转专用语义）。
2. **去除根骨伪缩放**：重定向 FBX 的根骨 `全ての親` 常被烘入 ~0.01 的伪缩放（cm/m 单位伪影），只影响动作姿态、导致整体塌缩到 ~1/100（旋转对、但平移塌陷下沉）。导出前把它的 pose scale 复位为 `(1,1,1)`。
3. **接地（grounding）**：SMPL 源与银狼绑定姿态的腿长/地面参考不同，整段动作可能整体悬空或下沉。预扫所有帧取最低脚踝，对齐到 rest 脚踝高度（MMD 地面），再整体竖直平移，使 `センター` 与 `足ＩＫ` 一起下移，双脚自然踩地。

根位移放在可移动的 `センター`（让 `腰` 落到源 `腰` 的绝对世界位置），足部由 `足ＩＫ/つま先ＩＫ` 跟随脚踝/脚尖。
