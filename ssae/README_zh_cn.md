[English](README.md)

# Structured Semantic Alignment Evaluation (SSAE)

SSAE 的原理是将复杂的 Prompt 拆解为一系列细粒度的问题，并要求 VLM（视觉语言模型）回答这些问题（（`是`或`否`），通过统计`是`的比例来评估模型生成视频的语义对齐度。我们按六种主要类型整理了评测用 Prompt 并放置在 `ssae_prompts` 目录下。

# 样例脚本简介

这里我们提供了一个样例脚本`gemini_video_ssae.py`，把视频上传到 Google Cloud Storage 并用 Gemini 自动分析，也可以自行选择其他VLM。

`gemini_video_ssae.py` 会从问题文件中读取每条数据的 `idx`，在本地对应目录`{source_folder}`按 `{idx}.mp4` 寻找对应视频，上传到 Google Cloud Storage（GCS），随后调用 Gemini 对视频进行分析并回答该条数据下的所有问题，最后将结果写入一个 JSONL 输出文件。

## 数据流

- **输入**：一个或多个问题文件（支持 `JSONL` 或 `JSON` 数组），以及本地视频目录
- **处理**：
  - 根据 `idx` 找本地视频：`{source_folder}/{idx}.mp4`
  - 上传到 GCS：`gs://{bucket}/{idx}.mp4`
  - 使用 Gemini 对视频回答多条问题
  - 每处理完一个 `idx` 后将结果写入到 `--output`
- **输出**：JSONL（每行一个结果对象，包含成功/跳过/失败状态）

## 环境准备

### 1. 安装依赖

脚本依赖 Google Cloud SDK以及一些python安装包：

```bash
curl https://sdk.cloud.google.com | bash
# 登录并授权完整 Scope，防止出现权限不足报错
gcloud auth application-default login
pip install --upgrade google-cloud-storage google-genai tqdm
```

### 2. 权限、认证说明与配置指南

该脚本使用 Application Default Credentials (ADC) 进行认证：

#### 1. 获取 Project ID (项目 ID)
Project ID 是 GCP 中所有资源的唯一标识符。

1. 登录 Google Cloud Console，点击页面上方Console。
2. 在弹出的对话框中，能看到项目列表。找到目标项目，右侧列出的字符串即为 ID（Project ID 通常是类似 my-awesome-project-123456 的字符串，请勿将其与项目名称（Project Name）混淆）。
   - *前提：需要目标 GCP Project 已开通 Vertex AI，并且当前身份具备调用 Gemini 的权限。*

#### 2. 获取 Bucket Name (存储桶名称)
Bucket 是 GCS 中存放数据的容器。

1. 在 Console 中进入 Cloud Storage -> Buckets。
2. 创建或选择一个存储桶，获取 Bucket Name。
   - *前提：需要目标 GCS bucket 对当前身份可写（拥有对象上传权限）。* 


## 数据与文件约定

### 1. 问题文件格式

`--questions` 可传多个文件，脚本会合并处理。
- 格式：JSONL （推荐）或 JSON。
- 必须字段：
  - **`idx`**：唯一标识，用于关联视频 `{idx}.mp4`。
  - **`questions`**：问题列表；每个元素至少包含 `question` 字段（字符串）。
  - **`category`**：可选；缺省为 `"unknown"`。


### 2. 本地视频组织方式

脚本固定按以下路径寻找视频（当前版本只找 `.mp4`）：

- 路径：`{source_folder}/{idx}.mp4`
- 异常处理：若本地视频不存在，会记录为 `status="skipped"` 并写入输出 JSONL。

## 使用方法（CLI）

```bash
# 单个 questions 文件
python3 gemini_video_ssae.py \
  --project YOUR_GCP_PROJECT \
  --bucket YOUR_BUCKET \
  --source_folder ./examples \
  --questions ssae_prompts/locomotion.jsonl \
  --output analysis_results.jsonl \
  --fps 5

# 多个 questions 文件合并处理
python3 gemini_video_ssae.py \
  --project YOUR_GCP_PROJECT \
  --bucket YOUR_BUCKET \
  --source_folder ./examples \
  --questions ssae_prompts/locomotion.jsonl ssae_prompts/fitness.jsonl \
  --output analysis_results.jsonl \
  --fps 5
```

### 参数说明

- **`--project`**：Google Cloud 项目 ID（必填
- **`--bucket`**：用于存放视频的 GCS Bucket 名称（必填）
- **`--source_folder`**：本地视频文件夹路径（必填）
- **`--questions`**：问题文件路径列表（必填，支持多个）
- **`--model`**：Gemini 模型 ID（默认：`gemini-3-pro-preview`）
- **`--output`**：输出 JSONL 文件路径（默认：`analysis_results.jsonl`）
- **`--max_workers`**：并行处理的线程数（默认：`1`）
- **`--fps`**：设置Gemini VideoMetadata参数（默认：`5`）


### 输出说明（JSONL）

输出文件为 JSONL，每行对象大致包含：

- 原始输入信息：`idx` / `category` / `original_questions`
- 处理状态信息：`status`：`success` / `skipped` / `error`
- 上传后的`gcs_uri`（成功时）： `gs://...`
- 成功时包含：`response`（已解析的 JSON 结构，包括`question`, `answer`, `confidence`以及`reason`）
- 失败时包含：`error`（错误信息字符串）
