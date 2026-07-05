#!/usr/bin/env python3
import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.cloud import storage
from google.genai import types
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Gemini analysis prompt template
ANALYSIS_PROMPT = """
You are an expert AI assistant specializing in video-based human motion and action analysis. Your task is to meticulously analyze the provided video and answer several questions about a person's motion.

Your analysis MUST be based solely on the visual evidence within the video. Do not make assumptions or infer information that is not directly visible.

Each question must be answered with a simple "yes" or "no". Your response must be delivered strictly in the JSON format specified below, with no additional text preceding or following the JSON object.

## JSON Response Format:
[
    {{
        "question": "The original question from the user",
        "answer": "yes" or "no",
        "confidence": A number between 0.0 and 1.0 representing your certainty in the answer. 1.0 means you are absolutely certain based on clear visual evidence. 0.0 means you have no confidence.,
        "reason": A concise but specific explanation for your answer. **Crucially, you must reference visual cues from the video.** For example, mention specific body parts, their positions, direction of movement, or timestamps if possible (e.g., "At 0:05, the person's left knee bends beyond a 90-degree angle.").
    }},
...
]

## Questions
{questions}
"""


class VideoUploader:
    """Upload video to GCS"""

    def __init__(self, project_id: str, bucket_name: str):
        """
        Initialize the uploader

        Args:
            bucket_name: GCS bucket name
        """
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)

    def upload_video(self, local_path: str, remote_name: Optional[str] = None, skip_if_exists: bool = True) -> str:
        """
        Upload a single video file to GCS

        Args:
            local_path: Local file path
            remote_name: File name in GCS, if None then use the local file name
            skip_if_exists: Skip upload if the file already exists in GCS

        Returns:
            GCS URI (e.g., gs://bucket_name/filename)
        """
        filename = remote_name or os.path.basename(local_path)
        blob = self.bucket.blob(filename)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")

        if skip_if_exists and blob.exists():
            logging.info(f"Skipping upload: gs://{self.bucket_name}/{filename} already exists.")
        else:
            logging.info(f"Uploading: {local_path} -> gs://{self.bucket_name}/{filename}")
            blob.upload_from_filename(local_path)

        return f"gs://{self.bucket_name}/{filename}"


class GeminiVideoAnalyzer:
    """Use Gemini API to analyze video"""

    def __init__(self, project_id: str, model_id: str = "gemini-3-pro-preview", fps: int = 5):
        """
        Initialize the Gemini analyzer

        Args:
            project_id: Google Cloud project ID
            model_id: Model ID
        """
        assert fps > 0, "FPS must be greater than 0, got: {fps}"
        self.project_id = project_id
        self.model_id = model_id
        self.client = genai.Client(vertexai=True, project=project_id)
        self.video_metadata = types.VideoMetadata(fps=fps)

    def _load_questions(self, questions_files: List[str]) -> List[Dict]:
        """
        Load all question data

        Args:
            questions_files: List of question file paths

        Returns:
            List of question items
        """
        ssae_questions = []

        for questions_file in questions_files:
            if not os.path.exists(questions_file):
                logging.warning(f"File not found, skipping: {questions_file}")
                continue

            logging.info(f"Loading: {questions_file}")
            file_items = []

            # Check file format: JSONL or JSON
            with open(questions_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                f.seek(0)

                # If the first line is a valid JSON object, read as JSONL format
                try:
                    json.loads(first_line)
                    # JSONL format: one JSON object per line
                    for line in f:
                        line = line.strip()
                        if line:
                            file_items.append(json.loads(line))
                except:
                    # JSON format: the entire file is a JSON array
                    f.seek(0)
                    data = json.load(f)
                    if isinstance(data, list):
                        file_items.extend(data)
                    else:
                        file_items.append(data)

            ssae_questions.extend(file_items)
            logging.info(f"  ✓ Loaded {len(file_items)} data")

        logging.info(f"\nTotal loaded {len(ssae_questions)} questions for videos")
        return ssae_questions

    def _analyze_single_item(self, item: Dict, video_folder: str, uploader: VideoUploader, output_file: str) -> Dict:
        """
        Worker function for threading: processes one video index.

        Args:
            item: A dictionary containing the video index and questions
            video_folder: The folder containing the videos
            uploader: The uploader instance
            output_file: The file to write the results to

        Returns:
            A dictionary containing the result
        """
        idx = item.get("idx")
        questions = [q["question"] for q in item["questions"]]
        category = item.get("category", "unknown")

        # Match multiple file extensions
        local_video_path = None
        path = os.path.join(video_folder, f"{idx}.mp4")
        if os.path.exists(path):
            local_video_path = path

        if not local_video_path:
            return {"idx": idx, "status": "skipped", "error": "File not found"}

        try:
            # Upload
            gcs_uri = uploader.upload_video(local_video_path, f"{idx}.mp4")

            # Analyze
            video_part = types.Part(file_data=types.FileData(file_uri=gcs_uri, mime_type="video/mp4"), video_metadata=self.video_metadata)
            prompt_text = ANALYSIS_PROMPT.format(questions="\n".join(questions))

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[video_part, prompt_text],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )

            # Safe handling of JSON parsing failure
            try:
                res_json = json.loads(response.text)
            except json.JSONDecodeError:
                res_json = {"raw_response": response.text, "error": "Invalid JSON from model"}

            result = {
                "idx": idx,
                "category": category,
                "status": "success",
                "gcs_uri": gcs_uri,
                "response": res_json,
                "original_questions": item["questions"],
            }
        except Exception as e:
            logging.error(f"Error processing {idx}: {e}")
            result = {"idx": idx, "status": "error", "error": str(e)}

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        return result

    def analyze_ssae_dataset_parallel(
        self,
        questions_files: List[str],
        video_folder: str,
        uploader: VideoUploader,
        output_file: str,
        max_workers: int = 4,
    ):
        """Analyze SSAE dataset

        Args:
            questions_files: List of question file paths (JSONL format)
            video_folder: Local video folder path
            uploader: VideoUploader instance, for uploading videos
            output_file: Output result file path (JSONL)
            max_workers: The number of workers to use
        """
        # Load data
        all_items = self._load_questions(questions_files)
        # Check if the idx has been processed
        processed_idxs = set()
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    try:
                        processed_idxs.add(json.loads(line).get("idx"))
                    except:
                        continue
        # check if the idx has been processed
        items_to_run = [it for it in tqdm(all_items) if it.get("idx") not in processed_idxs]
        total_tasks = len(items_to_run)
        logging.info(f"Progress: {len(processed_idxs)} finished, {total_tasks} remaining.")

        # Analyze
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._analyze_single_item, it, video_folder, uploader, output_file): it.get("idx")
                for it in items_to_run
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    data = future.result()
                    logging.info(f"Finished {idx}: Status {data['status']}")
                except Exception as e:
                    logging.error(f"Worker crashed for {idx}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Video upload and analysis tool - upload and analyze videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--source_folder", required=True, help="Local video folder path")
    parser.add_argument("--questions", nargs="+", required=True, help="Question file paths")
    parser.add_argument("--project", required=True, help="Google Cloud project ID")
    parser.add_argument(
        "--model", default="gemini-3-pro-preview", help="Gemini model ID (default: gemini-3-pro-preview)"
    )
    parser.add_argument(
        "--output",
        default="analysis_results.jsonl",
        help="Analysis result output file path (default: analysis_results.jsonl)",
    )
    parser.add_argument("--max_workers", type=int, default=1, help="The number of workers to use (default: 1)")
    parser.add_argument("--fps", type=int, default=5, help="The frame rate of the video (default: 5), must be greater than 0")
    args = parser.parse_args()

    try:
        # Initialize uploader and analyzer
        logging.info("=" * 80)
        logging.info("Video upload and analysis process")
        logging.info("=" * 80)
        logging.info(f"Video folder: {args.source_folder}")
        logging.info(f"GCS Bucket: {args.bucket}")
        logging.info(f"Will load questions from the following {len(args.questions)} files:")
        for qf in args.questions:
            logging.info(f"  - {qf}")
        logging.info(f"Output file: {args.output}")

        uploader = VideoUploader(args.project, args.bucket)
        analyzer = GeminiVideoAnalyzer(args.project, args.model, args.fps)

        # Upload and analyze videos
        logging.info("\n" + "=" * 80)
        logging.info("Start processing videos")
        logging.info("=" * 80)

        analyzer.analyze_ssae_dataset_parallel(
            questions_files=args.questions,
            video_folder=args.source_folder,
            uploader=uploader,
            output_file=args.output,
            max_workers=args.max_workers,
        )

        logging.info("\n" + "=" * 80)
        logging.info("✓ All tasks completed")
        logging.info("=" * 80)

    except Exception as e:
        logging.error(f"Execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
