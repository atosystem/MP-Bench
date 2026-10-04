<h1 align="center">MP-Bench</h1>

<h3 align="center">Evaluating Voice Agents as a Multiparty Conversation Participant</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2609.13076"><img src="https://img.shields.io/badge/arXiv-2609.13076-b31b1b.svg" alt="arXiv:2609.13076"></a>
</p>

<p align="center">
  Yi-Jen Shih<sup>1,*</sup> &middot; Shih-Yun Shan Kuan<sup>2,*</sup> &middot; Guan-Ting Lin<sup>2,*</sup><br>
  Kai-Wei Chang<sup>3</sup> &middot; Siddhant Arora<sup>4</sup> &middot; Shu-wen Yang<sup>2</sup> &middot; Abdelrahman Mohamed<sup>5</sup><br>
  Shinji Watanabe<sup>4</sup> &middot; Hung-yi Lee<sup>2</sup> &middot; David Harwath<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> The University of Texas at Austin &nbsp; <sup>2</sup> National Taiwan University<br>
  <sup>3</sup> Massachusetts Institute of Technology &nbsp; <sup>4</sup> Carnegie Mellon University &nbsp; <sup>5</sup> Meta AI<br>
  <sup>*</sup> Equal contribution
</p>

# Overview

MP-Bench tests a conversational voice agent as an active participant in a four-person spoken conversation: three simulated humans and the agent. The set has **927 tasks** drawn from two scenarios.

- **Discussion.** A multiparty debate on one question. The clip ends either with a comprehension question about the conversation, or with an explicit invitation for the agent to settle who is right.
- **Turn-based Game.** A structured game (word chain, increasing number, or decreasing number). The agent is one team's follower and must speak only when the opposing leader has just played.

Each task is either an **understanding** task or a **behavior** task. Understanding (543 tasks, Discussion only) checks whether the agent followed who said what, including a speaker-name subset whose chance level is 33.3% for more intuitive comparison (three human speakers). Behavior checks whether the agent knows when to take a turn and, if it speaks, whether the reply fits: 128 Discussion tasks, where the agent is always addressed and expected to answer, and 256 Turn-based tasks, balanced between "speak" and "stay silent" so chance turn-taking is 50%.

Follow the protocol below: leave 30 seconds of silence after each input, transcribe the recording with Parakeet, then compute the Table 3 metrics.

MP-Bench is for evaluation only. Do not use the tasks or the audio to train a model.

## Metrics


| Split                     | Tasks | Scores                                                  |
| ------------------------- | ----- | ------------------------------------------------------- |
| Understanding: Discussion | 543   | Overall accuracy, speaker-name accuracy, latency        |
| Behavior: Discussion      | 128   | Turn-taking accuracy, appropriateness accuracy, latency |
| Behavior: Turn-based      | 256   | Turn-taking accuracy, appropriateness accuracy, latency |


Understanding accuracy and speaker-name accuracy judge the spoken answer. Turn-taking accuracy uses word timestamps: the agent counts as having spoken when any word it generated starts after the last input word. Appropriateness accuracy requires both that turn-taking decision and a fitting reply. On Discussion the reply has to be logically consistent with the debate, and on Turn-based Game it has to follow the game rule. Latency is the time from the end of the last input word to the start of the agent's first word, averaged over responses that are not silent.

## Repository


| Path                    | Role                                                      |
| ----------------------- | --------------------------------------------------------- |
| `benchmark/tasks.jsonl` | The 927 tasks and their labels                            |
| `mpbench_eval/`         | Turn-taking detection, LLM judges, and the Table 3 report |
| `prompts/`              | Judge prompts and output schemas                          |
| `scripts/transcribe.py` | Parakeet transcription of scenario wavs                   |
| `examples/`             | One transcript folder for each task family                |


Scoring uses the Python standard library (`requirements.txt`). Transcription needs a GPU plus NeMo, PyTorch, and soundfile.

## Pipeline

1. [Download the task audio](#1-data).
2. [Run the agent](#2-inference), with 30 seconds of silence after each input.
3. [Transcribe](#3-transcribe) the recordings.
4. [Score](#4-score) the transcripts.



### 1. Data

Task audio:

[https://drive.google.com/file/d/1SxMUFbhUH4FHl6nB-yxC4R9sBcpsU3yL/view?usp=drive_link](https://drive.google.com/file/d/1SxMUFbhUH4FHl6nB-yxC4R9sBcpsU3yL/view?usp=drive_link)

### 2. Inference

How the agent is served depends on the provider. For every task, append **30 seconds of silence** after the input audio before the model consumes it, and keep the session recording through that tail so the model has time to respond.

The run should leave one folder per task, named with the task id (`scenario_...`). That folder is often nested under the run directory. Find the scenario folders, then use their parent as the audio root for transcription:

```bash
find /path/to/run -type d -name 'scenario_*' | head
```

The parent of those folders is `--input_dir`. Two layouts from the same model, one already flat and one nested under the provider's output directory:

```
.../gpt_v1prompt_rerun/scenario_argument-diag_11-tier_2-task_0/

.../inference_result_elevenlabs_v1prompt/openai_output/scenario_argument-diag_11-tier_2-task_0/
```

Inside each scenario folder the three wavs share one timeline:


| File               | What it is                                                         |
| ------------------ | ------------------------------------------------------------------ |
| `all_stitched.wav` | Input audio, including the 30s tail                                |
| `response.wav`     | The model's audio. Check this file first when a task looks missing |
| `combined.wav`     | Input and model together. May be stereo, one channel each          |




### 3. Transcribe

`scripts/transcribe.py` runs `nvidia/parakeet-tdt-0.6b-v2` (NeMo Parakeet TDT), the ASR model used for the paper's word timestamps. Point `--input_dir` at the parent of the `scenario_*` folders. It writes `all_stitched.json`, `response.json`, and `combined.json` beside each other, with the same relative paths as the wavs.

Leave `--batch_size` at 1. A batch of 4 matched one-file decoding on a small check, but on the full GPT audio it changed some transcripts. To spread scenarios across GPUs, run one process per GPU with `--shard_index` and `--shard_count`.

```bash
python scripts/transcribe.py \
  --input_dir /path/to/openai_output \
  --output_dir /path/to/my_model_asr
```

Each JSON is Whisper-style:

```json
{"text": "...", "chunks": [{"text": "...", "timestamp": [0.0, 0.4]}]}
```

`all_stitched.json` is the input transcript and `response.json` is the model transcript. `combined.json` is the same timeline with both sides and is accepted by the scorer. See `examples/` for one folder of each task family. The directory you pass as `--output_dir` is `--result-dir` in the next step.

### 4. Score

`--result-dir` must contain all **927** `scenario_*` folders.

Understanding, discussion appropriateness, and turn-based appropriateness use `gpt-5.4`. Speaker name uses `gpt-5.2`. With the default sampling settings the CLI sends `temperature=0`, `top_p=1`, and `max_completion_tokens=4096`.

```bash
export OPENAI_API_KEY=...
python -m mpbench_eval detect --result-dir /path/to/my_model_asr
python -m mpbench_eval judge --result-dir /path/to/my_model_asr \
  --model gpt-5.4 --provider openai --workers 24 \
  --kinds understanding discussion turnbased
python -m mpbench_eval judge --result-dir /path/to/my_model_asr \
  --model gpt-5.2 --provider openai --workers 24 \
  --kinds speaker_name --resume
python -m mpbench_eval report --result-dir /path/to/my_model_asr --name MyModel
```

Outputs land in `<result-dir>/mpbench_eval_out/` (`RESULTS.md`, `RESULTS.json`).

If the transcripts are already in that layout, the same four commands are the whole run. `detect` reads the timestamps and decides turn-taking and latency. `judge` calls the LLM. `report` aggregates Table 3. `judge --dry-run` writes the prompts and does not call the API. `--resume` keeps judgments that already have a verdict.

`python -m mpbench_eval run` is the same three stages in one process, and it uses the CLI's default judge (`gpt-5`) for every kind. If an endpoint rejects `temperature`, pass `--no-sampling-params`. `--provider gemini` selects the Gemini API. If you already have a `spoke_after_input_results.tsv`, pass `--spoke-tsv` to `detect` or `run`.


## License

[CC BY-NC 4.0](LICENSE).

## Citation

```bibtex
@misc{shih2026mpbench,
      title={MP-Bench: Evaluating Voice Agents as a Multiparty Conversation Participant}, 
      author={Yi-Jen Shih and Shih-Yun Shan Kuan and Guan-Ting Lin and Kai-Wei Chang and Siddhant Arora and Shu-wen Yang and Abdelrahman Mohamed and Shinji Watanabe and Hung-yi Lee and David Harwath},
      year={2026},
      eprint={2609.13076},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2609.13076}, 
}
```

