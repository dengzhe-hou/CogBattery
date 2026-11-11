# Submissions

Each submission is two files with the same name.

- `<id>.cogbattery.jsonl.gz` is the packed run from `python -m scripts.release.pack_submission pack`.
- `<id>.json` gives the model's family and its served parameters in billions, for example

```json
{"family": "qwen", "params_b": 7.6, "notes": "Q4_K_M from the Ollama library"}
```

Use a short id such as `qwen2.5-7b-ollama`. The model name and the runner backend are read from the pack.

Open a pull request that adds both files. A check rebuilds the materials, replays every request in the pack, scores the replies with the paper's scorers, and shows your key measures in the check summary. After the pull request is merged, the row appears in the community table on the [project page](https://dengzhe-hou.github.io/CogBattery/#community). To update a submission, replace both files under the same id.

[REPRODUCING.md](../REPRODUCING.md) describes how to run your model and pack the run.
