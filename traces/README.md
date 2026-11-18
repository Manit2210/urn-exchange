# Traces

`urn-exchange-v1-traces.jsonl.gz` is attached to the v1 GitHub release and is not committed to the
repository. Download it into this folder.

- Size: 6,374,075 bytes (31.0 MB uncompressed), 78 lines (one match record per line)
- SHA-256: `c26f06bd26e54c521256007dad8f6d26cf50315707ca69f489f61532dfeb1765`
- Datasets: `v1` (72 matches, the main run) and `v1-xhigh` (6 matches, the side run at effort xhigh)
- License: CC BY 4.0 (see `LICENSE-DATA`)

Check and use it:

```bash
shasum -a 256 traces/urn-exchange-v1-traces.jsonl.gz
urn-exchange ratings --run traces/urn-exchange-v1-traces.jsonl.gz --dataset v1 --out /tmp/v1
```

The record format is described in `docs/harness.md`, section 5. Encrypted reasoning and signature blobs
were removed before publication; readable reasoning text and summaries are kept.

Your own runs export with `urn-exchange traces --run results/runs/<run_id> --out traces/<name>.jsonl.gz`.
