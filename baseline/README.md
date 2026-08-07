# Local checkpoint manifest

The checkpoint binaries below are intentionally excluded from regular Git:
each exceeds GitHub's 100 MiB blob limit, and the repository policy keeps
model weights out of source history.  Evaluation artifacts record the exact
checkpoint path, step, and hash used.  This manifest makes the local inputs
auditable without committing multi-gigabyte payloads.

| Local path | Bytes | SHA-256 |
|---|---:|---|
| `s_baseline.ckpt` | 1,110,490,728 | `38ad1af5aba40ce1086a345d652a004ba627d5eff6e6a522fb66e0adc6ae2147` |
| `MDLM_best.pt` | 818,049,679 | `7b5e5cfc69094e7cd17f45aa8b7f2ae911f42bd55563d361814e6efde319ba7d` |
| `BD3LM_best.pt` | 818,051,271 | `c6d4ac8e5b265e565bed7819463a57be3058742eadb2696413cf288c0cd4644f` |
| `tinystories/last (2).ckpt` | 624,013,675 | `1a60e059397a99343ab17e6e6f3339d622d69906ece5db6b39c3252ad4998a74` |
| `tinystories/NEW/tinystories_a100/checkpoints/best.ckpt` | 624,013,037 | `a7a85b5626c9bd2c743a8773827b1ea1f24de831186490b934ceecab4a41ba6f` |
| `NEW.zip` | 574,917,859 | `b8b4d1cba1b4cf0c12ca046bddf25dca4ec62bf6f9c82a16f9f2b6f6263968f4` |

The uploaded baseline source archive `code.zip` is small and is committed,
along with its extracted source under `external/bcfm_baselines_uploaded_20260807`.
