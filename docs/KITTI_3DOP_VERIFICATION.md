# Published KITTI protocol: verification and review

Executed 2026-10-29 on Windows 10, Python 3.9.25, PyTorch 2.8.0 CPU; CUDA absent.
Branch `develop`, unchanged HEAD `50eac764a015e696f3deba3efe563470e42407f7`.
Implementation is local and uncommitted. This report separates source/file-handling
checks from the unexecuted real-data and GPU acceptance gates.

## F3-01 follow-up — conflicting staging archive

The conditional review found that a valid final archive caused staging identity
validation to be skipped. Fixed locally: every receipt-bound `.part` is now
checked against the receipt SHA-256 **and byte size** before selecting the final
or staged recovery path. Conflicts fail before transfer, publication or receipt
updates; neither archive is overwritten or deleted. Matching files remain
recoverable, including after interruption between hard-link creation and unlink.
Complete, pending and legacy valid receipts retain their existing support.
Unknown archive adoption still requires a trusted expected hash or valid receipt;
ZIP CRC alone is insufficient. Atomic no-overwrite publication is unchanged.

Latest CPU verification: **29 tests passed**, 20 Python syntax files/imports,
five inherited configurations and 23 Bash syntax checks. Existing isolated fusion
and strict frozen official DA-V2 load checks also passed. No CPU failures remain.
The existing consolidated command ran the complete suite:

```powershell
& 'C:\Users\swunnthut.wonn\.conda\envs\depth_monoia\python.exe' -B scripts/research.py verify --experiment A0 --output results/KITTI_3DOP_F3_01_LOCAL_VERIFICATION.json
```

Exit code **1** reflects BLOCKED G1–G4, not failed CPU checks. Actual output is
preserved byte-for-byte in
[F3-01 CPU evidence](reproducibility/KITTI_3DOP_F3_01_LOCAL_VERIFICATION.json).
The earlier 27-test report remains unchanged as historical evidence.

Two new regression tests exercise valid-final/conflicting-staging and
valid-final/identical-staging cases for complete, pending and legacy receipts.
The conflict fixture is itself a valid CRC-passing ZIP of genuine source bytes.
Tests assert preservation of both files and unchanged receipt bytes on rejection,
and zero network transfers. Existing pending-receipt/conflicting-staging and
interrupted-hard-link recovery tests were strengthened with receipt/staging
preservation assertions. These are file-handling tests, not KITTI/GPU evidence.

Before/after file hashes confirm unchanged `kitti_3dop_chen` source metadata,
both original raw split files (including ordering), every configuration and all
baseline training/model/loss/evaluation code. Both upstream pins remain clean.
The owner-created `KITTI_3DOP_REVIEW.patch` and other prior changes are preserved.

Exact follow-up scope: four existing files modified—`depth_monoia/kitti_download.py`,
`tests/test_kitti_download.py`, this document and `PROJECT_STATUS.md`—plus the
new evidence JSON linked above. No commit, staging or push. No KITTI archive
download, GPU operation or paid resource. G1–G4 and their missing inputs below
remain BLOCKED; this fix does not grant reproduction or GPU readiness acceptance.

## Results

| Item | Result | Evidence / scope |
| --- | --- | --- |
| Published 3DOP/Chen source attribution | PASS | Author repository and paper chain, plus independent exact copy; [source audit](KITTI_3DOP_PROTOCOL.md) |
| Immutable source revision and original raw file identities | PASS | Frustum PointNets `2ffdd345e1fce4775ecb508d207e0ad465bcca80`; exact paths, SHA-256 and Git blob SHA-1 recorded |
| Counts, ASCII format, uniqueness, disjointness, all 7481 IDs | PASS | Executed checks on downloaded real ID lists, not generated IDs |
| Original ordering/bytes, LF/no final newline | PASS | Raw byte lengths/hash/blob checks; independent OpenPCDet base64 payload comparison |
| Common A0–A3 protocol and immutable hash enforcement | PASS | Inherited configuration checks; reordered same-member list and altered manifest/hash/protocol rejected |
| Baseline setting preservation | PASS | Resolved common config equals HEAD after removing only new dataset.protocol; model/loss/optimizer/schedule/evaluation source unchanged |
| F3/F3-01 publication recovery and both-path identity | PASS locally | Conflicting staging rejected; identical files and interrupted hard links recover; complete/pending/legacy receipts covered |
| Syntax/imports and existing operational checks | PASS | 20 Python files, 5 configs, 29 tests, 23 Bash syntax checks |
| Existing isolated fusion and frozen official DA-V2 load | PASS | Existing consolidated CPU checks; not detector training/evaluation evidence |
| Upstream pins | PASS | MonoIA `69d6ee30ca5e44848727387dd44af6d56d62c24f`; DA-V2 `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`, both clean |
| Remaining implemented CPU checks | FAIL: none | Initial Windows read-only-fsync failure was corrected; final checks pass |
| Official archive access and actual extraction/G1 | BLOCKED | Operator-authorized archive URLs/session access and full genuine files unavailable locally |
| Linux CUDA build and real-batch forward/backward/G2 | BLOCKED | No authorized instance or GPU execution; batch 16 fit on 24 GB unknown |
| Full-state GPU resume/G3 | BLOCKED | Requires same actual runtime/data and measured next-step agreement |
| Independent KITTI R40 evaluator reference/G4 | BLOCKED | Real predictions plus independently executed metrics for the pinned validation SHA-256 missing |
| Exact original MonoIA reproduction | BLOCKED for that claim | Original author split confirmation unavailable; not required for the owner-approved named published protocol |
| Paid deployment/training | BLOCKED | Human review/source transfer and explicit owner approval required |

No fabricated benchmark values, synthetic labels/images or dummy detector outputs
were used. File-handling ZIP fixtures contain genuine repository README bytes.
Real split-only checks do not establish authenticity/completeness of KITTI images,
labels or calibration. Linux directory-fsync/power-loss durability and actual
official archive contents remain unexecuted; local tests do not certify them.

## Original implementation commands and evidence (27-test snapshot)

From the project root, using the existing CPU environment:

```powershell
& 'C:\Users\swunnthut.wonn\.conda\envs\depth_monoia\python.exe' -B -m unittest discover -s tests -v
& 'C:\Users\swunnthut.wonn\.conda\envs\depth_monoia\python.exe' -B scripts/research.py verify --experiment A0 --output results/KITTI_3DOP_LOCAL_VERIFICATION.json
git -c core.excludesFile= diff --check
```

The direct unittest run passed all 25 then-present tests. Two additional focused
recovery tests were added afterward; the final consolidated command ran **all 27**
successfully. Its exit code is **1**, intentionally reflecting BLOCKED G1–G4,
not a static/test failure. Do not report this command as overall reproduction PASS.
No new verification framework or GPU task was added.

The consolidated report was copied byte-for-byte to
[KITTI_3DOP_LOCAL_VERIFICATION.json](reproducibility/KITTI_3DOP_LOCAL_VERIFICATION.json).
It records actual environment, tested code hashes, test count and blocked gates.
The separate [split evidence JSON](reproducibility/KITTI_3DOP_SPLIT_VERIFICATION.json)
records actual membership/byte checks and the baseline-config comparison. Both
report the real uncommitted working state rather than a fabricated commit.

Focused regression coverage:

- Corrupted ZIP CRC and unsafe member rejection before final file or receipt.
- Interruption after pending receipt, before file publication: no-network recovery.
- Interruption after hard link, before staging unlink: final file reused safely.
- Interruption after publication, before complete receipt: no-network recovery.
- Complete-cache rerun and legacy verified-receipt reuse without downloading.
- Unknown final cache rejection; trusted expected-hash adoption still requires CRC.
- Changed staged/final bytes, conflicting expected hash and racing destination
  rejection without overwriting them.
- Undefined protocols and split hash overrides fail; reordered original IDs fail
  despite unchanged membership. Missing local source-list caches cause the two
  real-list tests to skip explicitly on other machines, never fabricate lists or
  silently download them. Both caches were present for this executed check.

## Original implementation file inventory (before F3-01)

Created in this implementation:

- `docs/KITTI_3DOP_PROTOCOL.md`
- `docs/KITTI_3DOP_VERIFICATION.md`
- `docs/reproducibility/kitti_3dop_chen.json`
- `docs/reproducibility/KITTI_3DOP_SPLIT_VERIFICATION.json`
- `docs/reproducibility/KITTI_3DOP_LOCAL_VERIFICATION.json`
- `tests/test_kitti_protocol.py`

Modified implementation/configuration/tests:

- `depth_monoia/kitti_download.py`, `depth_monoia/provenance.py`, `depth_monoia/runner.py`
- `configs/common.yaml`, `configs/vast/kitti_sources.example.json`
- `docs/reproducibility/dataset.example.json`
- `scripts/vast/prepare_kitti.sh`, `tests/test_kitti_download.py`

Modified instructions/status/guides:

- `AGENTS.md`, `PROJECT_STATUS.md`, `README.md`
- `docs/VAST_AI_RUNBOOK.md`, `docs/GPU_SETUP_GUIDE.md`, `docs/GPU_TRAINING_GUIDE.md`
- `docs/REPRODUCIBILITY.md`, `docs/EXPERIMENTS.md`, `docs/EVALUATION_PROTOCOL.md`
- `docs/KITTI_SOURCE_AND_SPLIT_AUDIT.md`: pre-existing untracked historical audit
  preserved, with a short superseding-decision/F3 note added. Its original findings
  are not rewritten as evidence for the new implementation.

Original snapshot Git scope: 17 tracked files modified plus 7 untracked files (including that prior
audit). Nothing staged; branch/HEAD unchanged. Raw split lists are ignored under
`data/protocols/kitti_3dop_chen/`; no dataset payloads/checkpoints/credentials are
added to Git. No commit, push, paid resource or training operation was performed.

Review the source chain, pins, protocol claim and F3 diff before deciding on a
commit/source transfer. Do not rent a GPU on the assumption G1–G4 have passed.
