# KITTI published 3DOP/Chen protocol

Decision and audit date: 2026-10-29. Protocol identifier: `kitti_3dop_chen`.
Repository: `wolverSTW/depth-mono-thesis`, branch `develop`.

The owner approved authentic KITTI with the published 3DOP/Chen research
partition: 3,712 training and 3,769 validation frames. The source and actual split
bytes below are **VERIFIED**. This is not an official KITTI train/validation
partition. Exact original MonoIA split equivalence remains **NOT VERIFIED**;
results must be described as MonoIA and depth-fusion comparisons under this named
published protocol. No reproduction, training, evaluation or GPU readiness result
is implied by verifying these two lists.

## 1. Source and research attribution

Original research citation: Xiaozhi Chen, Kaustav Kundu, Yukun Zhu, Andrew G.
Berneshawi, Huimin Ma, Sanja Fidler and Raquel Urtasun. **3D Object Proposals for
Accurate Object Class Detection**. Advances in Neural Information Processing
Systems 28 (NIPS 2015), pp. 424–432.
[Official proceedings](https://papers.neurips.cc/paper_files/paper/2015/hash/6da37dd3139aa4d9aa55b8d237ec5d4a-Abstract.html).

The chosen public distributor is the original research repository
[charlesq34/frustum-pointnets](https://github.com/charlesq34/frustum-pointnets),
at immutable commit **`2ffdd345e1fce4775ecb508d207e0ad465bcca80`**.
The [pinned README](https://github.com/charlesq34/frustum-pointnets/blob/2ffdd345e1fce4775ecb508d207e0ad465bcca80/README.md)
identifies the authors' CVPR 2018 release and explicitly uses
`kitti/image_sets/train.txt` and `kitti/image_sets/val.txt` in data preparation.
Only the ID lists are used here; its processed point clouds, labels, detections,
training code and pretrained models are not KITTI acquisition sources.

The attribution rests on primary evidence, not equal list sizes:

1. Qi et al., [Frustum PointNets, arXiv:1711.08488v2](https://arxiv.org/pdf/1711.08488v2),
   sections 5.1–5.2, PDF page 6, explicitly use the same KITTI partition as their
   reference [6], MV3D.
2. Chen et al., [MV3D, arXiv:1611.07759v3](https://arxiv.org/pdf/1611.07759v3),
   section 4, PDF page 6, follow reference [4] for the roughly half-and-half
   partition. Reference [4], PDF page 9, is the 2015 3DOP paper cited above.
3. As corroboration of the name, the established
   [CenterNet DATA.md](https://github.com/xingyizhou/CenterNet/blob/4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c/readme/DATA.md)
   at commit `4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c` explicitly calls Chen's
   `files/mv3d/imagesets.tar.gz` partition **3DOP**, separately from SubCNN.
   CenterNet annotation conversion is not used in this project.
4. An independent established implementation,
   [OpenPCDet ImageSets](https://github.com/open-mmlab/OpenPCDet/tree/233f849829b6ac19afb8af8837a0246890908755/data/kitti/ImageSets)
   at commit `233f849829b6ac19afb8af8837a0246890908755`, distributes the same two
   files. Git blob identities and base64-decoded raw contents were compared:
   both are byte-for-byte identical to the selected Frustum PointNets files.

This supports identification of the published research partition. It does not
establish MonoIA's exact original membership or prove byte identity with Chen's
unavailable historical tarball. Its URL returned 404 in the earlier audit; no
replacement URL or missing author confirmation is invented. Sequence separation
is described in the research literature; it has not been independently checked
against original KITTI video mappings here.

The Frustum PointNets supplement, PDF page 11, prints **3,717** training frames,
inconsistent with its stated total and validation count. The actual released file
has **3,712**. This discrepancy is recorded rather than silently repairing the
paper or inferring membership from a count. File evidence, the explicit paper
attribution and independent byte identity are the basis of this protocol.

## 2. Immutable files and raw-byte evidence

| Split | Exact source path | Entries | Bytes | Git blob SHA-1 |
| --- | --- | ---: | ---: | --- |
| Train | `kitti/image_sets/train.txt` | 3712 | 25983 | `505b1e26668c85104eb1da09d68caede331fbd25` |
| Validation | `kitti/image_sets/val.txt` | 3769 | 26382 | `258ca1147742bbf8172311feb7e37e1e1d67e5c9` |

Original raw file SHA-256:

```text
b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb  train.txt
657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86  val.txt
```

Direct immutable HTTPS payloads:

- [train.txt](https://raw.githubusercontent.com/charlesq34/frustum-pointnets/2ffdd345e1fce4775ecb508d207e0ad465bcca80/kitti/image_sets/train.txt)
- [val.txt](https://raw.githubusercontent.com/charlesq34/frustum-pointnets/2ffdd345e1fce4775ecb508d207e0ad465bcca80/kitti/image_sets/val.txt)

The GitHub contents API returned base64, decoded directly to binary files without
text normalization. Both use ASCII IDs and LF separators, with **no trailing
newline**. Counts used parsed records, not `wc -l` (which reports one fewer).
The Git blob SHA-1 was recomputed over `blob <byte-length>\0<raw-bytes>` and
matched the API identity. The independent OpenPCDet base64 payloads also matched.
No sorting, shuffling, newline insertion or reserialization was performed.

| Content requirement | Result | Evidence |
| --- | --- | --- |
| Train exactly 3712; validation exactly 3769 | PASS | Actual raw files parsed |
| Unique IDs within each file | PASS | Set cardinalities equal record counts |
| Train/validation disjoint | PASS | Intersection empty |
| Union covers all 7481 official labeled IDs | PASS | Exactly `000000` through `007480` |
| Six ASCII digits per ID | PASS | Full-match `[0-9]{6}` |
| Original source bytes and ordering preserved | PASS | SHA-256, byte lengths and Git blob identities match |
| Source attribution beyond matching counts | PASS | Author repository + paper chain + independent exact copy |
| Exact original MonoIA split | NOT VERIFIED | No author-confirmed binding to these files |

Machine-readable evidence:
[protocol pins](reproducibility/kitti_3dop_chen.json) and
[executed split checks](reproducibility/KITTI_3DOP_SPLIT_VERIFICATION.json).
The small downloaded lists remain under ignored `data/protocols/kitti_3dop_chen/`;
they are not added as Git dataset payloads. No KITTI image, calibration, label or
archive was downloaded to Windows.

## 3. Preparation on an approved Vast.ai server

Follow [VAST_AI_RUNBOOK.md](VAST_AI_RUNBOOK.md) for source transfer, environment
setup, backups and approval boundaries. These commands are prepared, not GPU
executed. From the reviewed project root on the Linux server:

```bash
cp -n configs/vast/kitti_sources.example.json local.kitti_sources.json
chmod 600 local.kitti_sources.json
# Edit the ignored JSON with authorized official archive URLs/provenance/auth.
# Keep protocol = kitti_3dop_chen. Split URLs/hashes load automatically.
bash scripts/vast/prepare_kitti.sh
```

Only obtain `data_object_image_2.zip`, `data_object_calib.zip` and
`data_object_label_2.zip` through the
[official KITTI 3D benchmark download flow](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
Obtain account authorization and exact resulting URLs yourself; anonymous access
to a final CDN object is not assumed. Set `authorized_official_downloads: true`
only after approval under KITTI terms. Archive SHA-256 values are optional when
there is no independently trusted expected digest: authorized official HTTPS is
the source trust, and locally recorded SHA-256/CRC are integrity evidence.
Credentials and signed URLs stay in ignored local configuration, not metadata.
Public research split downloads receive no KITTI cookies/netrc.

Default server paths under `TASK_ROOT=/workspace/depth-monoia`:

```text
downloads/kitti/{train,val}.txt
downloads/kitti/data_object_{image_2,calib,label_2}.zip
downloads/kitti/verified_archives.json
datasets/KITTIDataset/ImageSets/{train,val}.txt
datasets/KITTIDataset/training/image_2/000000.png ... 007480.png
datasets/KITTIDataset/training/calib/000000.txt ... 007480.txt
datasets/KITTIDataset/training/label_2/000000.txt ... 007480.txt
evidence/local.dataset.json
```

Archives may also contain testing RGB/calibration files; these never enter
training. Original KITTI 3D label and calibration TXT files are used directly.
No YOLO/COCO conversion, LiDAR, right images or synthetic annotations are used.
Actual official ZIP member compatibility and extraction remain **BLOCKED** until
authorized acquisition; strict layout rejection is retained rather than relaxed.

Preparation downloads the immutable split bytes, checks the pinned hashes and
membership, validates every archive before extraction, copies split bytes
unchanged, and generates a manifest only after the existing full-data checks.
An optional local `splits.train.path` / `splits.val.path` can reference already
acquired lists, but their bytes must still match the pins. Attempts to change
source URL, counts, hashes or manifest protocol are rejected. Existing unspecified
or original-MonoIA manifests must not be relabelled; generate a new manifest at
a new path and preserve the old evidence. Successful reruns preserve added
manifest fields and refuse conflicting identities.

## 4. Common A0–A3 membership and research scope

All A0–A3 configurations inherit `dataset.protocol: kitti_3dop_chen` from
`configs/common.yaml`. G1 requires exact pinned train/val bytes, full required
files, disjoint coverage and annotation/calibration inventory. Run identity also
includes the protocol definition hash. Configuration/manifest mismatches fail.
No old gates or checkpoints are silently relabelled for the new protocol.

Original A0 architecture, optimizer, losses, schedule, 250 epochs, seed 444,
batch 16, FP32, augmentations and R40 evaluation remain unchanged. The approved
named split choice is explicit. Batch reductions still require a separate
recorded deviation shared across comparisons. A0–A3 use existing separate run,
checkpoint and log directories. After actual data and G1–G4 acceptance on an
approved server, existing commands remain:

```bash
bash scripts/vast/train_a0.sh
bash scripts/vast/train_a1.sh
bash scripts/vast/train_a2.sh
bash scripts/vast/train_a3.sh
bash scripts/vast/evaluate_all.sh
```

These commands were **not run**. Do not describe a future A0 score as an exact
original MonoIA reproduction without additional author split evidence. Do not
claim the original validation checkpoint was trained without overlap with this
chosen validation set. Independent G4 evidence must bind the exact validation
SHA-256 above; published table values cannot replace reference execution.

## 5. F3 interruption recovery and rerun contract

Archive staging uses `<archive>.part`. After optional expected-hash comparison,
safe-member/layout checks and full ZIP CRC verification, preparation fsyncs the
staged bytes and atomically writes a `verified_pending_publish` receipt. An
atomic same-filesystem hard link publishes the archive without overwriting an
existing destination; the staging name is then removed. Finally the receipt is
atomically updated to `complete`. Linux directory fsyncs establish ordering.
The existing cache lock serializes preparation. This is a recoverable journal,
not a claim of one atomic transaction across two files.

Rerun the same preparation command after an interruption. A pending receipt plus
matching staged or final bytes recovers without another transfer; a complete
verified cache is reused. Hash, size, ZIP safety and CRC are rechecked. A crash
after creating the hard link may leave both names for the same bytes; the valid
final cache is reusable without overwriting the staging name. Existing extracted
files are checked and retained; publication cannot clobber a racing conflict.

An unknown old final ZIP without a receipt or independently trusted expected hash
is refused even if its ZIP CRC is good. Preserve/quarantine it for investigation,
or supply a genuinely trusted expected hash with provenance; do not manufacture
a receipt by hashing unknown bytes. Changed receipts, staged/final hash conflicts
and extracted-file conflicts also stop without overwriting evidence.

Incomplete unverified `.part` transfers use curl range resume with retries.
Expired authorization or unsupported range requests fail with partial bytes
retained. If a complete unjournaled partial cannot be resumed by the server,
preserve it under a separate name and explicitly start a fresh authorized
transfer; existence alone never promotes it. Receipts/cache/manifest must be
backed up if acquisition recovery across destroyed containers is needed. Vast
container storage is temporary and volumes are not assumed portable across hosts.

## 6. Acceptance and remaining blockers

| Requirement | Status | Acceptance / missing input |
| --- | --- | --- |
| Published source, immutable files, hashes and membership | VERIFIED | Evidence recorded above; no random split |
| Protocol enforcement and F3 source-byte regression tests | VERIFIED locally | See verification report; Linux crash durability not exercised here |
| Official dataset access | BLOCKED | Authorized account flow, actual image/calibration/label URLs and any required session files |
| Genuine acquisition, archive structure, complete dataset and G1 | BLOCKED | Execute server preparation against official packages and retain manifest/receipts |
| Linux CUDA build, real batch capacity and full-state GPU resume | BLOCKED | Approved Vast instance and actual G2/G3; 24 GB fit is unknown |
| Evaluator correctness G4 | BLOCKED | Real predictions and independent R40 measurements for this exact validation hash |
| Exact original MonoIA reproduction | NOT VERIFIED | Author-confirmed original split bytes/membership still absent; not required for the approved published protocol claim |
| Paid operations | BLOCKED | Owner review, reviewed source transfer and explicit rental/training approval |

Full GFLOPs/custom-op coverage remains a reporting limitation, not a dataset
acquisition blocker. No GPU was rented, training started, result fabricated,
commit created or push performed. See [verification report](KITTI_3DOP_VERIFICATION.md).
