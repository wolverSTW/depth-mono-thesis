# KITTI acquisition and original MonoIA split audit

Historical audit, preserved. A subsequent owner decision adopts the verified
published `kitti_3dop_chen` partition; see [new protocol](KITTI_3DOP_PROTOCOL.md)
and [implementation checks](KITTI_3DOP_VERIFICATION.md). That decision supersedes
the requirement to recover exact MonoIA lists for this project's published-split
comparisons. Exact original MonoIA equivalence remains unverified. F3 is now
implemented and locally tested; the findings below describe the audited old code.

Audit date: 2026-10-29. Repository: `wolverSTW/depth-mono-thesis`, branch `develop`.
Audited local HEAD: `50eac764a015e696f3deba3efe563470e42407f7`.
Pinned MonoIA: `69d6ee30ca5e44848727387dd44af6d56d62c24f`.
The working tree was clean at the start of this audit. The preceding RTX 4090
implementation had already been committed before this session; this audit makes
no commit or push. No downloader/model/source-pin changes are included here.

**Decision: BLOCKED for original-protocol reproduction.** The loader contract and
official package categories are established. Exact original MonoIA train/val
membership, ordering and raw-file hashes are not established. The inherited
evaluator's MV3D split link is a useful lead, not sufficient proof of the training
protocol, and returned HTTP 404 to this audit's header-only request. Do not rent
a GPU on the assumption that this provenance problem is resolved.

## Classification and requirement matrix

VERIFIED means supported by the cited source or the stated local check, limited
to that scope. BLOCKED means an identified missing input prevents acceptance.
NOT VERIFIED means an execution/identity claim has not been established; it does
not mean that failure has been demonstrated.

| Request | Classification | Evidence / remaining acceptance condition |
| --- | --- | --- |
| 1. Inspect pinned MonoIA and loader | VERIFIED | Pinned files, config, tree, available history and loader path construction inspected; see sections below. |
| 2. Exact required directory structure | VERIFIED | Loader lines 73–83 and 122–150 establish split/image/calibration/label paths. |
| 3. Inspect official 3D benchmark download page | VERIFIED | Anonymous benchmark page and its linked login/download-policy page read on audit date. |
| 4. Required monocular archives | VERIFIED | Left RGB, camera calibration and training labels; official categories and loader use agree. Final authorized URLs are a separate requirement. |
| 5. Authentication requirements | VERIFIED | Official entry point requires registered download access. Authentication of an actual final archive/CDN URL is NOT VERIFIED. |
| 6. No invented URLs/access bypass | VERIFIED | No archive endpoint guessed, no registration/login bypass, no authenticated session attempted. |
| 7. Investigate original split provenance | VERIFIED | Pinned evaluator citation, repository trees/history, author site/paper/models and public MonoIA issues/comments inspected. |
| 8. Establish exact original split files | BLOCKED | No author-provided lists/hashes or explicit binding of a recoverable canonical pair to this MonoIA experiment was found. |
| 9. No unrelated/random replacement split | VERIFIED | No split substituted, generated, installed or added to configuration. |
| 10. Inspect downloader/specification | VERIFIED | Current committed `kitti_download.py`, example JSON, preparation wrapper and common paths reviewed. |
| 11. Correct real official-archive extraction | NOT VERIFIED | Path mapping is source-compatible; no real ZIP listing/extraction was executed. Strict member filtering needs confirmation against the acquired packages. |
| 12. Safe reruns | NOT VERIFIED | Matching-cache/extraction behavior checked with source-file bytes. Arbitrary interruption recovery is incomplete: published archive without receipt blocks a retry. See finding F3. |
| 13. Missing acquisition inputs identified | VERIFIED | Exact missing fields and distinction from GPU/G4 prerequisites listed below. |
| 14. Evidence-backed fixes | VERIFIED | Concrete configuration, author-evidence and cache-publication remedies specified; no fabricated source/hash or silent code change. |
| 15. No local dataset download | VERIFIED | No KITTI archive, image, label, calibration or split payload downloaded. Only public source/web metadata and one split-archive HEAD request were read. |
| 16. No paid GPU work | VERIFIED | No instance purchase, allocation, setup or training executed. |
| 17. No commit or push | VERIFIED | HEAD unchanged; deliverable and status note remain local changes. |

## 1. MonoIA's actual filesystem contract

The [pinned loader](https://github.com/alanzhangcs/MonoIA/blob/69d6ee30ca5e44848727387dd44af6d56d62c24f/lib/datasets/kitti/kitti_dataset.py#L71-L85)
reads `ImageSets/<split>.txt` below the configured root. Both `train` and `val`
resolve to `training/`; only `test` selects `testing/`. Getters format sample IDs
as six digits and read PNG images and text calibration/labels.
The [pinned config](https://github.com/alanzhangcs/MonoIA/blob/69d6ee30ca5e44848727387dd44af6d56d62c24f/config/monoia.yaml#L3-L7)
selects `train` / `val`. Thus the local directory called `training` contains both
research partitions; do not move held-out validation images to a new `validation/`
directory. The [author's data-preparation section](https://github.com/alanzhangcs/MonoIA/blob/69d6ee30ca5e44848727387dd44af6d56d62c24f/README.md#data-preparation)
agrees with this layout.

```text
$KITTI_ROOT/                         # Vast default: /workspace/depth-monoia/datasets/KITTIDataset
  ImageSets/
    train.txt                       # exact original IDs/order/bytes still BLOCKED
    val.txt                         # exact original IDs/order/bytes still BLOCKED
  training/
    image_2/000000.png               # analogous files for all designated IDs
    calib/000000.txt
    label_2/000000.txt
  testing/                          # may arrive in full image/calibration archives
    image_2/...
    calib/...
```

The extension substitutes the absolute dataset root through `resolve_config`.
Do not extract into `$KITTI_ROOT/KITTIDataset/training` or
`$KITTI_ROOT/data_object_image_2/training`. Research A0–A3 do not need `test.txt`
or `trainval.txt`. Those belong to other upstream modes. There is no testing-label
requirement for this train/val workflow.

The loader's depth-directory assignment is commented out, and the active
`__getitem__` path does not call `get_depth_map`; its presence as an unused helper
does not create an external depth-map dependency. Default config does not enable
`use_filtered_labels_for_target_focal`, so standard `label_2` is required, not
an invented `label_2_<focal>` dataset. LiDAR, right-camera RGB, raw sequences,
road planes and KITTI depth-completion packages are not read by this workflow.

## 2. Official acquisition evidence and authentication boundary

| Required official package | Downloader's cache filename | Consumed content |
| --- | --- | --- |
| Object-detection left colour images | `data_object_image_2.zip` | `training/image_2/*.png` (testing images may also be packaged) |
| Object-detection camera calibration | `data_object_calib.zip` | `training/calib/*.txt` (testing calibration may also be packaged) |
| Object-detection training labels | `data_object_label_2.zip` | `training/label_2/*.txt` |

These are the three categories on the [official KITTI 3D download page](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
It lists approximately 12 GB, 16 MB and 5 MB respectively, with 7481 labeled
training frames and 7518 test frames. The table's filenames are the current
downloader's internal cache names; they are not a claim to have resolved final
authorized HTTP URLs. [Torchvision 0.19.1 source](https://github.com/pytorch/vision/blob/v0.19.1/torchvision/datasets/kitti.py)
independently documents the image/label names and `training` / `testing` layout.
Its downloader does not supply MonoIA's calibration or original research split.
The upstream KITTI evaluator is already vendored, so a separate development-kit
download is not an input required by the present runner.

Anonymous archive links on the official page resolve to the login page. The
[official download policy](https://www.cvlibs.net/datasets/kitti/user_login.php)
requires registration and a declared use; its first registration stage covers
download-only access. The site also assigns research train/validation partitioning
to participants. Therefore its official train/test distinction does **not**
authenticate a particular MonoIA 3712/3769 research partition.

VERIFIED: the official browser entry flow gates access. NOT VERIFIED: whether
each final URL currently uses cookies, a signed query, another authorized handoff,
or no additional object-store credential. A static historical URL in third-party
client code does not establish present authorization or justify skipping the
official flow. No archive endpoint was requested in this audit.

The owner must complete the official login flow and obtain the three actual
authorized HTTPS download URLs. Keep tokens and session material out of this
report/Git/chat. The current downloader can pass existing Netscape cookie files
or netrc settings to curl, but cannot perform a JavaScript/form login itself.
Netrc/basic HTTP authentication is not proof of compatibility with KITTI's web
form. Session compatibility, redirects and range support remain NOT VERIFIED
until an authorized server-side transfer. See the [curl manual](https://curl.se/docs/manpage.html)
for cookie, netrc and continue-at behavior.

No expected archive SHA-256 values are published in the inspected anonymous
download listing. This is not proof that none exist elsewhere. Known trusted
expected hashes may be supplied; their absence is not a mandatory blocker in
the current design. Without them, the code records computed hashes and CRC
checks after operator-authorized HTTPS acquisition. CRC proves ZIP integrity,
not origin; a newly computed hash is not an independent authenticity reference.

## 3. Original split provenance investigation

### Findings that are VERIFIED

- The [pinned MonoIA recursive tree](https://api.github.com/repos/alanzhangcs/MonoIA/git/trees/69d6ee30ca5e44848727387dd44af6d56d62c24f?recursive=1)
  contains 91 entries, is not truncated, and contains no ImageSets directory,
  `train.txt`, `val.txt`, split archive or split-generation file. The available
  local history has three commits; inspecting dataset/split paths found no lists.
  `data/` is ignored. Config names alone do not identify file contents.
- Remote `main` matched this pin during the audit; the public releases collection
  was empty. All six public MonoIA issues and their available comments were
  inspected; none supplied original split files/hashes or a split-use confirmation.
  These are bounded findings, not a statement about private author files.
- The [author model-repository listing](https://huggingface.co/zhihao406/MonoIA/tree/e9c56d4f474bda66de8926f9dae8008c656addf7)
  exposes two model checkpoints, README and Git attributes, not split lists.
  The [project page](https://alanzhangcs.github.io/monoia-page/) likewise supplies
  paper/code/model links rather than an identified split artifact.
- Crucially, [MonoIA's pinned evaluator README, line 27](https://github.com/alanzhangcs/MonoIA/blob/69d6ee30ca5e44848727387dd44af6d56d62c24f/lib/datasets/kitti/kitti_eval_python/README.md#L27)
  points its example validation list to
  `https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz`.
  This is a real cited URL, not an invented candidate. The document explicitly
  identifies the evaluator as borrowed code. Its blob SHA-1 is
  `888eee3aebe63ea658358a9b02ffe79659731919`.
- That identical README blob occurs in the inspected author MonoCoP and MonoDETR
  trees. Their shared citation is inherited documentation, not three independent
  confirmations of MonoIA's training IDs. Inspected related trees also contain
  no split lists:

| Related primary repository | Inspected revision | Tree entries / split-file finding |
| --- | --- | --- |
| [Author MonoCoP](https://github.com/alanzhangcs/MonoCoP/tree/7d3932444f7ea94be564ae61a0b334fc6673213b) | `7d3932444f7ea94be564ae61a0b334fc6673213b` | 98; no lists; same evaluator README blob |
| [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR/tree/6994b9f512400b258c6edb75f77423beb9c126f2) | `6994b9f512400b258c6edb75f77423beb9c126f2` | 80; no lists; same evaluator README blob |
| [MonoDGP](https://github.com/PuFanqi23/MonoDGP/tree/aa059a18214aebf644510e7f0793971b403f9d14) | `aa059a18214aebf644510e7f0793971b403f9d14` | 100; no lists found |

The [MonoIA paper, section 4.1 and appendix C](https://arxiv.org/html/2603.27059v1)
identifies its framework lineage. The inspected full HTML did not resolve an
exact split artifact, membership list or checksum; searches for 3712 and 3769
found no matches. The CVF supplemental PDF endpoint returned 403 to this browsing
environment; that inaccessible endpoint is not evidence of its contents.

### Original MV3D link availability: VERIFIED failure, not verified payload

This exact header-only request was executed, without an archive body:

```powershell
curl.exe --head --location --max-time 30 --silent --show-error --proto '=https' --proto-redir '=https' 'https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz'
```

Observed response: `HTTP/1.1 404 Not Found`, `Content-Type: text/html; charset=utf-8`,
server `GitHub.com`, date `Thu, 29 Oct 2026 14:14:55 GMT`.
The web tool could not access the author's old homepage, and the corresponding
GitHub Pages repository file lookup also returned 404. Only the exact cited
endpoint's observed response is asserted; this does not prove that every copy
of the historical split is unavailable. A HEAD response does not identify any
archive members or compute a split hash.

### What remains BLOCKED

No original split payload or author-authenticated hashes were obtained. Thus
the exact raw SHA-256 values, ordered IDs, line endings and MonoIA-specific
membership are NOT VERIFIED. The example's 3712/3769 counts are enforced by
the local preparation code; matching those counts and covering all 7481 IDs
would still not authenticate a candidate pair. A random disjoint pair of the
same sizes could pass those structural checks.

Do not promote the dead MV3D URL, an unrelated repository's lists, a matching
published AP score, a checkpoint filename or a self-written provenance string
into evidence that the original MonoIA partition has been verified.

Preferred resolution: the MonoIA authors publish the two original files (or an
immutable archive) and identify the training/validation experiment they belong
to. Alternatively obtain an explicit author statement binding this pinned
experiment to an exact recoverable MV3D artifact or immutable pair of lists,
then independently compare the actual IDs/order and bytes/hashes. Preserve that
public statement and acquisition reference. No author message was sent.

A concrete question for the owner to send, if needed:

> For MonoIA commit 69d6ee30ca5e44848727387dd44af6d56d62c24f and its KITTI
> validation training run, could you publish the exact ImageSets/train.txt and
> val.txt, their counts and raw-file SHA-256 hashes? Does the inherited MV3D
> citation identify the exact lists used, including ID order? Its linked archive
> currently returns 404. Please provide an immutable replacement or confirm a
> specific recoverable artifact.

## 4. Downloader compatibility and rerun review

Audited files: [downloader](../depth_monoia/kitti_download.py),
[source specification](../configs/vast/kitti_sources.example.json),
[preparation task](../scripts/vast/prepare_kitti.sh),
[common task environment](../scripts/vast/common.sh), and
[existing G1 verification](../depth_monoia/provenance.py).

| Area | Classification | Current evidence and limit |
| --- | --- | --- |
| Extraction root | VERIFIED | `root / member.filename` preserves the loader's training/image_2, calib and label_2 paths without adding a package-named wrapper directory. |
| Expected ZIP members | NOT VERIFIED | Code accepts only six-digit PNG/TXT members in the specified training/testing folders. Real ZIP central-directory listings have not been examined; unexpected metadata files would be rejected. |
| All required official packages acquired/extracted | NOT VERIFIED | No real archives were transferred or unpacked. Source compatibility alone is insufficient. |
| Origin/authentication | NOT VERIFIED | A true authorization flag, fixed landing-page string and nonempty provenance text are operator attestations; arbitrary HTTPS hosts are not independently authenticated by code. Actual URLs must be traced to the official authorized flow. |
| Integrity before extraction | VERIFIED | Expected SHA-256 is checked when supplied; all ZIPs are CRC-checked before extraction. Cached hashes/receipts and ZIP path/link checks are implemented. |
| Matching cache reuse | VERIFIED | A matching expected hash/receipt returns the cached file without a network request. Confirmed with repository source bytes. |
| Ordinary matching extraction rerun | VERIFIED | Existing size/CRC matches are retained. A source-file ZIP probe produced one new file, then zero new files on rerun. Conflicts raise instead of overwriting. |
| Interrupted download range resume on actual source | NOT VERIFIED | curl continue-at/retry is configured; actual server range support and session expiry behavior remain untested. |
| Crash between archive publication and receipt | BLOCKED | F3 below: receipt-less cache with no expected hash fails closed. It is safe from overwrite but not automatic recovery. |
| Original split enforcement | BLOCKED | Expected split hashes/counts/coverage are checked, but authentic expected values and author binding are missing. |
| Dataset manifest generation | VERIFIED | The existing verifier must pass before manifest publication. Additional manual fields are retained; conflicting identities/provenance fail. Real generated manifest remains absent. |
| Wrapper invocation | VERIFIED | Uses existing research.py prepare-kitti, documented paths, Linux/Conda activation; it does not provision a GPU or train. |

Dataset extraction is not a transaction over the entire directory. An interruption
can leave already published matching files or temporary files; a retry validates
existing files and the final manifest is only published after G1. Presence of
partial folders is not evidence of completed preparation. Cache locking serializes
cooperating invocations using the same cache directory.

## 5. Exact missing inputs and evidence-backed fixes

### F1 — Authorized acquisition inputs: BLOCKED

Current `kitti_sources.example.json` deliberately sets authorization false and
contains REQUIRED URL/provenance placeholders. It is not runnable acquisition
configuration. The missing owner inputs are:

- Registration/download authorization and the three actual authorized HTTPS URLs.
- A valid cookie/netrc file **only if the actual authorized endpoint needs one**;
  do not assume either mechanism authenticates the site's login form.
- Nonsecret public `archives.<kind>.download_source` evidence for each URL.
- The exact original split pair, each trusted expected SHA-256, `sha256_source`,
  counts and author-bound `split_source`; use a verified local path or immutable
  HTTPS URL for each original file.

Fix: copy the existing example to ignored `local.kitti_sources.json`, populate
these real values, and set authorization true only after authorization exists.
Leave archive expected hashes null when unavailable rather than inventing them.
Store session secrets privately, not in public provenance strings or Git.

```bash
# Later, on an authorized Linux acquisition machine; NOT executed in this audit:
cp -n configs/vast/kitti_sources.example.json local.kitti_sources.json
chmod 600 local.kitti_sources.json
# Populate authentic inputs, then use the existing single task:
bash scripts/vast/prepare_kitti.sh
```

Missing G4 reference predictions/metrics and GPU batch fit are separate full-
training requirements, not inputs required by `prepare-kitti`. Resolve author
provenance and download authorization before rental; fetching the actual dataset
can remain a later server-side operation. No GPU is needed to obtain that evidence.

### F2 — Split authentication and durable evidence: BLOCKED

Fix: obtain the author evidence described in section 3. Preserve the exact original
bytes and order; compute raw SHA-256 after trusted acquisition, compare with the
author/canonical evidence, then fill the existing split entries. Do not rewrite
line endings or regenerate/sort a list to make it fit expectations. If authoritative
evidence contradicts the assumed counts, review the protocol explicitly rather
than silently modifying data to satisfy local checks.

Proposed small metadata improvement: retain a public `split_evidence` object in
the generated manifest containing the immutable source reference/author statement
and each split's `sha256_source`. Currently those per-file fields are required on
input but not copied into `manifest.train`/`manifest.val`. Do not copy the private
specification wholesale because it may contain signed URLs/authentication paths.
No code change is made by this report.

### F3 — Post-download interruption recovery: BLOCKED for automatic reruns

Evidence: `download` atomically renames `.part` to the final path at line 91.
The archive caller only checks CRC and publishes `verified_archives.json` at
lines 216–225. If interrupted in this interval with archive sha256 null, the
next call reaches lines 50–52: existing archive, no receipt-derived expected hash,
therefore refusal. The same state can follow a CRC failure after publication.
This was reproduced offline with a ZIP of the real README, zero network calls.

Proposed exact publication order for an archive fix:

1. Download to the existing `.part` staging path; do not publish the final ZIP yet.
2. Check expected SHA-256 when supplied, validate members and ZIP CRC, and compute
   the actual SHA-256 while staged.
3. Atomically publish a verified pending receipt containing the target name and
   staged-file hash (no URL/secrets), then atomically rename `.part` to final.
4. On restart, accept either staged/final bytes only if they match that receipt;
   absent bytes trigger a normal authorized retry. Never blindly trust or delete
   a receipt-less cached file. Preserve hash conflicts for inspection.

This ordering makes a receipt available before the final file becomes visible
and lets either interruption point recover by hash. Add only a focused source-
byte regression for the two interruption boundaries when implementing the fix;
no synthetic KITTI data or extra gate framework is needed.

Existing manual recovery: investigate a receipt-less file's authorized origin,
check ZIP integrity, and supply its independently justified expected hash and
provenance in the private specification before retrying. Merely hashing an
unknown cached file does not establish its official origin. No file was deleted,
relabelled as trusted, or repaired during this audit.

## 6. Acceptance criteria before claiming acquisition/protocol readiness

| Acceptance item | Current classification | Evidence required to change it |
| --- | --- | --- |
| MonoIA root/path agreement | VERIFIED | Preserve the loader/config contract above. |
| Authorized real sources | BLOCKED | Owner obtains and records the official-flow provenance and actual URLs/session requirements; no guessed endpoints. |
| Exact original research split | BLOCKED | Author-provided pair or explicit original-use confirmation tied to a recoverable immutable artifact; verify ordered IDs and exact hashes. Counts alone are insufficient. |
| Split structural/content checks | BLOCKED | Authenticate source first, then six-digit unique IDs, documented counts, disjoint union 000000–007480 and unchanged raw hashes. |
| Official archive extraction | NOT VERIFIED | On the later authorized server, inspect all actual member paths, verify expected hash when available, CRC and final required training files; publish manifest only after existing G1 passes. |
| Normal no-redownload rerun | VERIFIED for source-byte probe | Later confirm identical real archive/split hashes, no unnecessary network transfers and no overwritten conflicts. |
| Arbitrary interruption recovery | BLOCKED | Implement/review F3 or explicitly accept documented manual recovery; test receipt publication boundaries without weakening trust. |
| Linux/authenticated transfer | NOT VERIFIED | Later demonstrate the real supported endpoint/session/redirect/range behavior; do not infer this from mocks. |
| Full training/reproduction | BLOCKED | Above evidence plus existing G1–G4, approved GPU work, actual memory fit, training/evaluation and verified backups. Outside this audit's execution scope. |

Optional archive expected checksums, a new GPU instance and benchmark scores are
not substitutes for original split provenance. No reproduction readiness claim
is justified at this point.

## 7. Checks actually executed and audit identity

- Local read-only inspection of pinned files/config/tree/history and current
  downloader/spec/wrappers; remote public metadata/source queries described above.
- One header-only request to the exact cited MV3D URL: observed 404; no payload.
- Offline source-byte probes with all downloader subprocess/network calls mocked:
  receipt-less final cache rejected; known-hash cache reused without network;
  generic ZIP extraction produced 1 then 0 new files on rerun. No fake images,
  annotations, trained weights or metric values were created.
- Prior existing acquisition tests were read as supporting context. No full
  model/static suite was rerun, because this audit changes no executable code.
- No KITTI dataset/split payload, authenticated login, GPU operation, commit,
  push or message to an author occurred. Real acquisition/extraction remains
  NOT VERIFIED. Source-file temporary probes were cleaned by their temporary-
  directory context.

SHA-256 of audited local executable/input-contract files:

| File | SHA-256 |
| --- | --- |
| `depth_monoia/kitti_download.py` | `c530c04aad561af4244c8f6b41774687b044819f98c27500573c59e3fc1ce8d2` |
| `configs/vast/kitti_sources.example.json` | `187e8095148fed1a8fb9b814dcd81d018c54db6064f529e5c194eecd12865871` |
| `scripts/vast/prepare_kitti.sh` | `bbf9c24a80f23f861e62d67f1733b45491ee16ae01f408ca3c1724916a523967` |
| `MonoIA/lib/datasets/kitti/kitti_dataset.py` | `6e5d0e09f4eb99d0b9cbc5982d0e1766e648c17d4a03fcc3c9fc844246a00690` |
