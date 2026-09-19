# Split policy — what the project documents specify, and a proposal for approval

**Status: NO SPLIT HAS BEEN MADE.** This document records the protocol found in
the supplied documents and proposes a design. Nothing below is executed until
approved.

## 1. What the documents actually say

| Source | Statement about the split | Ratio fixed? |
|---|---|---|
| `AI_TECHTAHON_DOCUMENTS/ROADMAO/NEW_ROADMPA_ENTIRE HACKATHON.docx` (Phase 8) | "CLEAN MASTER → DEVELOPMENT DATA / FINAL TEST (never touch) → 10-FOLD CV on development data"; "**The exact percentage can be decided based on your final dataset size/class distribution.**"; diagram labels it "STRATIFIED SPLIT" | **No** |
| `AI_TECHTAHON_DOCUMENTS/Review_dcouments/ai_hackathon_method.docx` (§8) | "Clean dataset → Train / Validation / Test. **For example:** 80 % → development data, 20 % → final TEST. Then 10-fold cross-validation on the 80 %" | No — given as an example |
| Same document (§5–6) | duplicate removal before splitting; exact + perceptual screening; same-fish (specimen) leakage must be prevented (LOFO cited); document where specimen ids are unavailable | protocol, not ratio |
| Teacher's workflow (as relayed in the task) | "Train / Validation / Test Split OR Train / Test Split → 10-fold Cross Validation" | No |
| `AI_TECHTAHON_DOCUMENTS/PRD_AquaHealthAI_Final.md` (old baseline) | stratified 1,680 / 360 / 360 (70/15/15) of a 2,400-image set | For the **old** dataset only |
| Existing implementation `src/manifest.py::group_aware_stratified_split` | per-class 70/15/15, seed 42, largest-remainder quotas, near-duplicate groups kept in one split | Old baseline; not stated for the new experiment |

**Conclusion: the ratio for the new experiment is unspecified.** The only fixed
requirements are: split after de-duplication, a frozen final test partition,
10-fold CV inside the development partition, and no duplicate / near-duplicate /
same-specimen leakage across partitions.

## 2. Facts about the clean corpus the design must respect (from `dedup_report.json`, `leakage_report.json`)

- 5,942 included images, 8 classes, 4 contributing sources (`paper_dataset` currently contributes 0: both of its classes are UNRESOLVED).
- 3,301 leakage groups; 564 groups have more than one image; the three largest (396, 259, 222 images) are MatsyaDx-BD *Healthy Fish* groups formed by chaining specimen ids with dHash matches between different specimens. A group is indivisible, so the finest possible partition granularity is bounded by these groups.
- Class × source is uneven (e.g. Aeromoniasis / Parasitic / Saprolegniasis / White Tail come from `current_freshwater` + `roboflow` only; Bacterial Red is 59 % MatsyaDx-BD).
- Resolution is class-correlated (feature-only upper bound 0.35 vs 0.24 majority baseline).

## 3. Proposed design (for approval — not executed)

1. **Unit of assignment: `group_id`** from `clean_manifest.csv` (never single images). This alone prevents exact, near-duplicate and same-specimen leakage.
2. **Two-level protocol, as both documents describe:**
   - Level 1 — one frozen **FINAL TEST** partition, group-aware and stratified by `(unified_class, source_dataset)` so every class *and* every source is represented in the test set; assigned once, written to `data/audit/split_v2/test.csv`, digest-guarded like `data/split_manifest.csv`, never read by training or tuning.
   - Level 2 — the remaining **DEVELOPMENT** partition is cut into **10 group-aware stratified folds** (`fold_01.csv … fold_10.csv`) once; every model uses the same folds. GAN augmentation (later phase) is generated inside each fold's training portion only.
3. **Ratio — to be decided by you.** Two candidates consistent with the documents:
   - **A. 80 / 20 development / test** — the method document's example; with 5,942 images ≈ 4,750 / 1,190; each CV fold's validation portion ≈ 475 images.
   - **B. 85 / 15** — closer to the existing baseline's 15 % test share (509 of 3,405); ≈ 5,050 / 890.
   Either way the largest group (396) forces ±6.7 % lumps; the stratification will be approximate (largest-remainder quotas per class as in `src/manifest.py`, then deficit-driven group assignment).
4. **Seed:** fixed, recorded in the manifest (proposal: 42, the project constant `src/config.py::SEED`).
5. **Implementation:** extend `src/manifest.py::group_aware_stratified_split` (it already does group-aware per-class quotas for three splits) to (a) a two-way dev/test call and (b) a 10-fold assignment; do not rewrite it.
6. **Record:** every image's `image_id`, `group_id`, `partition` (`test` | `dev`) and `fold` (1–10 | none) in one manifest with a SHA-256 digest; the test partition is written once and refused for overwrite, exactly like the existing `build_split_manifest.py` behaviour.
7. **Open questions to settle before running it:** the 6 UNRESOLVED classes (1,277 images; their decision changes class counts), whether MatsyaDx-BD healthy specimens may be split by specimen id only (would break the 396/259/222 super-groups into 86 specimen groups; requires accepting that two *different* fish with identical dHash may land in different partitions), and whether `roboflow` (279 of 454 images are copies of the current dataset) should be dropped as a source altogether.
