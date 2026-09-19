# Preprocessing / CLAHE validation (Phase 7) — existing implementation on the new dataset

## What is reused

- **decode**: src/dataset.py::load_image — Pillow Image.open(...).convert('RGB'); failure raises ImageDecodeError naming the file
- **pipeline**: src/preprocessing.py::build_eval_transform (validation/test/inference) and src/augmentation.py::build_train_transform (training) — reused unchanged
- **clahe_class**: src/preprocessing.py::CLAHE (lines 79-118): cv2.createCLAHE on the L channel of cv2.COLOR_RGB2LAB, back with cv2.COLOR_LAB2RGB

## Configuration (read from the code, not invented)

| item | value |
|---|---|
| colour space / channel | LAB / L (lightness) only; A and B untouched |
| clip limit | 2.0 |
| tile grid size | [8, 8] |
| position | first stage, on the full-resolution decoded RGB image, before resize/crop |
| resize | Resize(shorter side -> 256, bicubic, antialias) then CenterCrop(224) [eval]; RandomResizedCrop(224) [train] |
| normalisation | mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225] |
| tensor | PILToTensor -> ToDtype(float32, scale=True) -> Normalize; output (3, 224, 224) |
| eval stages | CLAHE → Resize → CenterCrop → PILToTensor → ToDtype → Normalize |
| train stages | CLAHE → RandomResizedCrop → RandomHorizontalFlip → RandomRotation → ColorJitter → PILToTensor → ToDtype → Normalize |

Order note: the team method note sketches Resize -> CLAHE; the existing project applies CLAHE -> Resize -> CenterCrop. Kept as implemented; changing it is a decision for approval.

Evidence so far: EXP-002 (CLAHE off) val Macro-F1 0.9182 vs EXP-001 (CLAHE on) 0.9004 under AdamW on the current dataset; whether CLAHE helps is to be measured.

## Validation samples (in memory; one clean image per class and per source)

`l_std` is the standard deviation of the LAB lightness channel before and after CLAHE (ratio > 1 = contrast increased). `output` is the tensor the model would receive.

| sample | dataset | class | input | L std before → after (×) | output | mean / std | sheet |
|---|---|---|---|---|---|---|---|
| class_Bacterial_Red_Disease | current_freshwater | Bacterial Red Disease | 224x224 | 48.121 → 48.705 (×1.012) | 3x224x224 torch.float32 | -0.5184 / 0.8783 | `data/audit/clahe_samples/class_Bacterial_Red_Disease.png` |
| class_Aeromoniasis | current_freshwater | Aeromoniasis | 128x128 | 93.767 → 86.017 (×0.917) | 3x224x224 torch.float32 | 0.7324 / 1.5219 | `data/audit/clahe_samples/class_Aeromoniasis.png` |
| class_EUS_Disease | current_freshwater | EUS Disease | 640x640 | 50.905 → 60.506 (×1.189) | 3x224x224 torch.float32 | 0.0648 / 1.3769 | `data/audit/clahe_samples/class_EUS_Disease.png` |
| class_Saprolegniasis | current_freshwater | Saprolegniasis | 128x128 | 60.167 → 68.639 (×1.141) | 3x224x224 torch.float32 | -0.3498 / 1.1566 | `data/audit/clahe_samples/class_Saprolegniasis.png` |
| class_Bacterial_Gill_Disease | current_freshwater | Bacterial Gill Disease | 128x128 | 55.336 → 61.616 (×1.113) | 3x224x224 torch.float32 | 0.4912 / 1.0817 | `data/audit/clahe_samples/class_Bacterial_Gill_Disease.png` |
| class_Parasitic_Disease | current_freshwater | Parasitic Disease | 640x640 | 58.23 → 67.091 (×1.152) | 3x224x224 torch.float32 | 0.9777 / 1.2294 | `data/audit/clahe_samples/class_Parasitic_Disease.png` |
| class_White_Tail_Disease | current_freshwater | White Tail Disease | 128x128 | 52.748 → 59.556 (×1.129) | 3x224x224 torch.float32 | 0.5804 / 1.1732 | `data/audit/clahe_samples/class_White_Tail_Disease.png` |
| class_Healthy_Fish | current_freshwater | Healthy Fish | 640x640 | 66.664 → 71.233 (×1.069) | 3x224x224 torch.float32 | 1.1422 / 1.3224 | `data/audit/clahe_samples/class_Healthy_Fish.png` |
| source_current_freshwater | current_freshwater | Bacterial Red Disease | 224x224 | 48.121 → 48.705 (×1.012) | 3x224x224 torch.float32 | -0.5184 / 0.8783 | `data/audit/clahe_samples/source_current_freshwater.png` |
| source_kaptai | kaptai | EUS Disease | 150x150 | 49.474 → 57.834 (×1.169) | 3x224x224 torch.float32 | 0.3953 / 0.943 | `data/audit/clahe_samples/source_kaptai.png` |
| source_mendeley | mendeley | Bacterial Gill Disease | 4000x3000 | 46.432 → 53.368 (×1.149) | 3x224x224 torch.float32 | 0.297 / 0.9727 | `data/audit/clahe_samples/source_mendeley.png` |
| source_roboflow | roboflow | Healthy Fish | 640x640 | 69.501 → 72.943 (×1.05) | 3x224x224 torch.float32 | -0.5413 / 1.1599 | `data/audit/clahe_samples/source_roboflow.png` |

## Corrupted-image handling

- status: **reported** — ImageDecodeError: could not decode image /var/folders/4f/f06lwyf14hgcn5cynr3fb6rw0000gn/T/tmp_q7o080k/corrupt.jpg: cannot identify image file '/var/folders/4f/f06lwyf14hgcn5cynr3fb6rw0000gn/T/tmp_q7o080k/corrupt.jpg'
- the master-manifest probe records undecodable files with status `corrupt` (0 found in the five datasets); the training loader raises instead of skipping

**CLAHE has not been applied to the dataset on disk.** It runs inside the transform of any experiment whose config sets `preprocess.clahe` (`configs/preprocess_v2_clahe.json`).
