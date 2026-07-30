# Гайд: оценка AR / MDLM / BD3-LM и TinyStories

Как после `git clone` на новой машине прогнать те же замеры gen-PPL и latency, что в `results/`.

Код бейзлайнов уже в репозитории: [`baselines/`](../baselines/) (пакет `bcfm_baselines`).  
Чекпоинты и датасеты **не** в git — их кладёте локально и указываете путями / env.

---

## 1. Окружение

```bash
cd cfm-text-distillation-smiles-2026
# как в корневом README: mamba/conda env из environment.yaml
# либо уже существующий .venv
source .venv/bin/activate   # скрипты scripts/*.sh ожидают именно это
```

Нужен GPU с запасом под **GPT-J-6B** (судья gen-PPL) плюс генератор.  
Скрипты по умолчанию ставят `CUDA_VISIBLE_DEVICES` из `EVAL_GPU` (дефолт `1`).

```bash
export EVAL_GPU=0   # свободная карта
```

---

## 2. Рекомендуемый layout на диске

```text
cfm-text-distillation-smiles-2026/
  baselines/                 # код (в git)
  data/
    text8/                   # character Text8 (train/val/test.bin + meta)
    tinystories/             # byte TinyStories (скрипт подготовит сам)
  checkpoints/
    baselines/
      ar_text8_seed12345/checkpoints/last.pt
      mdlm_text8_seed12345/checkpoints/last.pt
      bd3lm_b16_text8_seed12345/checkpoints/last.pt
      ar_tinystories_seed12345/checkpoints/last.pt
      mdlm_tinystories_seed12345/checkpoints/last.pt
      tinystories_a100/checkpoints/best.ckpt   # CFM M1/M2 на TinyStories
  results/                   # metrics.json + figures (часть уже в git)
```

`data/` и `checkpoints/` в `.gitignore`. Можно не копировать, а **symlink**:

```bash
mkdir -p checkpoints
ln -sfn /path/to/unzipped/runs checkpoints/baselines
```

---

## 3. Куда писать пути к чекпоинтам

Все пути задаются **переменными окружения** (или правятся в шапке `scripts/*.sh`).  
Дефолт после обновления: `$PWD/checkpoints/baselines/...`.

| Переменная | Что это | Типичное значение |
|---|---|---|
| `BASE_RUNS` | корень с распакованными run'ами Text8 | `$PWD/checkpoints/baselines` |
| `AR_CKPT` | AR Text8 / TinyStories | `$BASE_RUNS/ar_text8_seed12345/checkpoints/last.pt` |
| `MDLM_CKPT` | MDLM | `$BASE_RUNS/mdlm_text8_seed12345/checkpoints/last.pt` |
| `BD3_CKPT` | BD3-LM B=16 Text8 | `$BASE_RUNS/bd3lm_b16_text8_seed12345/checkpoints/last.pt` |
| `M1_CKPT` | Semicat/CFM TinyStories (M1=M2 веса) | `$BASE_RUNS/tinystories_a100/checkpoints/best.ckpt` |
| `BASELINES_CODE` | код пакета | `$PWD/baselines` (обычно не трогать) |
| `TEXT8_DATA_DIR` | корпус для декода/eval | `$PWD/data/text8` или `.../tinystories` |
| `TS_DATA` | byte TinyStories | `$PWD/data/tinystories` |

Если CFM TinyStories-чекпоинт лежит в другом месте (например `NEW/tinystories_a100`), либо сделайте symlink `tinystories_a100 → NEW/tinystories_a100`, либо задайте `M1_CKPT` явно.

Пример на одной машине (как у нас сейчас):

```bash
export BASE_RUNS=/home/zolotovskijal/baselines
export AR_CKPT=$BASE_RUNS/ar_text8_seed12345/checkpoints/last.pt
export MDLM_CKPT=$BASE_RUNS/mdlm_text8_seed12345/checkpoints/last.pt
export BD3_CKPT=$BASE_RUNS/bd3lm_b16_text8_seed12345/checkpoints/last.pt
export M1_CKPT=$BASE_RUNS/NEW/tinystories_a100/checkpoints/best.ckpt
```

Одноразовая проверка одного чекпоинта без shell-обвязки:

```bash
export BASELINES_CODE=$PWD/baselines
export TEXT8_DATA_DIR=$PWD/data/text8

python -m eval.run_baseline_eval \
  --checkpoint /path/to/ar_text8_seed12345/checkpoints/last.pt \
  --exp-name smoke_ar_L256_seed0 \
  --n-samples 8 --batch-size 8 --length 256 --seed 0 \
  --judge --judge-model EleutherAI/gpt-j-6b
```

---

## 4. Данные

### Text8

Нужен каталог `data/text8` с бинарным корпусом (как в README / `bcfm_baselines`):

```bash
PYTHONPATH=$PWD/baselines python -m bcfm_baselines.data.prepare_text8 \
  --output-dir data/text8
```

Либо положите уже готовый Text8 (тот же формат, что у коллег / DFM).

### TinyStories (byte, vocab 256)

Скрипт `run_tinystories_nfe.sh` сам вызовет prepare, если нет `data/tinystories/meta.pkl`:

```bash
PYTHONPATH=$PWD/baselines python -m bcfm_baselines.data.prepare_tinystories \
  --output-dir data/tinystories
```

---

## 5. Запуск замеров

### Text8: AR / MDLM / BD3-LM (gen-PPL)

```bash
BASE_RUNS=$PWD/checkpoints/baselines \
EVAL_GPU=0 \
bash scripts/run_baselines_nfe.sh
```

Полезные флаги:

| Env | Дефолт | Смысл |
|---|---|---|
| `SEEDS` | `0 1 2` | сиды |
| `N_SAMPLES` | `512` | сколько строк на точку |
| `MDLM_STEPS` | `16 32 64 128` | шаги MDLM |
| `BD3_STEPS` | `1 2 4 8 16` | steps/block BD3 |
| `RUN_AR` / `RUN_MDLM` / `RUN_BD3` | `1` | выключить часть |
| `SKIP_EXISTING` | `1` | не пересчитывать готовые `metrics.json` |
| `ONLY_AGGREGATE` | `0` | только пересобрать графики |
| `BD3_FIRST_HITTING` | `0` | `1` = official first-hitting (медленно) |

Результаты: `results/base_*/metrics.json`, сводка `results/summary.csv`, фигуры `results/figures/text8/`.

### Text8: latency (batch=1, L=256)

Сначала должны существовать соответствующие `metrics.json` (после gen-PPL):

```bash
BASE_RUNS=$PWD/checkpoints/baselines \
EVAL_GPU=0 \
bash scripts/measure_baselines_latency.sh
```

Обновляет поля `sequence_latency_*` in-place и пересобирает `sequence_latency.png`.

### TinyStories: M1/M2 (CFM) + AR/MDLM

```bash
BASE_RUNS=$PWD/checkpoints/baselines \
M1_CKPT=$BASE_RUNS/tinystories_a100/checkpoints/best.ckpt \
AR_CKPT=$BASE_RUNS/ar_tinystories_seed12345/checkpoints/last.pt \
MDLM_CKPT=$BASE_RUNS/mdlm_tinystories_seed12345/checkpoints/last.pt \
EVAL_GPU=0 \
bash scripts/run_tinystories_nfe.sh
```

Latency:

```bash
# те же CKPT_* / BASE_RUNS
bash scripts/measure_tinystories_latency.sh
```

Фигуры: `results/figures/tinystories/`.

Только перерисовать графики из уже лежащих `results/*/metrics.json`:

```bash
ONLY_AGGREGATE=1 bash scripts/run_baselines_nfe.sh
# или
python -m eval.aggregate
```

---

## 6. Что лежит в `metrics.json`

Схема `network_forwards_v2` (совместима с `eval.aggregate`):

- `model`: `AR` / `MDLM` / `BD3LM` / `M1-TS` / …
- `gen_ppl`, `nfe_total`, `steps_per_block`, `block_size`
- `sequence_latency_ms` (после latency-скрипта)
- `ckpt` — абсолютный путь, с которым гоняли (информативно)

Имена экспериментов задают скрипты, например:

- `base_ar_L256_seed0`
- `base_mdlm_s32_L256_seed1`
- `base_bd3_B16_s4_nofh_L256_seed0`
- `ts_m2_kv_B16_s1_L256_seed0`

---

## 7. Обучение бейзлайнов с нуля (опционально)

Если чекпоинтов нет — тренировка из vendored-кода (см. [`baselines/README.md`](../baselines/README.md)):

```bash
cd baselines   # или PYTHONPATH=$PWD/baselines из корня CFM
python -m bcfm_baselines.train --config configs/text8/ar.yaml
# mdlm.yaml, bd3lm_b16.yaml, configs/tinystories/...
```

После обучения укажите `AR_CKPT` / … на получившийся `checkpoints/last.pt`.

---

## 8. Чеклист «новая машина»

1. Clone + env (`.venv` или mamba из `environment.yaml`).
2. Положить / распаковать чекпоинты в `checkpoints/baselines/` (или `export BASE_RUNS=...`).
3. Подготовить `data/text8` (и при необходимости TinyStories).
4. `EVAL_GPU=… bash scripts/run_baselines_nfe.sh`
5. `bash scripts/measure_baselines_latency.sh`
6. Аналогично TinyStories-скрипты.
7. `python -m eval.aggregate` → `results/figures/{text8,tinystories}/`.

Судья `EleutherAI/gpt-j-6b` качается с Hugging Face при первом `--judge` (нужен сеть / кэш HF).

---

## 9. Таблица точек с графиков

После `python -m eval.aggregate` появляются:

- `results/plot_points.md` — markdown-таблица (model, **B**, **steps/block**, NFE, gen-PPL, NLL, latency)
- `results/plot_points.csv` / `plot_points_{text8,tinystories}.csv` — те же числа для Excel/pandas

Агрегация как на фигурах: mean±std по сидам для quality, median для latency.
