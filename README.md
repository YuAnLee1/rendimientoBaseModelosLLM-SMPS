# Rendimiento base de LLMs en análisis de sentimiento del español peruano (SMPS)

Código fuente del primer objetivo específico de la tesis **"Evaluación y Fine-Tuning de LLMs mediante Quantized Low-Rank Adaptation para el análisis de sentimientos en el español peruano"** (Yu An Lee Quezada Chang, asesor: Mg. Luis Vives Garnique, Pontificia Universidad Católica del Perú).

El repositorio contiene:

1. La preparación del dataset SocialMediaPeruvianSentiment (SMPS): exploración, limpieza y partición.
2. La evaluación de cinco prompts zero-shot sobre el conjunto de validación, para seis LLMs generativos en su forma base (sin ajuste fino).
3. La evaluación única del prompt seleccionado para cada modelo sobre el conjunto de test (rendimiento base).

---

## Estructura del repositorio

```
.
├── ExploracionHF.ipynb                  # Exploración del dataset original en Hugging Face
├── PreprocesamientoHF.ipynb             # Limpieza → smps_clean_pool.csv
├── Estratificacion.ipynb                # Partición 80/10/10 → train/val/test_final.csv
├── smps_hf_raw_pool.csv                 # Dataset original unificado (14,589 registros)
├── smps_clean_pool.csv                  # Dataset limpio (14,544 registros)
├── train_final.csv                      # 11,635 registros
├── val_final.csv                        # 1,454 registros
├── test_final.csv                       # 1,455 registros
├── requirements.txt
│
├── EvaluacionLLMs - Forma base - Pruebas Prompts/      # Etapa 1: selección de prompt (validación)
│   ├── plantilla_seleccion_prompt_validacion.ipynb
│   └── <Modelo>/
│       ├── <Modelo>.ipynb                              # evalúa los 5 prompts
│       ├── val_<modelo>_prompt{1..5}.csv               # predicciones por ejemplo
│       ├── comparacion_prompts_<modelo>_validation.csv # métricas por prompt
│       ├── log_experimentos_<modelo>_validation.csv    # tiempos, memoria, versiones
│       └── matrices_confusion_<modelo>_validation.png
│
└── EvaluacionLLMs - Conjunto Test - Mejores Prompts/   # Etapa 2: rendimiento base (test)
    └── <Modelo>/
        ├── evaluacion_<Modelo>_test.ipynb
        ├── test_<modelo>_prompt<N>.csv                 # predicciones por ejemplo
        ├── resultado_test_<modelo>.csv                 # métricas finales
        ├── log_test_<modelo>.csv
        ├── metadata_carga_test_<modelo>.csv            # GPU y versiones de librerías
        └── matriz_confusion_test_<modelo>.png
```

Los archivos de predicciones contienen, para cada ejemplo: `idx_original`, `text_clean`, `label_name` (etiqueta real), `prompt_id`, `respuesta_cruda` (texto generado por el modelo) y `prediccion` (etiqueta extraída).

---

## Datos

| Recurso | Repositorio en Hugging Face | Revisión (commit) |
|---|---|---|
| SocialMediaPeruvianSentiment (Calizaya et al., 2026) | [`pyupeu/social-media-peruvian-sentiment`](https://huggingface.co/datasets/pyupeu/social-media-peruvian-sentiment) | `29c842ea9e189e5865925ac0ef6cb01b1388f4b6` |

Los archivos de datos no cambian desde la revisión `c440ef6` (18/08/2024). El commit posterior solo modifica el README del dataset.

**Limpieza** (`PreprocesamientoHF.ipynb`), aplicada sobre los tres splits originales unificados (14,589 registros):

1. Eliminación de los textos con etiquetas contradictorias: 3 textos, 6 registros.
2. Eliminación de duplicados, conservando una ocurrencia: 39 registros.
3. Eliminación de URLs y del símbolo `#` de los hashtags (se conserva la palabra).
4. Anonimización de menciones a usuarios como `@usuario`. El `@` usado dentro de palabras (por ejemplo, `to@ll@`) no se modifica.
5. Se conservan mayúsculas, tildes y emojis.

Resultado: **14,544 registros** (negativo 45.63%, positivo 32.54%, neutral 21.84%).

**Partición** (`Estratificacion.ipynb`): 80/10/10, estratificada por clase, con `random_state=42`.

| Split | Negativo | Positivo | Neutral | Total |
|---|---|---|---|---|
| Train | 5,309 | 3,785 | 2,541 | 11,635 |
| Validación | 663 | 473 | 318 | 1,454 |
| Test | 664 | 474 | 317 | 1,455 |

> **Diferencias con el trabajo de referencia.** Calizaya et al. (2026) obtuvieron sus resultados con una versión anterior del dataset (11,835 registros; revisión `4e79711` de Hugging Face, idéntica a los CSV de su repositorio) y con una partición efectiva de ≈64/20/16. Además, aplicaron la normalización propia de RoBERTuito (minúsculas, sin tildes, emojis convertidos a texto). Por ello, toda comparación con RoBERTuito.pe es referencial.

---

## Modelos

| Modelo | Repositorio en Hugging Face | Revisión (commit) | Tipo de carga |
|---|---|---|---|
| Qwen3.5-9B | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | multimodal |
| RigoChat-7b-v2 | [`IIC/RigoChat-7b-v2`](https://huggingface.co/IIC/RigoChat-7b-v2) | `666a48562f47b0cc09aca23f1ae8df3d0b3ac25c` | causal_lm |
| Salamandra-7B-Instruct-2606 | [`BSC-LT/salamandra-7b-instruct-2606`](https://huggingface.co/BSC-LT/salamandra-7b-instruct-2606) | `91055ea2c907ddcf28209d2cb4e078a093ea44d8` | causal_lm |
| EuroLLM-9B-Instruct | [`utter-project/EuroLLM-9B-Instruct`](https://huggingface.co/utter-project/EuroLLM-9B-Instruct) | `f7ae2bc3bcbb538c0b93fa6cfbf388d9898b1ace` | causal_lm |
| Gemma 4-12B-it | [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | multimodal |
| Llama 3.1-8B-Instruct | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | `0e9e39f249a16976918f6564b8830bc894c89659` | causal_lm |

Los cuadernos cargan la rama `main`. La tabla indica la revisión vigente en la fecha de ejecución de los experimentos (julio–agosto de 2026). En Salamandra, los commits posteriores a `f7f7ce4` solo modifican el README.

Para fijar una revisión al reproducir:

```python
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision="<commit>", ...)
```

Todos los modelos se cargan en `bfloat16` con `device_map="auto"`. Llama 3.1 y Gemma 4 son modelos restringidos (*gated*): hay que aceptar su licencia en Hugging Face antes de descargarlos.

---

## Protocolo de evaluación

| Parámetro | Valor |
|---|---|
| Estrategia | Zero-shot, 5 prompts (roles `system`/`user` en los prompts 4 y 5) |
| Plantilla | `apply_chat_template(..., add_generation_prompt=True)`; `enable_thinking=False` en los modelos que lo soportan (Qwen3.5, Gemma 4) |
| Decodificación | Greedy (`do_sample=False`), `max_new_tokens=10` |
| Semilla | 42 (`random`, `numpy`, `torch`, `torch.cuda`), fijada antes de cada corrida |
| Extracción de etiqueta | Minúsculas y sin tildes; búsqueda de `negativ` / `neutral` / `positiv`. Si se encuentra ninguna o más de una, la respuesta es `invalido` y cuenta como error |
| Métricas | Accuracy, precision, recall y F1 por clase, macro F1, % de respuestas inválidas (`scikit-learn`, `zero_division=0`) |
| Selección | Mayor macro F1 en validación |
| Test | Una sola evaluación por modelo, con el prompt seleccionado |

Los cinco prompts están definidos en el diccionario `PROMPTS` de cada cuaderno (Tabla 6 de la tesis).

---

## Cómo reproducir

**Requisitos:** Google Colab con GPU. Los modelos de 7–9B caben en una NVIDIA L4 (24 GB). Gemma 4-12B-it requiere una GPU con más memoria (se usó una A100).

1. **Token de Hugging Face.** En Colab, crea un secreto llamado `HF_TOKEN` (menú *Secrets*). Los cuadernos lo leen con `userdata.get("HF_TOKEN")`.
2. **Datos.** Usa los CSV incluidos (`val_final.csv`, `test_final.csv`) o regenéralos ejecutando `PreprocesamientoHF.ipynb` y luego `Estratificacion.ipynb`.
3. **Selección de prompt (validación).**
   - Abre el cuaderno del modelo en `EvaluacionLLMs - Forma base - Pruebas Prompts/<Modelo>/`, o la plantilla.
   - Sube `val_final.csv` a `/content/`.
   - Revisa la sección de configuración (`MODEL_ID`, `MODEL_TYPE`, `SUPPORTS_ENABLE_THINKING`) y ejecuta todas las celdas.
4. **Rendimiento base (test).**
   - Abre `evaluacion_<Modelo>_test.ipynb`.
   - Sube `test_final.csv` a `/content/`.
   - Ajusta `PROMPT_ID` al prompt ganador y ejecuta.
5. Las salidas (CSV y PNG) se guardan en el directorio de trabajo de Colab.

**Dependencias:** ver `requirements.txt`. En Colab solo es necesario instalar la versión indicada de `transformers`, más `accelerate`, y `kernels` en el caso de Qwen. El resto viene preinstalado.

---

## Entorno registrado

Versiones tomadas de los archivos `metadata_carga_test_*.csv`:

| Modelo | GPU | transformers | torch |
|---|---|---|---|
| Qwen3.5-9B | NVIDIA L4 | 5.10.4 | 2.11.0+cu128 |
| RigoChat-7b-v2 | NVIDIA L4 | 5.13.1 | 2.11.0+cu128 |
| Salamandra-7B-Instruct-2606 | NVIDIA L4 | 5.13.1 | 2.11.0+cu128 |
| EuroLLM-9B-Instruct | NVIDIA L4 | 5.13.1 | 2.11.0+cu128 |
| Gemma 4-12B-it | NVIDIA A100 | 5.13.1 | 2.11.0+cu128 |
| Llama 3.1-8B-Instruct | NVIDIA L4 | 5.13.1 | 2.11.0+cu128 |

> **Nota de reproducibilidad.** Aunque la decodificación es determinista, cambiar la versión de las librerías o la GPU puede alterar algunas predicciones límite, por diferencias numéricas en `bfloat16`.

---

## Resultados (macro F1)

| Modelo | Prompt seleccionado | Validación | Test |
|---|---|---|---|
| Qwen3.5-9B | 4 | 0.6075 | 0.6079 |
| RigoChat-7b-v2 | 4 | 0.5750 | 0.5998 |
| Salamandra-7B-Instruct-2606 | 2 | 0.6808 | 0.6686 |
| EuroLLM-9B-Instruct | 3 | 0.5107 | 0.5036 |
| Gemma 4-12B-it | 1 | 0.6403 | 0.6329 |
| Llama 3.1-8B-Instruct | 1 | 0.5215 | 0.5283 |

Las métricas completas por clase están en `comparacion_prompts_*_validation.csv` y `resultado_test_*.csv`.

---

## Referencias

- Calizaya-Milla, S. E., Santos, J., & Huanca Torres, F. A. (2026). Fine-tuning monolingual pre-trained BERT models for sentiment analysis in Peruvian slang contexts. *Journal of Applied Artificial Intelligence*. https://doi.org/10.1080/08839514.2026.2641381
- Dataset SMPS: https://huggingface.co/datasets/pyupeu/social-media-peruvian-sentiment (DOI 10.57967/hf/2669)
