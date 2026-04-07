import sqlite3, os, json
from typing import Optional
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(name)s: %(message)s')
import torch

# 1) Настройки кэша: RAM окно 16 ГБ и I/O лимит (см. ниже)
os.environ["COGNIFLEX_CACHE_MEM_GB"] = "16.0"
# По умолчанию стараемся работать офлайн, чтобы избежать долгих/нестабильных загрузок HF
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def hotset_frac_gb(target_gb: float = 1.6) -> float:
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_bytes = float(props.total_memory)
            return max(0.01, min(0.95, (target_gb * (1024**3)) / total_bytes))
    except Exception:
        pass
    return 0.0

MB_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CFG = {
    "cache_io_rate_limit_bps": 64 * 1024 * 1024 * 8,  # 64 МБ/с -> 512 Мбит/с
    "cache_io_burst_factor": 1.2,
    # Макроблоки и горячее окно на GPU ~1.6 ГБ (через долю VRAM)
    "nlp_macroblocks_enabled": True,
    "nlp_macroblocks_device": MB_DEVICE,
    "nlp_hotset_enabled": True,
    "nlp_hotset_target_vram_frac": hotset_frac_gb(1.6),
}


def open_models_db(path: str = r"core/cogniflex_cache/models/models.db") -> Optional[sqlite3.Connection]:
    try:
        if os.path.exists(path):
            return sqlite3.connect(path)
    except Exception:
        pass
    return None


def main():
    # 2) Инициализируем CoreBrain
    try:
        from eva_ai.core.core_brain import CoreBrain
    except Exception as e:
        print("[ERR] Не удалось импортировать CoreBrain:", e)
        return

    brain = CoreBrain(config=CFG)

    # 3) Регистрируем и загружаем русскоязычную GPT (ruGPT3 XL)
    try:
        from eva_ai.mlearning.model_manager import ModelManager
    except Exception as e:
        print("[ERR] Не удалось импортировать ModelManager:", e)
        return

    # Позволяем ModelManager выбрать вложенный каталог по умолчанию (cogniflex/mlearning/cogniflex_models)
    mm = ModelManager(brain=brain, autoload=False)

    # Вспомогательная функция загрузки и выбора модели (синхронно)
    def try_select(model_id: str, model_path: str, model_type: str, priority: int = 90) -> Optional[str]:
        try:
            meta = mm.register_model(model_id=model_id, model_path=model_path, model_type=model_type, priority=priority)
            # Используем канонический ID, который вернула регистрация
            can_id = getattr(meta, 'id', None) or model_id
            # Синхронная загрузка для детерминированного smoke-теста
            mm.load_model(can_id, low_impact=False)
            inst = mm.get_model(can_id)
            if inst:
                print(f"[OK] Загружена модель: id={can_id} type={inst.metadata.model_type} path={inst.metadata.model_path}")
                return can_id
        except Exception as e:
            print(f"[WARN] Не удалось загрузить {model_id} ({model_path}):", e)
        return None

    def has_valid_config(dir_path: str) -> bool:
        try:
            cfg_path = os.path.join(dir_path, "config.json")
            if not os.path.isfile(cfg_path):
                return False
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return bool(cfg.get("model_type"))
        except Exception:
            return False

    selected_id = None

    # Сначала: явный путь из переменной окружения (если задан)
    explicit_dir = os.environ.get("COGNIFLEX_MODEL_DIR")
    if explicit_dir:
        try:
            if os.path.isdir(explicit_dir) and has_valid_config(explicit_dir):
                mtype_env = os.environ.get("COGNIFLEX_MODEL_TYPE", "").strip().lower()
                if not mtype_env:
                    lower = os.path.basename(explicit_dir).lower()
                    if "qwen" in lower:
                        mtype_env = "qwen"
                    elif "rugpt" in lower or "gpt" in lower:
                        mtype_env = "rugpt3" if "rugpt" in lower else "gpt2"
                    elif "bart" in lower:
                        mtype_env = "bart"
                    elif "t5" in lower:
                        mtype_env = "t5"
                    else:
                        mtype_env = "transformer"
                selected_id = try_select("explicit_local", explicit_dir, mtype_env or "transformer", priority=100)
            else:
                print("[WARN] COGNIFLEX_MODEL_DIR задан, но не содержит валидной модели:", explicit_dir)
        except Exception as e:
            print("[WARN] Ошибка при использовании COGNIFLEX_MODEL_DIR:", e)

    # Базовый каталог локальных моделей: вложенный рядом с ML-модулями
    base_models_dir = os.path.join(os.path.dirname(__file__), "cogniflex", "mlearning", "cogniflex_models")
    # Сканируем вложенную директорию и ищем любую валидную локальную модель
    if not selected_id and os.path.isdir(base_models_dir):
        try:
            for entry in sorted(os.listdir(base_models_dir)):
                local_dir = os.path.join(base_models_dir, entry)
                if os.path.isdir(local_dir) and has_valid_config(local_dir):
                    # Грубое определение типа по имени папки
                    lower = entry.lower()
                    if "qwen" in lower:
                        mtype = "qwen"
                    elif "rugpt" in lower or "gpt" in lower:
                        mtype = "rugpt3" if "rugpt" in lower else "gpt2"
                    elif "bart" in lower:
                        mtype = "bart"
                    elif "t5" in lower:
                        mtype = "t5"
                    else:
                        mtype = "transformer"
                    selected_id = try_select(f"{entry}_local", local_dir, mtype, priority=95)
                    if selected_id:
                        break
        except Exception as e:
            print("[WARN] Не удалось просканировать локальные модели:", e)

    # Попытка 3: HF ruGPT3 Large (только если офлайн-флаги выключены)
    if not selected_id and os.environ.get("TRANSFORMERS_OFFLINE", "1") != "1" and os.environ.get("HF_HUB_OFFLINE", "1") != "1":
        hf_id = "sberbank-ai/rugpt3large_based_on_gpt2"
        selected_id = try_select(hf_id, hf_id, "rugpt3", priority=90)

    if not selected_id:
        print("[ERR] Не удалось найти рабочую локальную модель и HF недоступен/отключен.")
        print("     Проверьте содержимое вложенной папки: cogniflex/mlearning/cogniflex_models/")
        print("     Требуются по крайней мере config.json и токенизатор (tokenizer.json или vocab.json + merges.txt).")
        return

    # 3.1) Регистрируем alias 'default_text_gen' на выбранную модель, чтобы использовать её по умолчанию
    if selected_id:
        inst = mm.get_model(selected_id)
        if inst and inst.metadata:
            # Подбираем теги под тип
            base_tags = ["default", "generation"]
            if "qwen" in inst.metadata.model_type.lower():
                base_tags += ["qwen", "russian"]
            elif "rugpt" in inst.metadata.model_type.lower() or "gpt" in inst.metadata.model_type.lower():
                base_tags += ["rugpt3", "russian"]
            try:
                alias_name = (getattr(inst.metadata, 'name', None) or selected_id) + " (default)"
            except Exception:
                alias_name = selected_id + " (default)"
            mm.register_model(
                model_id="default_text_gen",
                model_path=inst.metadata.model_path,
                model_type=inst.metadata.model_type,
                priority=100,
                name=alias_name,
                tags=base_tags,
            )
            # Загружаем alias, чтобы удостовериться, что всё работает под дефолтным ID
            try:
                mm.load_model("default_text_gen", low_impact=False)
                if mm.get_model("default_text_gen"):
                    print("[OK] Установлен alias default_text_gen ->", inst.metadata.model_path)
                else:
                    print("[WARN] Alias default_text_gen зарегистрирован, но не загружен")
            except Exception as e:
                print("[WARN] Не удалось загрузить alias default_text_gen:", e)

    # 4) Печатаем краткую сводку по БД моделей (если есть)
    con = open_models_db()
    if con is not None:
        try:
            cur = con.cursor()
            total = cur.execute("SELECT COUNT(*) FROM models").fetchone()[0]
            print("Models in DB:", total)
            sample = cur.execute("SELECT id,name,model_path,model_type,priority FROM models ORDER BY last_updated DESC LIMIT 5").fetchall()
            for r in sample:
                print("  -", r)
        except Exception as e:
            print("[WARN] Не удалось прочитать БД моделей:", e)
        finally:
            con.close()
    else:
        print("[INFO] База моделей не найдена (будет создана позже системой при необходимости)")


if __name__ == "__main__":
    main()
