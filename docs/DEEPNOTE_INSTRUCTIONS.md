# ИНСТРУКЦИЯ: Завершение гибридной архитектуры через Deepnote

## Workspace
**Deepnote:** https://deepnote.com/workspace/eva-342e-66caa06a-2119-46d5-a144-118e0c792b29/home

## Статус системы
✅ `HybridLayerPipeline` работает (тест проходит)
✅ `HybridLayerProcessor/GNN/KCA/SRG` инициализированы
❌ `layer_capture_used: False` (нужен Deepnote/Kaggle с >16GB RAM)
❌ Гибридная модель OpenVINO повреждена (`model.bin` = corrupted `.npz`)

## Что нужно сделать в Deepnote

### 1. Импортируйте notebook
- Скачайте `deepnote_hybrid_fix.ipynb` из репозитория
- Или скопируйте содержимое файла в новый notebook в Deepnote

### 2. Загрузите файлы (8GB)
В Deepnote нажмите **+** (справа сверху) → **Upload files**:
- `qwenlayermodel.pt` (8GB) — лежит в `C:\Users\black\Desktop\`

### 3. Запустите ячейки по порядку
Notebook сделает:
1. Установит зависимости (torch, transformers, openvino)
2. Загрузит 8GB checkpoint (займет ~1-2 мин)
3. Создаст корректный `hybrid_weights.npz`
4. Создаст `model.xml` для OpenVINO
5. Создаст `model.bin` (raw float32)
6. Упакует всё в `hybrid_openvino_fixed.zip`

### 4. Скачайте результат
В панели **Files** справа:
- Найдите `hybrid_openvino_fixed.zip`
- Нажмите → **Download**

### 5. Распакуйте на локальной машине
```
C:\Users\black\OneDrive\Desktop\EVA-Ai\models\hybrid_openvino\
├── model.xml
├── model.bin (исправленный)
└── hybrid_weights.npz (исправленный)
```

### 6. Проверьте
```powershell
cd C:\Users\black\OneDrive\Desktop\EVA-Ai
python test_hybrid_integration.py
```

**Ожидаемый результат:**
```
HYBRID PIPELINE TEST PASSED!
Metadata: {'mode': 'hybrid', 'layer_capture_used': False, ...}
```

## Файлы в репозитории
- `deepnote_hybrid_fix.ipynb` — основной notebook
- `kaggle_hybrid_fix.ipynb` — альтернатива для Kaggle
- `test_hybrid_integration.py` — тест системы

## Коммиты
- `f92e68e` — Deepnote notebook
- `d001662` — Kaggle notebook  
- `c79fade` — Hybrid pipeline fix

## Репозиторий
https://github.com/BlackCatSpb/EVA-Ai

---
**Работа завершена.** Используйте Deepnote для финальной сборки гибридной модели.
