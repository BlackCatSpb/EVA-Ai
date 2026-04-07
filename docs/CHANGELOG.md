# Changelog

## [0.2.0] - 2026-04-07

### Added
- **ConceptMiner** - модуль автономного концептуального вывода (850+ строк)
- **ConceptDialogIntegrator** - интегратор самодиалогов
- EventTypes: CONCEPT_MINING_START, CONCEPT_MINING_COMPLETE, MEMORY_CLUSTERING_COMPLETE
- Алгоритм детекции семантических лакун (centroid distance)
- Генерация гипотез через LLM с валидацией NLI-когерентности
- Жизненный цикл концептов: provisional → confirmed → stable → archived

### Changed
- max_tokens: Model A (1024→2048), Model B (512→1024)
- Model B: убран timeout, добавлена строгая проверка русского языка
- Исправлена передача контекста от Model A к Model B
- Переименование директории eva → eva_ai для избежания конфликтов

---

## [0.1.2] - 2026-03-17

### Changed
- UI: Заменены кнопки навигации на ttk.Notebook (настоящие вкладки)
- Исправлена проблема "прыгающих" кнопок при переключении модулей

---

## [0.1.1] - 2026-03-17

### Fixed
- Fixed bare except in 8 files
- Replaced print() with logger in 7+ files
- Removed backup file core_gui.py.bak

### Added
- Agent Swarm coordination system
- Documentation: AGENT_SWARM.md, TECHNICAL_DEBT.md

---

## [0.1.0] - 2026-03-16

### Added
- Initial project structure
- Core modules: adaptation, contradiction, generation, gui, knowledge, learning, memory, mlearning
