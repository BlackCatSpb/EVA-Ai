"""
EvaContainer - Универсальный загрузчик .eva файлов

Поддерживает:
- Загрузку директории .eva/
- Загрузку упакованного .eva файла
- Валидацию структуры
- Версионирование формата
- Кэширование распакованных файлов
- Обратную совместимость
"""

import os
import json
import shutil
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("eumi.container")


def compute_file_hash(file_path: Path) -> str:
    """Вычислить SHA256 хеш файла."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


@dataclass
class EvaStructure:
    """Структура .eva директории."""
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    backends_path: Path
    tokenizer_path: Path
    embeddings_path: Path
    knowledge_path: Path
    graph_path: Optional[Path]
    
    @property
    def version(self) -> str:
        """Версия формата .eva."""
        return self.config.get("eva_format_version", "1.0.0")
    
    @property
    def model_type(self) -> str:
        """Тип модели (gguf, transformers, onnx)."""
        return self.config.get("model_type", "gguf")
    
    @property
    def supports_virtual_tokens(self) -> bool:
        """Поддерживает ли виртуальные токены."""
        return self.config.get("virtual_tokens", {}).get("enabled", False)


class EvaContainer:
    """
    Контейнер для загрузки и управления .eva моделями.
    
    Поддерживает два режима:
    1. Директория .eva/ - распакованная структура
    2. Файл .eva - упакованная структура (tar/zip)
    
    Attributes:
        path: Путь к .eva файлу или директории
        temp_dir: Временная директория (для упакованных файлов)
        structure: Структура загруженной модели
        is_mounted: Флаг, что контейнер смонтирован
        cache_dir: Директория для кэширования
    """
    
    SUPPORTED_VERSIONS = ["1.0.0", "1.1.0"]
    REQUIRED_FILES = ["config.json", "metadata.json"]
    
    def __init__(
        self,
        path: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True
    ):
        """
        Инициализация контейнера.
        
        Args:
            path: Путь к .eva файлу или директории
            cache_dir: Директория для кэширования (default: ~/.cache/eva)
            use_cache: Использовать ли кэширование
        """
        self.path = Path(path)
        self.temp_dir: Optional[Path] = None
        self.structure: Optional[EvaStructure] = None
        self.is_mounted = False
        self._model_cache: Dict[str, Any] = {}
        self._use_cache = use_cache
        
        # Настройка директории кэша
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "eva"
        
        self._cache_key: Optional[str] = None
        self._cache_manifest_path: Optional[Path] = None
        
        logger.info(f"EvaContainer initialized: {self.path}")
    
    def mount(self, force_extract: bool = False) -> "EvaContainer":
        """
        Смонтировать контейнер.
        
        Для упакованных .eva файлов:
        - Проверяет кэш
        - Распаковывает содержимое (если нужно)
        
        Для директорий:
        - Проверяет структуру
        
        Args:
            force_extract: Принудительно распаковать, игнорируя кэш
            
        Returns:
            self для chaining
            
        Raises:
            FileNotFoundError: Если путь не существует
            ValueError: Если структура невалидна
        """
        if self.is_mounted:
            logger.warning("Container already mounted")
            return self
        
        if not self.path.exists():
            raise FileNotFoundError(f".eva not found: {self.path}")
        
        # Определяем тип (файл или директория)
        if self.path.is_file():
            if self._use_cache and not force_extract:
                # Пробуем использовать кэш
                cache_path = self._try_cache()
                if cache_path:
                    self.temp_dir = cache_path
                    logger.info(f"Using cached container: {cache_path}")
                else:
                    # Распаковываем в кэш
                    self._mount_to_cache()
            else:
                # Распаковываем во временную директорию
                self._mount_from_file()
        else:
            self._mount_from_directory()
        
        # Валидация структуры
        self._validate_structure()
        
        # Загрузка структуры
        self._load_structure()
        
        self.is_mounted = True
        logger.info(f"Container mounted successfully (version: {self.structure.version})")
        
        return self
    
    def unmount(self) -> None:
        """
        Размонтировать контейнер.
        
        Очищает временные файлы для упакованных .eva.
        """
        if not self.is_mounted:
            return
        
        # Очистка кэша
        self._model_cache.clear()
        
        # Удаление временной директории
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"Temp directory cleaned: {self.temp_dir}")
        
        self.temp_dir = None
        self.structure = None
        self.is_mounted = False
        
        logger.info("Container unmounted")
    
    def get_model_path(self, model_type: str = "condensed") -> Path:
        """
        Получить путь к модели определённого типа.
        
        Args:
            model_type: Тип модели (condensed, extended, coder)
            
        Returns:
            Путь к файлу модели
            
        Raises:
            ValueError: Если модель не найдена
        """
        self._ensure_mounted()
        
        backend_path = self.structure.backends_path
        
        # Поиск модели в разных форматах
        extensions = [".gguf", ".bin", ".onnx", ".pt", ".safetensors"]
        
        for ext in extensions:
            model_path = backend_path / f"{model_type}{ext}"
            if model_path.exists():
                return model_path
        
        # Проверка в поддиректориях
        for subdir in ["gguf", "transformers", "onnx"]:
            subpath = backend_path / subdir / f"{model_type}.gguf"
            if subpath.exists():
                return subpath
        
        raise ValueError(f"Model not found: {model_type} in {backend_path}")
    
    def get_graph_path(self) -> Optional[Path]:
        """
        Получить путь к графу знаний.
        
        Returns:
            Путь к SQLite файлу или None
        """
        self._ensure_mounted()
        return self.structure.graph_path
    
    def get_tokenizer_config(self) -> Dict[str, Any]:
        """
        Получить конфигурацию токенизатора.
        
        Returns:
            Словарь с конфигурацией
        """
        self._ensure_mounted()
        
        config_path = self.structure.tokenizer_path / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        return {}
    
    def get_backend_config(self, backend_type: str) -> Dict[str, Any]:
        """
        Получить конфигурацию бэкенда.
        
        Args:
            backend_type: Тип бэкенда (gguf, transformers, onnx)
            
        Returns:
            Словарь с конфигурацией
        """
        self._ensure_mounted()
        
        config_path = self.structure.backends_path / backend_type / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        return {}
    
    def _mount_from_file(self) -> None:
        """Смонтировать из упакованного файла .eva."""
        import tarfile
        
        # Создаём временную директорию
        self.temp_dir = Path(tempfile.mkdtemp(prefix="eva_"))
        
        # Распаковываем
        if tarfile.is_tarfile(self.path):
            with tarfile.open(self.path, "r:*") as tar:
                tar.extractall(self.temp_dir)
            logger.info(f"Extracted tar archive to: {self.temp_dir}")
        else:
            # Пробуем как zip
            import zipfile
            if zipfile.is_zipfile(self.path):
                with zipfile.ZipFile(self.path, "r") as zf:
                    zf.extractall(self.temp_dir)
                logger.info(f"Extracted zip archive to: {self.temp_dir}")
            else:
                raise ValueError(f"Unknown archive format: {self.path}")
        
        # Находим корневую директорию (может быть вложена)
        root = self._find_root_directory(self.temp_dir)
        self.temp_dir = root
    
    def _mount_from_directory(self) -> None:
        """Смонтировать из директории."""
        self.temp_dir = self.path
        logger.info(f"Using directory: {self.temp_dir}")
    
    def _find_root_directory(self, base: Path) -> Path:
        """
        Найти корневую директорию с config.json.
        
        Архив может содержать вложенную директорию.
        """
        # Проверяем текущий уровень
        if (base / "config.json").exists():
            return base
        
        # Ищем на уровень глубже
        for subdir in base.iterdir():
            if subdir.is_dir() and (subdir / "config.json").exists():
                return subdir
        
        raise ValueError("config.json not found in archive")
    
    def _validate_structure(self) -> None:
        """Валидировать структуру .eva."""
        if self.temp_dir is None:
            raise RuntimeError("Container not mounted")
        
        # Проверка обязательных файлов
        for required in self.REQUIRED_FILES:
            if not (self.temp_dir / required).exists():
                raise ValueError(f"Required file missing: {required}")
        
        # Проверка директорий
        required_dirs = ["backends", "tokenizer", "knowledge"]
        for dirname in required_dirs:
            if not (self.temp_dir / dirname).exists():
                logger.warning(f"Recommended directory missing: {dirname}")
    
    def _load_structure(self) -> None:
        """Загрузить структуру контейнера."""
        # Загрузка конфигов
        with open(self.temp_dir / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        with open(self.temp_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Проверка версии
        version = config.get("eva_format_version", "1.0.0")
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported .eva version: {version}")
        
        # Поиск графа
        graph_path = None
        knowledge_path = self.temp_dir / "knowledge"
        if knowledge_path.exists():
            potential_graphs = [
                knowledge_path / "graph.sqlite",
                knowledge_path / "graph.db",
                knowledge_path / "knowledge.sqlite"
            ]
            for gp in potential_graphs:
                if gp.exists():
                    graph_path = gp
                    break
        
        self.structure = EvaStructure(
            config=config,
            metadata=metadata,
            backends_path=self.temp_dir / "backends",
            tokenizer_path=self.temp_dir / "tokenizer",
            embeddings_path=self.temp_dir / "embeddings" if (self.temp_dir / "embeddings").exists() else knowledge_path,
            knowledge_path=knowledge_path,
            graph_path=graph_path
        )
    
    def _ensure_mounted(self) -> None:
        """Проверить, что контейнер смонтирован."""
        if not self.is_mounted:
            raise RuntimeError("Container not mounted. Call mount() first.")
    
    def __enter__(self) -> "EvaContainer":
        """Context manager entry."""
        return self.mount()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.unmount()
    
    def __repr__(self) -> str:
        """Строковое представление."""
        status = "mounted" if self.is_mounted else "unmounted"
        version = self.structure.version if self.structure else "unknown"
        cached = "cached" if self._cache_key else "uncached"
        return f"EvaContainer(path={self.path}, status={status}, version={version}, {cached})"
    
    # ==================== Cache Methods ====================
    
    def _compute_cache_key(self) -> str:
        """Вычислить ключ кэша на основе хеша файла."""
        if self.path.is_file():
            # Для файлов: SHA256 первых 1MB + размер
            with open(self.path, "rb") as f:
                chunk = f.read(1024 * 1024)  # 1 MB
                file_hash = hashlib.sha256(chunk).hexdigest()[:16]
            size = self.path.stat().st_size
            return f"{file_hash}_{size}"
        else:
            # Для директорий: хеш от пути + mtime
            stat = self.path.stat()
            key = f"{self.path.name}_{int(stat.st_mtime)}"
            return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def _try_cache(self) -> Optional[Path]:
        """Попробовать использовать кэшированную версию."""
        self._cache_key = self._compute_cache_key()
        cache_subdir = self.cache_dir / f"eva_{self._cache_key}"
        manifest_path = cache_subdir / "manifest.json"
        
        if not manifest_path.exists():
            return None
        
        # Проверяем валидность кэша
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                cached_manifest = json.load(f)
            
            if self._validate_cache(cache_subdir, cached_manifest):
                self._cache_manifest_path = manifest_path
                return cache_subdir
        except Exception as e:
            logger.warning(f"Cache validation failed: {e}")
        
        # Кэш невалиден - удаляем
        shutil.rmtree(cache_subdir, ignore_errors=True)
        return None
    
    def _validate_cache(self, cache_dir: Path, manifest: Dict) -> bool:
        """Проверить целостность кэша."""
        files = manifest.get("files", [])
        
        for file_info in files:
            file_path = cache_dir / file_info["name"]
            if not file_path.exists():
                return False
            
            # Проверяем хеш если есть
            if "sha256" in file_info:
                actual_hash = compute_file_hash(file_path)
                if actual_hash != file_info["sha256"]:
                    return False
        
        return True
    
    def _mount_to_cache(self) -> None:
        """Распаковать в директорию кэша."""
        self._cache_key = self._compute_cache_key()
        cache_subdir = self.cache_dir / f"eva_{self._cache_key}"
        cache_subdir.mkdir(parents=True, exist_ok=True)
        
        # Распаковываем
        import tarfile
        import zipfile
        
        if tarfile.is_tarfile(self.path):
            with tarfile.open(self.path, "r:*") as tar:
                tar.extractall(cache_subdir)
        elif zipfile.is_zipfile(self.path):
            with zipfile.ZipFile(self.path, "r") as zf:
                zf.extractall(cache_subdir)
        else:
            raise ValueError(f"Unknown archive format: {self.path}")
        
        # Создаём манифест кэша
        manifest = self._create_cache_manifest(cache_subdir)
        manifest_path = cache_subdir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        
        self._cache_manifest_path = manifest_path
        self.temp_dir = self._find_root_directory(cache_subdir)
        
        logger.info(f"Extracted to cache: {cache_subdir}")
    
    def _create_cache_manifest(self, cache_dir: Path) -> Dict:
        """Создать манифест для кэша."""
        files = []
        
        for file_path in cache_dir.rglob("*"):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(cache_dir))
                files.append({
                    "name": rel_path,
                    "sha256": compute_file_hash(file_path)
                })
        
        return {
            "cache_version": "1.0",
            "original_path": str(self.path),
            "files": files,
            "created_at": datetime.now().isoformat()
        }
    
    def cleanup_cache(self) -> int:
        """Очистить кэш этого контейнера."""
        if self._cache_key:
            cache_subdir = self.cache_dir / f"eva_{self._cache_key}"
            if cache_subdir.exists():
                size = sum(f.stat().st_size for f in cache_subdir.rglob("*") if f.is_file())
                shutil.rmtree(cache_subdir, ignore_errors=True)
                logger.info(f"Cleaned cache: {cache_subdir} ({size / 1024 / 1024:.1f} MB)")
                return size
        return 0
    
    @classmethod
    def cleanup_all_cache(cls, cache_dir: Optional[Path] = None) -> int:
        """Очистить весь кэш."""
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "eva"
        
        if not cache_dir.exists():
            return 0
        
        total_size = 0
        for subdir in cache_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith("eva_"):
                size = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file())
                total_size += size
                shutil.rmtree(subdir, ignore_errors=True)
        
        logger.info(f"Cleaned all cache: {total_size / 1024 / 1024:.1f} MB")
        return total_size
