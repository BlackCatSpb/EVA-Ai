def _load_legacy_format(self, index_path: str, containers_path: str, data_dir: str) -> bool:
    """
    Загружает фрактальную структуру в устаревшем формате.
    
    Args:
        index_path: Путь к файлу индекса
        containers_path: Путь к файлу контейнеров
        data_dir: Директория с данными
        
    Returns:
        bool: Успешность загрузки
    """
    try:
        logger.info("Попытка загрузки в устаревшем формате...")
        
        # Загружаем индекс
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
            logger.info(f"Индекс загружен: {len(self.index)} записей")
        else:
            logger.error(f"Файл индекса не найден: {index_path}")
            return False
        
        # Загружаем контейнеры
        if os.path.exists(containers_path):
            with open(containers_path, 'r', encoding='utf-8') as f:
                containers_data = json.load(f)
            
            # Восстанавливаем контейнеры
            self.containers = {}
            for cid, container_info in containers_data.items():
                container = FractalContainer(
                    id=cid,
                    data=np.array([]),  # Будет загружено позже по запросу
                    metadata=container_info.get('metadata', {})
                )
                self.containers[cid] = container
            
            logger.info(f"Контейнеры загружены: {len(self.containers)}")
        else:
            logger.warning(f"Файл контейнеров не найден: {containers_path}")
            self.containers = {}
        
        # Проверяем директорию данных
        if os.path.exists(data_dir):
            self.data_dir = data_dir
            logger.info(f"Директория данных установлена: {data_dir}")
        else:
            logger.warning(f"Директория данных не найдена: {data_dir}")
            self.data_dir = None
        
        logger.info("Загрузка в устаревшем формате завершена успешно")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка загрузки в устаревшем формате: {e}", exc_info=True)
        return False
