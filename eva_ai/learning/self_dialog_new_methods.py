    def extract_key_concepts(self, query: str) -> List[str]:
        """
        Извлекает ключевые понятия из запроса для анализа.
        
        Args:
            query: Запрос пользователя
            
        Returns:
            Список ключевых понятий
        """
        import re
        
        # Очищаем запрос
        query = query.lower().strip()
        
        # Убираем обычные приветствия и простые фразы
        simple_patterns = ['привет', 'здравствуй', 'как дела', 'спасибо', 'пока', 'да', 'нет']
        for pattern in simple_patterns:
            if query == pattern or query.startswith(pattern + ' '):
                return []
        
        # Разбиваем на слова
        words = re.findall(r'\b[а-яёa-z]{3,}\b', query)
        
        # Убираем стоп-слова
        stop_words = {'это', 'что', 'как', 'где', 'когда', 'почему', 'потому', 'для', 'от', 'до', 'при', 'над', 'под', 'между', 'среди', 'который', 'которая', 'которое', 'которые', 'этот', 'эта', 'эти', 'тот', 'та', 'те', 'свой', 'своя', 'своё', 'свои', 'весь', 'всё', 'все', 'один', 'одна', 'одно', 'одни', 'два', 'три', 'четыре', 'пять', 'либо', 'нибудь', 'только', 'уже', 'ещё', 'еще', 'быть', 'был', 'была', 'было', 'были', 'иметь', 'есть', 'быть', 'will', 'are', 'was', 'were', 'have', 'has', 'had'}
        
        concepts = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Оставляем уникальные
        concepts = list(set(concepts))
        
        return concepts[:10]  # Максимум 10 понятий
    
    def analyze_unknown_concepts(self, query: str, response: str) -> List[Dict[str, Any]]:
        """
        Анализирует какие понятия из запроса модель не знает.
        
        Args:
            query: Запрос пользователя
            response: Ответ модели
            
        Returns:
            Список неизвестных понятий с метаданными
        """
        unknown_patterns = [
            'я не знаю', 'не знаю', 'не могу ответить', 'не имею информации',
            'неизвестно', 'затрудняюсь', 'недостаточно информации', 'мне неизвестно',
            "i don't know", 'i cannot', 'i do not know', 'unable to'
        ]
        
        response_lower = response.lower()
        
        # Проверяем ответ на "не знаю"
        if not any(p in response_lower for p in unknown_patterns):
            return []
        
        # Извлекаем ключевые понятия
        concepts = self.extract_key_concepts(query)
        
        unknown_concepts = []
        for concept in concepts:
            # Проверяем, упоминается ли понятие в ответе адекватно
            if concept not in response_lower or len(response_lower) < len(concept) * 3:
                unknown_concepts.append({
                    'concept': concept,
                    'source': query,
                    'type': 'semantic_gap',
                    'priority': 0.7
                })
        
        return unknown_concepts
    
    def search_and_learn_concepts(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Выполняет поиск и обучение по неизвестным понятиям.
        
        Args:
            concepts: Список неизвестных понятий
            
        Returns:
            Результаты обучения
        """
        if not self.brain:
            return []
        
        results = []
        
        for concept_info in concepts[:5]:  # Обрабатываем макс 5 понятий
            concept = concept_info.get('concept', '')
            if not concept or len(concept) < 3:
                continue
            
            try:
                # Используем веб-поиск
                web_search = getattr(self.brain, 'web_search_engine', None)
                if web_search and hasattr(web_search, 'search'):
                    search_result = web_search.search(concept, max_results=3)
                    
                    # Сохраняем в историю обучения
                    results.append({
                        'concept': concept,
                        'search_results': search_result.get('results', []) if search_result else [],
                        'status': 'learned'
                    })
                    
                    logger.info(f"Самодиалог: изучено понятие '{concept}' через веб-поиск")
                
                # Также пробуем сохранить в граф знаний
                kg = getattr(self.brain, 'knowledge_graph', None)
                if kg and hasattr(kg, 'add_entity'):
                    try:
                        kg.add_entity(
                            name=concept,
                            entity_type='learned_concept',
                            properties={'source': 'self_dialog_learning', 'learned_from': 'web_search'}
                        )
                    except Exception as e:
                        logger.debug(f"Не удалось сохранить в граф: {e}")
                        
            except Exception as e:
                logger.error(f"Ошибка обучения понятия {concept}: {e}")
        
        return results
