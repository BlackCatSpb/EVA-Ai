import re

# Читаем файл
with open('cogniflex/mlearning/fractal_model_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Находим второй блок (после logger.info с параметрами модели)
pattern = r'(logger\.info\(f"Определены параметры модели: vocab_size=\{vocab_size\}, n_embd=\{n_embd\}, n_layer=\{n_layer\}, n_head=\{n_head\}, has_bias=\{has_bias\}"\).*?# Загружаем веса с strict=False для пропуска отсутствующих bias\s+missing_keys, unexpected_keys = self\.model\.load_state_dict\(self\.state_dict, strict=False\))'

# Заменяем на исправленный вариант
replacement = '''logger.info(f"Определены параметры модели: vocab_size={vocab_size}, n_embd={n_embd}, n_layer={n_layer}, n_head={n_head}, has_bias={has_bias}")

                    # Создаем конфигурацию GPT2 с правильными параметрами
                    self.config = GPT2Config(
                        vocab_size=vocab_size,
                        n_embd=n_embd,
                        n_layer=n_layer,
                        n_head=n_head,
                        n_positions=n_positions,
                        n_ctx=n_positions,
                        # Устанавливаем use_bias в зависимости от наличия bias в весах
                        resid_pdrop=0.1,
                        embd_pdrop=0.1,
                        attn_pdrop=0.1,
                        # Важно: устанавливаем правильные параметры для bias
                        use_cache=True,
                    )
                    
                    # Создаем модель
                    self.model = GPT2LMHeadModel(self.config)
                    
                    # Специальная обработка для lm_head.weight
                    temp_missing_keys, temp_unexpected_keys = self.model.load_state_dict(self.state_dict, strict=False)
                    if 'lm_head.weight' in temp_missing_keys and 'transformer.wte.weight' in self.state_dict:
                        logger.info("lm_head.weight отсутствует, но есть transformer.wte.weight - используем общие веса")
                        self.model.lm_head.weight = self.model.transformer.wte.weight
                        temp_missing_keys.remove('lm_head.weight')
                    
                    # Загружаем веса с strict=False для пропуска отсутствующих bias
                    missing_keys, unexpected_keys = self.model.load_state_dict(self.state_dict, strict=False)'''

# Выполняем замену только для второго вхождения
parts = content.split('logger.info(f"Определены параметры модели: vocab_size=')
if len(parts) > 2:
    # Находим второе вхождение и заменяем его
    new_content = parts[0] + 'logger.info(f"Определены параметры модели: vocab_size=' + parts[1]
    
    # Находим конец блока для замены
    search_start = 'logger.info(f"Определены параметры модели: vocab_size=' + parts[1]
    idx = new_content.find('# Загружаем веса с strict=False для пропуска отсутствующих bias')
    if idx != -1:
        # Находим конец строки с missing_keys
        end_idx = new_content.find('\n\n', idx)
        if end_idx != -1:
            # Заменяем блок
            before = new_content[:idx]
            after = new_content[end_idx:]
            new_content = before + replacement + after
    
    # Записываем обратно
    with open('cogniflex/mlearning/fractal_model_manager.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Исправление применено успешно")
else:
    print("Не найдено второе вхождение логгера")
