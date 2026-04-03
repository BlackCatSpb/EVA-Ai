filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\core\recursive_model_pipeline.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the final section - add experience saving
old_final = '''        results['final_quality'] = model_b_result['quality']
        
        # Сохранение в фрактальный граф отключено - создаёт петлю загрязнения
        
        logger.info("Three-GGUF пайплайн завершён")
        
        return results'''

new_final = '''        results['final_quality'] = model_b_result['quality']
        
        # Сохраняем опыт в граф обучения (цикл обучения через граф)
        if self.fractal_memory and hasattr(self.fractal_memory, 'save_experience'):
            # Сохраняем оба ответа как опыт
            self.fractal_memory.save_experience(
                query=query,
                response=model_a_result['natural_response'],
                model_used='model_a',
                quality_score=model_a_result['quality'].get('score', 0.5)
            )
            self.fractal_memory.save_experience(
                query=query,
                response=model_b_result['natural_response'],
                model_used='model_b',
                quality_score=model_b_result['quality'].get('score', 0.5)
            )
        
        logger.info("Three-GGUF пайплайн завершён")
        
        return results'''

if old_final in content:
    content = content.replace(old_final, new_final)
    print('Fixed final section')
else:
    print('ERROR: final section not found')

# Add context enrichment from graph learning (clean, not contaminated)
old_enrich = '''        # Fractal memory enrichment отключена - подтягивает загрязнённый контекст из предыдущих ответов
        enriched_query = query'''

new_enrich = '''        # Контекст из графа обучения (чистый — только концепты и качественные опыты)
        enriched_query = query
        if self.fractal_memory and hasattr(self.fractal_memory, 'get_context_for_query'):
            graph_context = self.fractal_memory.get_context_for_query(query)
            if graph_context:
                enriched_query = f"{query}\\n\\nКонтекст из опыта:\\n{graph_context}"
                results['fractal_context'] = graph_context
                logger.info(f"Контекст из графа обучения: {len(graph_context)} символов")'''

if old_enrich in content:
    content = content.replace(old_enrich, new_enrich)
    print('Fixed enrichment')
else:
    print('ERROR: enrichment not found')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')