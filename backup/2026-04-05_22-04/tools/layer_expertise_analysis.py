"""
Анализ экспертизы слоёв Qwen 2.5 3B
Запускается ОДНОКРАТНО. Результаты сохраняются в JSON для импорта в граф памяти.

Определяет, какие слои модели отвечают за:
- Синтаксис/грамматику
- Факты/знания
- Логику/рассуждения
- Код
- Креатив/стиль
"""
import os
import sys
import json
import logging
import numpy as np
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eva.layer_expertise")

# Тестовые тексты по категориям
TEST_TEXTS = {
    "syntax": [
        "Составь предложение с причастным оборотом.",
        "Исправь ошибки: Я пошёл в магазин и купил молоко хлеб и яйца.",
        "Как правильно: в течении дня или в течение дня?",
    ],
    "facts": [
        "Какая планета ближе всего к Солнцу?",
        "В каком году началась Вторая мировая война?",
        "Столица Австралии — это Сидней или Канберра?",
    ],
    "logic": [
        "Если все кошки млекопитающие, а Барсик — кошка, то кто Барсик?",
        "У меня 3 яблока, я съел одно и купил ещё два. Сколько у меня яблок?",
        "Докажи, что сумма углов треугольника равна 180 градусам.",
    ],
    "code": [
        "Напиши функцию на Python для сортировки списка пузырьком.",
        "Как сделать HTTP запрос в JavaScript?",
        "Напиши SQL запрос для выборки всех пользователей старше 18 лет.",
    ],
    "creative": [
        "Напиши короткое стихотворение про осень.",
        "Придумай название для кафе в стиле ретро.",
        "Опиши закат над морем тремя предложениями.",
    ]
}


def analyze_layer_expertise(model_path: str, output_path: str = None):
    """
    Анализирует экспертизу каждого слоя модели.
    
    Args:
        model_path: Путь к модели (transformers format)
        output_path: Куда сохранить результаты
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   'memory', 'layer_expertise.json')
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        logger.error("Установите transformers: pip install transformers")
        return
    
    logger.info(f"Загрузка модели: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True
    )
    model.eval()
    
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    logger.info(f"Модель: {num_layers} слоёв, hidden_dim={hidden_dim}")
    
    # Собираем активации слоёв для каждого текста
    layer_activations: Dict[str, List[np.ndarray]] = {cat: [] for cat in TEST_TEXTS}
    
    with torch.no_grad():
        for category, texts in TEST_TEXTS.items():
            logger.info(f"Анализ категории: {category} ({len(texts)} текстов)")
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_tokens=512)
                if hasattr(inputs, 'input_ids'):
                    input_ids = inputs.input_ids
                    if hasattr(model, 'device'):
                        input_ids = input_ids.to(model.device)
                    
                    outputs = model(input_ids)
                    hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)
                    
                    # Mean pooling по последовательности для каждого слоя
                    for layer_idx in range(num_layers + 1):  # +1 для embedding layer
                        layer_out = hidden_states[layer_idx].cpu().numpy()
                        pooled = layer_out.mean(axis=1)  # [1, hidden_dim]
                        layer_activations[category].append(pooled.flatten())
    
    # Вычисляем профиль каждого слоя: насколько он активен для каждой категории
    logger.info("Вычисление профилей слоёв...")
    layer_profiles = []
    
    for layer_idx in range(num_layers + 1):
        profile = {}
        for category in TEST_TEXTS:
            activations = layer_activations[category]
            if activations:
                # Среднее значение активации для категории
                mean_activation = np.mean([a[layer_idx % len(a)] for a in activations])
                profile[category] = float(mean_activation)
            else:
                profile[category] = 0.0
        
        # Определяем доминирующую категорию
        dominant = max(profile, key=profile.get)
        profile['dominant'] = dominant
        profile['layer'] = layer_idx
        
        layer_profiles.append(profile)
    
    # Кластеризация слоёв по схожести профилей
    logger.info("Кластеризация слоёв...")
    feature_matrix = np.array([[p[cat] for cat in TEST_TEXTS] for p in layer_profiles])
    
    # Нормализация
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # KMeans с 5 кластерами (по числу категорий + embedding)
    n_clusters = min(5, num_layers)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(feature_matrix_scaled)
    
    # Назначаем метки кластерам по доминирующей категории
    cluster_labels = {}
    for c in range(n_clusters):
        mask = clusters == c
        if mask.any():
            dominant_cats = [layer_profiles[i]['dominant'] for i in range(len(layer_profiles)) if clusters[i] == c]
            cluster_labels[c] = max(set(dominant_cats), key=dominant_cats.count)
    
    # Формируем результат
    result = {
        'model_path': model_path,
        'num_layers': num_layers,
        'hidden_dim': hidden_dim,
        'layers': [],
        'cluster_summary': {},
    }
    
    for i, profile in enumerate(layer_profiles):
        cluster_id = int(clusters[i])
        layer_info = {
            'layer': i,
            'type': 'embedding' if i == 0 else 'hidden',
            'dominant_category': profile['dominant'],
            'cluster': cluster_id,
            'cluster_label': cluster_labels.get(cluster_id, 'unknown'),
            'activation_profile': {k: round(v, 4) for k, v in profile.items() if k != 'dominant'},
        }
        result['layers'].append(layer_info)
    
    # Сводка по кластерам
    for c in range(n_clusters):
        layers_in_cluster = [i for i in range(len(layer_profiles)) if clusters[i] == c]
        result['cluster_summary'][f'cluster_{c}'] = {
            'label': cluster_labels.get(c, 'unknown'),
            'layers': layers_in_cluster,
            'layer_range': f"{min(layers_in_cluster)}-{max(layers_in_cluster)}" if layers_in_cluster else '',
        }
    
    # Сохраняем
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Результаты сохранены: {output_path}")
    logger.info(f"Найдено {n_clusters} кластеров слоёв:")
    for c, info in result['cluster_summary'].items():
        logger.info(f"  {c}: {info['label']} (слои {info['layer_range']})")
    
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Анализ экспертизы слоёв модели")
    parser.add_argument("--model-path", required=True, help="Путь к модели transformers")
    parser.add_argument("--output", default=None, help="Путь для сохранения результатов")
    args = parser.parse_args()
    
    analyze_layer_expertise(args.model_path, args.output)
