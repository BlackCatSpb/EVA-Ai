"""
Download and integrate knowledge datasets into EVA
"""
import os
import json
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eva_ai.knowledge.data_loader")

# Create knowledge_data directory
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), 'knowledge_data')
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

def download_file(url, dest_path, description="File"):
    """Download a file with progress indication"""
    if os.path.exists(dest_path):
        logger.info(f"{description} already exists at {dest_path}")
        return True
    
    logger.info(f"Downloading {description} from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"{description} downloaded successfully to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {description}: {e}")
        return False

def create_sample_wikidata():
    """Create comprehensive Russian knowledge dataset"""
    triplets = [
        # Geography - Russia
        {"subject": "Москва", "relation": "столица", "object": "Россия"},
        {"subject": "Москва", "relation": "находится", "object": "Европа"},
        {"subject": "Санкт-Петербург", "relation": "находится", "object": "Россия"},
        {"subject": "Волга", "relation": "длина", "object": "3530 км"},
        {"subject": "Урал", "relation": "горы", "object": " Россия"},
        {"subject": "Байкал", "relation": "глубина", "object": "1642 м"},
        {"subject": "Россия", "relation": "площадь", "object": "17.1 млн км2"},
        {"subject": "Россия", "relation": "население", "object": "146 млн"},
        
        # Science
        {"subject": "Исаак Ньютон", "relation": "родился", "object": "1643"},
        {"subject": "Исаак Ньютон", "relation": "открыл", "object": "законы движения"},
        {"subject": "Альберт Эйнштейн", "relation": "создал", "object": "теория относительности"},
        {"subject": "теория относительности", "relation": "описывает", "object": "гравитация"},
        {"subject": "Менделеев", "relation": "открыл", "object": "периодическая таблица"},
        {"subject": "Ломоносов", "relation": "родился", "object": "1711"},
        {"subject": "Попов", "relation": "изобрел", "object": "радио"},
        
        # Technology
        {"subject": "Python", "relation": "язык", "object": "программирования"},
        {"subject": "Python", "relation": "поддерживает", "object": "ООП"},
        {"subject": "нейронная сеть", "relation": "тип", "object": "машинное обучение"},
        {"subject": "машинное обучение", "relation": "раздел", "object": "искусственный интеллект"},
        {"subject": "OpenVINO", "relation": "инструмент", "object": "оптимизация нейросетей"},
        
        # General facts
        {"subject": "вода", "relation": "химическая_формула", "object": "H2O"},
        {"subject": "кислород", "relation": "элемент", "object": "номер 8"},
        {"subject": "солнце", "relation": "тип", "object": "звезда"},
        {"subject": "земля", "relation": "тип", "object": "планета"},
        {"subject": "кот", "relation": "животное", "object": "млекопитающее"},
        
        # Russian facts
        {"subject": "Толстой", "relation": "написал", "object": "Война и мир"},
        {"subject": "Пушкин", "relation": "написал", "object": "Евгений Онегин"},
        {"subject": "Чайковский", "relation": "композитор", "object": "Россия"},
        {"subject": "Репин", "relation": "художник", "object": "Россия"},
        
        # More technology
        {"subject": "EVA AI", "relation": "система", "object": "искусственный интеллект"},
        {"subject": "FractalGraph", "relation": "тип", "object": "граф знаний"},
        {"subject": "OpenVINO", "relation": "компания", "object": "Intel"},
    ]
    
    filepath = os.path.join(KNOWLEDGE_DIR, 'wikidata_russian.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({'triplets': triplets}, f, ensure_ascii=False, indent=2)
    logger.info(f"Created wikidata dataset with {len(triplets)} triplets")
    return filepath

def create_sample_nerel():
    """Create NEREL-style dataset with Russian entities"""
    documents = [
        {
            "id": "doc1",
            "text": "Владимир Путин родился в Ленинграде в 1952 году.",
            "entities": [
                {"id": 0, "type": "PER", "start": 0, "end": 16, "text": "Владимир Путин"},
                {"id": 1, "type": "LOC", "start": 25, "end": 34, "text": "Ленинград"},
                {"id": 2, "type": "DATE", "start": 40, "end": 44, "text": "1952"}
            ],
            "relations": [
                {"type": "PER_LOC", "head": 0, "tail": 1},
                {"type": "PER_DATE", "head": 0, "tail": 2}
            ]
        },
        {
            "id": "doc2",
            "text": "Москва - столица России и крупнейший город страны.",
            "entities": [
                {"id": 0, "type": "LOC", "start": 0, "end": 6, "text": "Москва"},
                {"id": 1, "type": "LOC", "start": 18, "end": 24, "text": "Россия"}
            ],
            "relations": [
                {"type": "LOC_LOC", "head": 0, "tail": 1}
            ]
        },
        {
            "id": "doc3",
            "text": "Компания Google была основана в СШАLarry Page и Sergey Brin.",
            "entities": [
                {"id": 0, "type": "ORG", "start": 0, "end": 6, "text": "Google"},
                {"id": 1, "type": "LOC", "start": 26, "end": 29, "text": "США"},
                {"id": 2, "type": "PER", "start": 30, "end": 40, "text": "Larry Page"},
                {"id": 3, "type": "PER", "start": 45, "end": 56, "text": "Sergey Brin"}
            ],
            "relations": [
                {"type": "ORG_LOC", "head": 0, "tail": 1},
                {"type": "ORG_PERS", "head": 0, "tail": 2},
                {"type": "ORG_PERS", "head": 0, "tail": 3}
            ]
        },
        {
            "id": "doc4",
            "text": "Альберт Эйнштейн создал теорию относительности в 1905 году.",
            "entities": [
                {"id": 0, "type": "PER", "start": 0, "end": 15, "text": "Альберт Эйнштейн"},
                {"id": 1, "type": "DATE", "start": 45, "end": 49, "text": "1905"}
            ],
            "relations": [
                {"type": "PER_DATE", "head": 0, "tail": 1}
            ]
        },
        {
            "id": "doc5",
            "text": "Исаак Ньютон открыл законы движения и гравитации.",
            "entities": [
                {"id": 0, "type": "PER", "start": 0, "end": 12, "text": "Исаак Ньютон"}
            ],
            "relations": []
        },
        {
            "id": "doc6",
            "text": "Python - высокоуровневый язык программирования.",
            "entities": [
                {"id": 0, "type": "ORG", "start": 0, "end": 6, "text": "Python"}
            ],
            "relations": []
        },
        {
            "id": "doc7",
            "text": "Ева AI использует FractalGraph для хранения знаний.",
            "entities": [
                {"id": 0, "type": "ORG", "start": 0, "end": 5, "text": "Ева AI"},
                {"id": 1, "type": "ORG", "start": 18, "end": 29, "text": "FractalGraph"}
            ],
            "relations": [
                {"type": "ORG_ORG", "head": 0, "tail": 1}
            ]
        },
        {
            "id": "doc8",
            "text": "Intel разработала OpenVINO для оптимизации нейросетей.",
            "entities": [
                {"id": 0, "type": "ORG", "start": 0, "end": 5, "text": "Intel"},
                {"id": 1, "type": "ORG", "start": 16, "end": 23, "text": "OpenVINO"}
            ],
            "relations": [
                {"type": "ORG_ORG", "head": 0, "tail": 1}
            ]
        },
        {
            "id": "doc9",
            "text": "Ломоносов родился в Архангельской губернии в 1711 году.",
            "entities": [
                {"id": 0, "type": "PER", "start": 0, "end": 9, "text": "Ломоносов"},
                {"id": 1, "type": "LOC", "start": 18, "end": 40, "text": "Архангельской губернии"},
                {"id": 2, "type": "DATE", "start": 46, "end": 50, "text": "1711"}
            ],
            "relations": [
                {"type": "PER_LOC", "head": 0, "tail": 1},
                {"type": "PER_DATE", "head": 0, "tail": 2}
            ]
        },
        {
            "id": "doc10",
            "text": "Пушкин написал Евгения Онегина и другие литературные произведения.",
            "entities": [
                {"id": 0, "type": "PER", "start": 0, "end": 7, "text": "Пушкин"},
                {"id": 1, "type": "ORG", "start": 17, "end": 30, "text": "Евгений Онегин"}
            ],
            "relations": [
                {"type": "PER_ORG", "head": 0, "tail": 1}
            ]
        }
    ]
    
    filepath = os.path.join(KNOWLEDGE_DIR, 'nerel_russian.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({'documents': documents}, f, ensure_ascii=False, indent=2)
    logger.info(f"Created NEREL dataset with {len(documents)} documents")
    return filepath

if __name__ == '__main__':
    print("=== EVA Knowledge Data Loader ===\n")
    
    # Create sample datasets
    print("Creating Russian knowledge datasets...")
    wikidata_path = create_sample_wikidata()
    nerel_path = create_sample_nerel()
    
    print(f"\nDatasets created:")
    print(f"  Wikidata: {wikidata_path}")
    print(f"  NEREL: {nerel_path}")
    
    print("\nNote: For full ConceptNet integration, run:")
    print("  from conceptnet_lite import download_db; download_db()")