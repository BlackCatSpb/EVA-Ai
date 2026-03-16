#!/usr/bin/env python3
"""
Диагностика и ремонт базы знаний CogniFlex
"""

import os
import sys
import sqlite3
import json
import shutil
from datetime import datetime

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_database_structure(db_path):
    """Проверяет структуру базы данных"""
    print(f"🔍 Проверка структуры БД: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Получаем схему таблицы nodes
        cursor.execute("PRAGMA table_info(nodes)")
        columns = cursor.fetchall()
        
        print("   📋 Колонки таблицы nodes:")
        for col in columns:
            print(f"      - {col[1]} ({col[2]})")
        
        # Получаем схему таблицы edges
        cursor.execute("PRAGMA table_info(edges)")
        edge_columns = cursor.fetchall()
        
        print("   📋 Колонки таблицы edges:")
        for col in edge_columns:
            print(f"      - {col[1]} ({col[2]})")
        
        # Проверяем наличие данных
        cursor.execute("SELECT COUNT(*) FROM nodes")
        nodes_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM edges")
        edges_count = cursor.fetchone()[0]
        
        print(f"   📊 Данные: {nodes_count} узлов, {edges_count} связей")
        
        conn.close()
        return True, columns, edge_columns
        
    except Exception as e:
        print(f"   ❌ Ошибка проверки БД: {e}")
        return False, [], []

def fix_database_schema(db_path):
    """Исправляет схему базы данных"""
    print(f"🔧 Ремонт схемы БД: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Проверяем наличие колонок spatial_info и temporal_info
        cursor.execute("PRAGMA table_info(nodes)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Добавляем недостающие колонки
        missing_columns = ['history', 'contradictions', 'keyword_index', 'concept_index']
        for col in missing_columns:
            if col not in columns:
                print(f"   ➕ Добавление колонки {col}")
                cursor.execute(f"ALTER TABLE nodes ADD COLUMN {col} TEXT")
        
        if 'spatial_info' not in columns:
            print("   ➕ Добавление колонки spatial_info")
            cursor.execute("ALTER TABLE nodes ADD COLUMN spatial_info TEXT")
        
        if 'temporal_info' not in columns:
            print("   ➕ Добавление колонки temporal_info")
            cursor.execute("ALTER TABLE nodes ADD COLUMN temporal_info TEXT")
        
        # Проверяем колонки для edges
        cursor.execute("PRAGMA table_info(edges)")
        edge_columns = [col[1] for col in cursor.fetchall()]
        
        # Добавляем недостающие колонки для edges
        missing_edge_columns = ['spatial_info', 'temporal_info', 'history', 'contradictions', 'last_updated']
        for col in missing_edge_columns:
            if col not in edge_columns:
                print(f"   ➕ Добавление колонки {col} в edges")
                cursor.execute(f"ALTER TABLE edges ADD COLUMN {col} TEXT")
        
        # Создаем индексы для оптимизации
        print("   🔍 Создание индексов...")
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_nodes_domain ON nodes(domain)",
            "CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)",
            "CREATE INDEX IF NOT EXISTS idx_nodes_strength ON nodes(strength)",
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)",
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)",
            "CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(relation_type)"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                print(f"   ✅ Индекс создан: {index_sql.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                print(f"   ⚠️ Индекс уже существует: {e}")
        
        conn.commit()
        conn.close()
        
        print("   ✅ Схема БД исправлена")
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка ремонта БД: {e}")
        return False

def backup_database(db_path):
    """Создает резервную копию БД"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{db_path}.backup_{timestamp}"
        shutil.copy2(db_path, backup_path)
        print(f"   💾 Резервная копия создана: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"   ❌ Ошибка создания резервной копии: {e}")
        return None

def test_database_operations(db_path):
    """Тестирует операции с БД"""
    print("🧪 Тестирование операций с БД...")
    
    try:
        from cogniflex.knowledge.knowledge_storage import KnowledgeStorage
        from cogniflex.knowledge.knowledge_nodes import KnowledgeNode, KnowledgeEdge, create_node_id, create_edge_id
        
        storage = KnowledgeStorage(db_path)
        
        # Тест добавления узла
        test_node = KnowledgeNode(
            id=create_node_id("test"),
            name="Тестовый узел",
            description="Узел для тестирования БД",
            node_type="test",
            domain="test",
            strength=0.8,
            spatial_info={"x": 0, "y": 0},
            temporal_info={"start": "2024-01-01"}
        )
        
        print("   📝 Тест сохранения узла...")
        storage.save_node(test_node)
        
        # Тест загрузки узла
        print("   📖 Тест загрузки узла...")
        loaded_nodes = storage.load_all_nodes()
        test_loaded = None
        for node in loaded_nodes.values():
            if node.name == "Тестовый узел":
                test_loaded = node
                break
        
        if test_loaded:
            print("   ✅ Узел успешно сохранен и загружен")
        else:
            print("   ❌ Ошибка загрузки узла")
            return False
        
        # Тест добавления связи
        test_edge = KnowledgeEdge(
            id=create_edge_id(test_node.id, test_node.id, "test"),
            source_id=test_node.id,
            target_id=test_node.id,
            relation_type="test",
            strength=0.9,
            spatial_info={"path": "test"},
            temporal_info={"duration": 10}
        )
        
        print("   📝 Тест сохранения связи...")
        storage.save_edge(test_edge)
        
        # Тест загрузки связей
        print("   📖 Тест загрузки связей...")
        loaded_edges = storage.load_all_edges()
        test_edge_loaded = None
        for edge in loaded_edges.values():
            if edge.relation_type == "test":
                test_edge_loaded = edge
                break
        
        if test_edge_loaded:
            print("   ✅ Связь успешно сохранена и загружена")
        else:
            print("   ❌ Ошибка загрузки связи")
            return False
        
        # Удаляем тестовые данные
        print("   🗑️ Очистка тестовых данных...")
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM nodes WHERE id = ?", (test_node.id,))
        cursor.execute("DELETE FROM edges WHERE id = ?", (test_edge.id,))
        conn.commit()
        conn.close()
        
        print("   ✅ Все операции с БД работают корректно")
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка тестирования БД: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основная функция"""
    print("🚀 Диагностика и ремонт базы знаний CogniFlex")
    print("=" * 60)
    
    # Пути к базам данных
    db_paths = [
        os.path.join(os.path.dirname(__file__), "cogniflex", "knowledge", "cogniflex_knowledge_cache", "knowledge_graph.db"),
        os.path.join(os.path.dirname(__file__), "cogniflex", "knowledge", "knowledge_cache", "knowledge_graph.db")
    ]
    
    results = []
    
    for db_path in db_paths:
        if os.path.exists(db_path):
            print(f"\n==================== {os.path.basename(db_path)} ====================")
            
            # Создаем резервную копию
            backup_path = backup_database(db_path)
            
            # Проверяем структуру
            success, columns, edge_columns = check_database_structure(db_path)
            
            if success:
                # Исправляем схему
                fix_success = fix_database_schema(db_path)
                
                if fix_success:
                    # Тестируем операции
                    test_success = test_database_operations(db_path)
                    results.append((os.path.basename(db_path), test_success))
                else:
                    results.append((os.path.basename(db_path), False))
            else:
                results.append((os.path.basename(db_path), False))
        else:
            print(f"\n==================== {os.path.basename(db_path)} ====================")
            print("   ⚠️ Файл БД не найден")
            results.append((os.path.basename(db_path), False))
    
    print("\n" + "=" * 60)
    print("📊 ИТОГИ РЕМОНТА:")
    
    for db_name, success in results:
        status = "✅ УСПЕХ" if success else "❌ НЕУДАЧА"
        print(f"   {db_name}: {status}")
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    print(f"\n🎯 Результат: {success_count}/{total_count} БД исправлено")
    
    if success_count == total_count:
        print("🎉 Все базы знаний успешно отремонтированы!")
        return True
    else:
        print("⚠️ Некоторые базы требуют внимания.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
