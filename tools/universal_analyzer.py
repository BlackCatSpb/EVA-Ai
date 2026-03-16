"""
Universal Python Project Analyzer
Универсальный анализатор Python-проектов для статического анализа кода
"""

import os
import ast
import json
import time
import datetime
import argparse
import configparser
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

class ErrorSeverity(Enum):
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

class ErrorType(Enum):
    SYNTAX_ERROR = "syntax_error"
    UNDEFINED_VARIABLE = "undefined_variable"
    UNDEFINED_FUNCTION = "undefined_function"
    UNDEFINED_CLASS = "undefined_class"
    UNDEFINED_METHOD = "undefined_method"
    UNDEFINED_ATTRIBUTE = "undefined_attribute"
    UNUSED_IMPORT = "unused_import"
    CIRCULAR_IMPORT = "circular_import"
    ARCHITECTURE_VIOLATION = "architecture_violation"
    CODE_SMELL = "code_smell"

@dataclass
class AnalysisError:
    id: int
    file_path: str
    line: int
    column: int
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    context: str
    suggestion: str
    rule_id: str

@dataclass
class AnalysisConfig:
    project_root: str
    exclude_dirs: List[str]
    include_extensions: List[str]
    check_undefined_variables: bool = True
    check_unused_imports: bool = True
    check_circular_imports: bool = True
    check_code_smells: bool = True
    check_architecture: bool = False
    core_files: List[str] = None
    forbidden_imports: Dict[str, List[str]] = None
    output_format: str = "console"
    save_to_file: bool = True
    log_directory: str = "analysis_logs"
    max_file_size_mb: int = 10
    parallel_analysis: bool = False

class UniversalPythonAnalyzer:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.project_root = Path(config.project_root).resolve()
        self.errors: List[AnalysisError] = []
        self.error_counter = 1
        self.module_definitions: Dict[str, Dict] = {}
        self.import_graph: Dict[str, Set[str]] = {}
        self.class_hierarchy: Dict[str, List[str]] = {}
        self.function_calls: Dict[str, List[str]] = {}
        self.current_file: Optional[str] = None
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None
        self.log_dir = self.project_root / config.log_directory
        self.log_dir.mkdir(exist_ok=True)
        self.session_start = datetime.datetime.now()
        self.standard_excludes = {
            'venv', '.venv', 'env', '.env', '.git', '__pycache__', 
            'node_modules', 'dist', 'build', '.pytest_cache', 
            '.mypy_cache', '.tox', 'htmlcov', 'site-packages',
            'lib', 'scripts', 'include', 'nest-simulator'
        }
        self.builtin_names = set(dir(__builtins__)) | {
            'ABC', 'Enum', 'object', 'type', 'list', 'dict', 'str', 
            'int', 'float', 'bool', 'Exception', 'BaseException',
            'FileSystemEventHandler', 'Directive', 'SphinxDirective'
        }
    
    def load_config_from_file(self, config_path: str) -> AnalysisConfig:
        config = configparser.ConfigParser()
        config.read(config_path)
        return AnalysisConfig(
            project_root=config.get('main', 'project_root', fallback='.'),
            exclude_dirs=config.get('main', 'exclude_dirs', fallback='').split(','),
            include_extensions=config.get('main', 'include_extensions', fallback='.py').split(','),
            check_undefined_variables=config.getboolean('rules', 'check_undefined_variables', fallback=True),
            check_unused_imports=config.getboolean('rules', 'check_unused_imports', fallback=True),
            output_format=config.get('output', 'format', fallback='console')
        )
    
    def should_analyze_file(self, file_path: Path) -> bool:
        if file_path.suffix not in self.config.include_extensions:
            return False
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > self.config.max_file_size_mb:
                return False
        except OSError:
            return False
        exclude_dirs = set(self.config.exclude_dirs) | self.standard_excludes
        file_str = str(file_path).lower()
        for exclude in exclude_dirs:
            if exclude.lower() in file_str:
                return False
        if ('site-packages' in file_str or 
            'lib\\python' in file_str or
            'appdata\\local\\programs\\python' in file_str or
            'nest-simulator' in file_str):
            return False
        return True
    
    def collect_project_files(self) -> List[Path]:
        files = []
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file() and self.should_analyze_file(file_path):
                files.append(file_path)
        return files
    
    def parse_file(self, file_path: Path) -> Optional[ast.AST]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            self.add_error(
                file_path=str(file_path),
                line=e.lineno or 0,
                column=e.offset or 0,
                error_type=ErrorType.SYNTAX_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"Синтаксическая ошибка: {e.msg}",
                context="Парсинг файла",
                suggestion="Исправьте синтаксическую ошибку",
                rule_id="SYNTAX_001"
            )
            return None
        except UnicodeDecodeError:
            self.add_error(
                file_path=str(file_path),
                line=0,
                column=0,
                error_type=ErrorType.SYNTAX_ERROR,
                severity=ErrorSeverity.ERROR,
                message="Ошибка кодировки файла",
                context="Чтение файла",
                suggestion="Проверьте кодировку файла (должна быть UTF-8)",
                rule_id="ENCODING_001"
            )
            return None
        except Exception as e:
            self.add_error(
                file_path=str(file_path),
                line=0,
                column=0,
                error_type=ErrorType.SYNTAX_ERROR,
                severity=ErrorSeverity.WARNING,
                message=f"Ошибка при чтении файла: {e}",
                context="Чтение файла",
                suggestion="Проверьте доступность и целостность файла",
                rule_id="FILE_001"
            )
            return None
    
    def collect_definitions(self, file_path: Path, tree: ast.AST):
        file_str = str(file_path)
        definitions = {
            "classes": {},
            "functions": {},
            "variables": set(),
            "imports": set(),
            "from_imports": {},
            "constants": set()
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    definitions["imports"].add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module not in definitions["from_imports"]:
                    definitions["from_imports"][node.module] = set()
                for alias in node.names:
                    definitions["from_imports"][node.module].add(alias.name)
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                definitions["classes"][node.name] = {
                    "line": node.lineno,
                    "methods": {},
                    "bases": [base.id for base in node.bases if isinstance(base, ast.Name)]
                }
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        definitions["classes"][node.name]["methods"][item.name] = {
                            "line": item.lineno,
                            "args": [arg.arg for arg in item.args.args]
                        }
            elif isinstance(node, ast.FunctionDef):
                definitions["functions"][node.name] = {
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args]
                }
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        definitions["variables"].add(target.id)
                        if target.id.isupper():
                            definitions["constants"].add(target.id)
        
        self.module_definitions[file_str] = definitions
    
    def add_error(self, file_path: str, line: int, column: int, error_type: ErrorType, 
                  severity: ErrorSeverity, message: str, context: str, 
                  suggestion: str, rule_id: str):
        error = AnalysisError(
            id=self.error_counter,
            file_path=file_path,
            line=line,
            column=column,
            error_type=error_type,
            severity=severity,
            message=message,
            context=context,
            suggestion=suggestion,
            rule_id=rule_id
        )
        self.errors.append(error)
        self.error_counter += 1
    
    def analyze_undefined_names(self, file_path: Path, tree: ast.AST):
        if not self.config.check_undefined_variables:
            return
        
        file_str = str(file_path)
        definitions = self.module_definitions.get(file_str, {})
        
        available_names = set()
        available_names.update(definitions.get("variables", set()))
        available_names.update(definitions.get("functions", {}).keys())
        available_names.update(definitions.get("classes", {}).keys())
        available_names.update(definitions.get("imports", set()))
        available_names.update(self.builtin_names)
        
        for module, names in definitions.get("from_imports", {}).items():
            available_names.update(names)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in available_names:
                    self.add_error(
                        file_path=file_str,
                        line=node.lineno,
                        column=node.col_offset,
                        error_type=ErrorType.UNDEFINED_VARIABLE,
                        severity=ErrorSeverity.ERROR,
                        message=f"Неопределенная переменная: '{node.id}'",
                        context=f"Использование переменной '{node.id}'",
                        suggestion=f"Определите переменную '{node.id}' или проверьте импорты",
                        rule_id="UNDEF_VAR_001"
                    )
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                if isinstance(node.value, ast.Name):
                    obj_name = node.value.id
                    if obj_name not in available_names:
                        self.add_error(
                            file_path=file_str,
                            line=node.lineno,
                            column=node.col_offset,
                            error_type=ErrorType.UNDEFINED_ATTRIBUTE,
                            severity=ErrorSeverity.WARNING,
                            message=f"Неопределенный объект для атрибута: '{obj_name}.{node.attr}'",
                            context=f"Доступ к атрибуту '{node.attr}' объекта '{obj_name}'",
                            suggestion=f"Убедитесь, что объект '{obj_name}' определен",
                            rule_id="UNDEF_ATTR_001"
                        )
    
    def analyze_unused_imports(self, file_path: Path, tree: ast.AST):
        if not self.config.check_unused_imports:
            return
        
        file_str = str(file_path)
        definitions = self.module_definitions.get(file_str, {})
        
        imports = set(definitions.get("imports", set()))
        from_imports = set()
        for names in definitions.get("from_imports", {}).values():
            from_imports.update(names)
        
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                used_names.add(node.value.id)
        
        for imp in imports:
            if imp not in used_names:
                self.add_error(
                    file_path=file_str,
                    line=1,
                    column=0,
                    error_type=ErrorType.UNUSED_IMPORT,
                    severity=ErrorSeverity.INFO,
                    message=f"Неиспользуемый импорт: '{imp}'",
                    context=f"Импорт '{imp}' не используется в коде",
                    suggestion=f"Удалите неиспользуемый импорт '{imp}'",
                    rule_id="UNUSED_IMP_001"
                )
        
        for imp in from_imports:
            if imp not in used_names:
                self.add_error(
                    file_path=file_str,
                    line=1,
                    column=0,
                    error_type=ErrorType.UNUSED_IMPORT,
                    severity=ErrorSeverity.INFO,
                    message=f"Неиспользуемый импорт: '{imp}'",
                    context=f"Импорт '{imp}' не используется в коде",
                    suggestion=f"Удалите неиспользуемый импорт '{imp}'",
                    rule_id="UNUSED_IMP_002"
                )
    
    def analyze_code_smells(self, file_path: Path, tree: ast.AST):
        if not self.config.check_code_smells:
            return
        
        file_str = str(file_path)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if func_lines > 50:
                    self.add_error(
                        file_path=file_str,
                        line=node.lineno,
                        column=node.col_offset,
                        error_type=ErrorType.CODE_SMELL,
                        severity=ErrorSeverity.WARNING,
                        message=f"Слишком длинная функция: '{node.name}' ({func_lines} строк)",
                        context=f"Функция '{node.name}' содержит {func_lines} строк",
                        suggestion="Разбейте функцию на более мелкие части",
                        rule_id="SMELL_LONG_FUNC_001"
                    )
                
                if len(node.args.args) > 7:
                    self.add_error(
                        file_path=file_str,
                        line=node.lineno,
                        column=node.col_offset,
                        error_type=ErrorType.CODE_SMELL,
                        severity=ErrorSeverity.WARNING,
                        message=f"Слишком много параметров в функции: '{node.name}' ({len(node.args.args)})",
                        context=f"Функция '{node.name}' имеет {len(node.args.args)} параметров",
                        suggestion="Используйте объекты или словари для группировки параметров",
                        rule_id="SMELL_MANY_PARAMS_001"
                    )
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
                depth = self._calculate_nesting_depth(node)
                if depth > 4:
                    self.add_error(
                        file_path=file_str,
                        line=node.lineno,
                        column=node.col_offset,
                        error_type=ErrorType.CODE_SMELL,
                        severity=ErrorSeverity.WARNING,
                        message=f"Слишком глубокая вложенность: {depth} уровней",
                        context=f"Блок кода имеет вложенность {depth} уровней",
                        suggestion="Вынесите логику в отдельные функции",
                        rule_id="SMELL_DEEP_NEST_001"
                    )
    
    def _calculate_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        max_depth = current_depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                child_depth = self._calculate_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
        return max_depth
    
    def analyze_architecture(self, file_path: Path, tree: ast.AST):
        if not self.config.check_architecture:
            return
        
        file_str = str(file_path)
        if "core_brain" in file_str.lower():
            return
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if (isinstance(target.value, ast.Name) and 
                            target.value.id.lower() in ['core', 'core_brain']):
                            self.add_error(
                                file_path=file_str,
                                line=node.lineno,
                                column=node.col_offset,
                                error_type=ErrorType.ARCHITECTURE_VIOLATION,
                                severity=ErrorSeverity.ERROR,
                                message="Прямое изменение состояния ядра системы",
                                context=f"Прямое присваивание в {target.value.id}.{target.attr}",
                                suggestion="Используйте методы интерфейса для изменения состояния",
                                rule_id="ARCH_CORE_MODIFY_001"
                            )
    
    def analyze_file(self, file_path: Path) -> bool:
        self.current_file = str(file_path)
        tree = self.parse_file(file_path)
        if not tree:
            return False
        
        self.collect_definitions(file_path, tree)
        self.analyze_undefined_names(file_path, tree)
        self.analyze_unused_imports(file_path, tree)
        self.analyze_code_smells(file_path, tree)
        self.analyze_architecture(file_path, tree)
        return True
    
    def analyze_project(self) -> Dict[str, Any]:
        print(f"Начинаем анализ проекта: {self.project_root}")
        files = self.collect_project_files()
        print(f"Найдено файлов для анализа: {len(files)}")
        
        if not files:
            return {
                "success": False,
                "error": "Не найдено файлов для анализа",
                "files_analyzed": 0,
                "errors_found": 0
            }
        
        analyzed_files = 0
        for file_path in files:
            try:
                if self.analyze_file(file_path):
                    analyzed_files += 1
                    if analyzed_files % 10 == 0 or analyzed_files <= 5:
                        print(f"Проанализирован: {file_path.name}")
            except Exception as e:
                print(f"Ошибка анализа {file_path}: {e}")
        
        result = {
            "success": True,
            "project_root": str(self.project_root),
            "files_total": len(files),
            "files_analyzed": analyzed_files,
            "errors_found": len(self.errors),
            "errors_by_type": self._group_errors_by_type(),
            "errors_by_severity": self._group_errors_by_severity(),
            "timestamp": self.session_start.isoformat()
        }
        
        if self.config.save_to_file:
            self._save_results(result)
        
        return result
    
    def _group_errors_by_type(self) -> Dict[str, int]:
        groups = {}
        for error in self.errors:
            error_type = error.error_type.value
            groups[error_type] = groups.get(error_type, 0) + 1
        return groups
    
    def _group_errors_by_severity(self) -> Dict[str, int]:
        groups = {}
        for error in self.errors:
            severity = error.severity.value
            groups[severity] = groups.get(severity, 0) + 1
        return groups
    
    def _save_results(self, result: Dict[str, Any]):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_file = self.log_dir / f"analysis_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "analysis_result": result,
                "errors": [asdict(error) for error in self.errors]
            }, f, ensure_ascii=False, indent=2)
        
        txt_file = self.log_dir / f"analysis_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Отчет анализа проекта\n")
            f.write(f"Время: {result['timestamp']}\n")
            f.write(f"Проект: {result['project_root']}\n")
            f.write(f"Файлов проанализировано: {result['files_analyzed']}\n")
            f.write(f"Ошибок найдено: {result['errors_found']}\n\n")
            
            if self.errors:
                f.write("НАЙДЕННЫЕ ОШИБКИ:\n")
                f.write("=" * 50 + "\n")
                
                for error in self.errors:
                    f.write(f"\n[{error.severity.value.upper()}] {error.error_type.value}\n")
                    f.write(f"Файл: {error.file_path}\n")
                    f.write(f"Строка: {error.line}, Колонка: {error.column}\n")
                    f.write(f"Сообщение: {error.message}\n")
                    f.write(f"Контекст: {error.context}\n")
                    f.write(f"Предложение: {error.suggestion}\n")
                    f.write(f"Правило: {error.rule_id}\n")
                    f.write("-" * 30 + "\n")
    
    def print_report(self):
        if not self.errors:
            print("\nОшибок не найдено!")
            return
        
        print(f"\nНайдено ошибок: {len(self.errors)}")
        
        by_severity = self._group_errors_by_severity()
        for severity, count in by_severity.items():
            print(f"  {severity}: {count}")
        
        print("\n" + "=" * 60)
        print("ДЕТАЛЬНЫЙ ОТЧЕТ")
        print("=" * 60)
        
        for error in self.errors:
            severity_prefix = {
                "critical": "[CRIT]",
                "error": "[ERR]",
                "warning": "[WARN]",
                "info": "[INFO]"
            }.get(error.severity.value, "[?]")
            
            print(f"\n{severity_prefix} {error.message}")
            print(f"   Файл: {error.file_path}:{error.line}:{error.column}")
            print(f"   Решение: {error.suggestion}")

def create_default_config(project_root: str) -> AnalysisConfig:
    return AnalysisConfig(
        project_root=project_root,
        exclude_dirs=[
            'venv', '.venv', 'env', '.env', '.git', '__pycache__',
            'node_modules', 'dist', 'build', '.pytest_cache',
            'memory_storage', 'models', 'project_analyzer', 'analysis_logs'
        ],
        include_extensions=['.py'],
        check_undefined_variables=True,
        check_unused_imports=True,
        check_circular_imports=False,
        check_code_smells=True,
        check_architecture=True,
        output_format="console",
        save_to_file=True,
        log_directory="analysis_logs"
    )

def main():
    parser = argparse.ArgumentParser(description="Универсальный анализатор Python-проектов")
    parser.add_argument("project_path", nargs="?", default=".", help="Путь к проекту")
    parser.add_argument("--config", help="Путь к файлу конфигурации")
    parser.add_argument("--format", choices=["console", "json", "html"], default="console", help="Формат вывода")
    parser.add_argument("--no-save", action="store_true", help="Не сохранять результаты в файл")
    
    args = parser.parse_args()
    
    if args.config and os.path.exists(args.config):
        analyzer = UniversalPythonAnalyzer(AnalysisConfig(project_root="."))
        config = analyzer.load_config_from_file(args.config)
    else:
        config = create_default_config(args.project_path)
    
    config.output_format = args.format
    config.save_to_file = not args.no_save
    
    analyzer = UniversalPythonAnalyzer(config)
    
    print("Запуск универсального анализатора Python-проектов")
    print(f"Проект: {analyzer.project_root}")
    
    result = analyzer.analyze_project()
    
    if result["success"]:
        print(f"\nАнализ завершен успешно!")
        print(f"Проанализировано файлов: {result['files_analyzed']}")
        print(f"Найдено проблем: {result['errors_found']}")
        
        if config.output_format == "console":
            analyzer.print_report()
        elif config.output_format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Ошибка анализа: {result.get('error', 'Неизвестная ошибка')}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())