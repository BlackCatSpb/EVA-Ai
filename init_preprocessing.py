# Read core_brain.py
with open('eva/core/core_brain.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the position after LlamaCpp initialization and add preprocessing_pipeline init
old_section = '''        except Exception as e:
            self.query_logger.debug(f"LlamaCpp не инициализирован: {e}")
            self.llama_cpp_deployment = None
        
        # Инициализация QwenModelManager'''

new_section = '''        except Exception as e:
            self.query_logger.debug(f"LlamaCpp не инициализирован: {e}")
            self.llama_cpp_deployment = None
        
        # Инициализация PreprocessingPipeline для извлечения сущностей
        self.preprocessing_pipeline = None
        try:
            from ..preprocess.preprocessing_pipeline import PreprocessingPipeline
            # Get llama instance for preprocessing
            llama_instance = None
            if self.llama_cpp_deployment and hasattr(self.llama_cpp_deployment, 'llama'):
                llama_instance = self.llama_cpp_deployment.llama
            
            self.preprocessing_pipeline = PreprocessingPipeline(
                llama_instance=llama_instance,
                hybrid_cache=self.hybrid_cache
            )
            self.query_logger.info("PreprocessingPipeline инициализирован")
        except ImportError as e:
            self.query_logger.debug(f"PreprocessingPipeline не найден: {e}")
        except Exception as e:
            self.query_logger.debug(f"Ошибка инициализации PreprocessingPipeline: {e}")
        
        # Инициализация QwenModelManager'''

content = content.replace(old_section, new_section)

# Write back
with open('eva/core/core_brain.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done - added preprocessing_pipeline initialization')
