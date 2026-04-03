# -*- coding: utf-8 -*-
"""
Remove fallback from SelfReasoningEngine
"""
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\reasoning\self_reasoning_engine.py'
with open(filepath, 'r', encoding='utf-8-sig') as f:
    content = f.read()

# Remove entire fallback section
old = '''        else:
            logger.warning("Two-Model Pipeline НЕДОСТУПЕН - используем fallback")
        
        # Fallback: используем llama_cpp_deployment если доступен
        logger.info("Checking llama_cpp_deployment fallback...")
        if hasattr(self.brain, 'llama_cpp_deployment') and self.brain.llama_cpp_deployment and getattr(self.brain, 'llama_cpp_ready', False):
            logger.info("Using GGUF fallback for generation...")
            try:
                gguf_response = self.brain.llama_cpp_deployment.generate(
                    prompt=enhanced_query,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9
                )
                if gguf_response:
                    logger.info(f"GGUF fallback generated: {gguf_response[:100]}...")
                    return {
                        "response": gguf_response,
                        "text": gguf_response,
                        "status": "ok",
                        "confidence": 0.8,
                        "reasoning": {"source": "gguf_fallback"},
                        "source": "self_reasoning_engine",
                        "processing_time": time.time() - start_time,
                        "conversation_history_used": len(conversation_history) > 0
                    }
            except Exception as e:
                logger.error(f"GGUF fallback failed: {e}")
        
        # Last resort: простой ответ
        logger.info("All generation methods failed, using simple response")
        simple_response = self._generate_simple_response(enhanced_query)
        
        return {
            "response": simple_response,
            "text": simple_response,
            "status": "ok",
            "confidence": 0.5,
            "reasoning": {"source": "simple_fallback"},
            "source": "self_reasoning_engine",
            "processing_time": time.time() - start_time,
            "conversation_history_used": len(conversation_history) > 0
        }'''

new = '''        else:
            logger.error("Two-Model Pipeline НЕДОСТУПЕН - нет резервного метода генерации")
            logger.error(f"self.two_model_pipeline: {self.two_model_pipeline is not None}")
            if self.brain:
                logger.error(f"brain.two_model_pipeline: {hasattr(self.brain, 'two_model_pipeline')}")
            return {
                "response": "Ошибка: Two-Model Pipeline недоступен.",
                "text": "Ошибка: Two-Model Pipeline недоступен.",
                "status": "error",
                "confidence": 0.0,
                "reasoning": {"source": "error", "message": "Two-Model Pipeline не инициализирован"},
                "source": "self_reasoning_engine",
                "processing_time": time.time() - start_time,
                "conversation_history_used": len(conversation_history) > 0
            }'''

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS: Fallback removed')
else:
    print('ERROR: pattern not found')
