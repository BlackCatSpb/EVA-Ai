def generate_with_fcp_api(self, prompt: str, max_new_tokens: int = 1024,
                               enable_thinking: bool = True, return_metadata: bool = False) -> str:
        """Генерация через FCP Inference API - используем внутренний LLMPipeline"""
        logger.info("[FCP API] Starting generation")
        
        # Используем self.pipeline (regular LLMPipeline) - он работает правильно
        if self.pipeline:
            try:
                formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                gen_cfg = self.get_generation_config(max_new_tokens)
                result = self.pipeline.generate(formatted_prompt, generation_config=gen_cfg)
                logger.info(f"[FCP API] Done via pipeline")
                return str(result)
            except Exception as e:
                logger.error(f"[FCP API] Pipeline generation failed: {e}")
        
        # Fallback
        return self._generate(prompt, max_new_tokens, **{})


def create_fcp_pipeline(model_path: str, graph_path: str = None, **kwargs):
    """Factory function"""
    return FCPPipelineV15(model_path, graph_path, **kwargs)