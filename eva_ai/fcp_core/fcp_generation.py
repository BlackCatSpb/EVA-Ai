"""
FCP Generation - Методы генерации текста
Часть FCPipeline - вынесена для модульности
"""
import logging
import time
import numpy as np
from typing import Optional, Dict, Any, Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eva_ai.core.fcp_pipeline import FCPipeline

logger = logging.getLogger("eva_ai.fcp_generation")


def _convert_tokenized_to_numpy(fcp: 'FCPipeline', tokenized) -> Optional[np.ndarray]:
    """Конвертация результата токенизации в numpy массив"""
    try:
        if hasattr(tokenized, 'input_ids'):
            ids = tokenized.input_ids
            
            if hasattr(ids, 'data'):
                try:
                    size = ids.get_size()
                    shape = ids.get_shape()
                    data = np.frombuffer(ids.data, dtype=np.int64, count=size)
                    if shape:
                        data = data.reshape(shape)
                    return data
                except Exception as e:
                    logger.debug(f"[FCP] get_size/shape failed: {e}")
                    if hasattr(ids, 'shape'):
                        shape = ids.shape
                        data = np.frombuffer(ids.data, dtype=np.int64)
                        data = data.reshape(shape)
                        return data
        
        return np.asarray(tokenized, dtype=np.int64)
    except Exception as e:
        logger.error(f"[FCP] Tokenized conversion failed: {e}")
        return None


def _build_prompt(fcp: 'FCPipeline', prompt: str, enable_thinking: bool) -> str:
    """Построить промпт с учётом шаблона"""
    try:
        if hasattr(fcp, 'tokenizer') and fcp.tokenizer:
            chat_template = getattr(fcp.tokenizer, 'chat_template', None)
            if chat_template:
                messages = [{"role": "user", "content": prompt}]
                return fcp.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        logger.debug(f"[FCP] Chat template failed: {e}")
    
    return prompt


def _generate(fcp: 'FCPipeline', prompt: str, max_new_tokens: int = 1024, **kwargs) -> str:
    """Основная генерация через OpenVINO pipeline"""
    if not fcp.pipeline:
        return "Pipeline not initialized"
    
    built_prompt = _build_prompt(fcp, prompt, kwargs.get('enable_thinking', True))
    
    try:
        streamer = None
        if kwargs.get('stream_callback'):
            from eva_ai.core.fcp_pipeline import SimpleStreamer
            streamer = SimpleStreamer(callback=kwargs['stream_callback'])
        
        result = fcp.pipeline.generate(
            built_prompt,
            max_new_tokens=max_new_tokens,
            streamer=streamer
        )
        
        if streamer:
            return streamer.generated_text
        return str(result)
        
    except Exception as e:
        logger.error(f"[FCP] _generate failed: {e}")
        return f"Generation error: {e}"


def generate_with_fcp_api(fcp: 'FCPipeline', prompt: str, max_new_tokens: int = 1024,
                          enable_thinking: bool = True, return_metadata: bool = False) -> str:
    """Генерация через низкоуровневый FCP Inference API"""
    logger.info(f"[FCP API GEN] ===== START =====")
    
    if not fcp.fcp_api or not fcp.fcp_api.is_initialized():
        logger.warning("[FCP API GEN] fcp_api not initialized, using fallback")
        return _generate(fcp, prompt, max_new_tokens)
    
    if not hasattr(fcp, 'tokenizer') or fcp.tokenizer is None:
        logger.warning("[FCP API GEN] No tokenizer, using fallback")
        return _generate(fcp, prompt, max_new_tokens)
    
    try:
        tokenized = fcp.tokenizer.encode(prompt)
        input_ids = _convert_tokenized_to_numpy(fcp, tokenized)
        
        if input_ids is None or input_ids.size == 0:
            raise RuntimeError("[FCP API GEN] Tokenization failed")
        
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        
        logger.info(f"[FCP API GEN] input_ids shape: {input_ids.shape}")
        
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        
        logger.info("[FCP API GEN] Running prefill...")
        prefill_outputs = fcp.fcp_api.infer(input_ids, attention_mask)
        logger.info(f"[FCP API GEN] Prefill done, keys: {list(prefill_outputs.keys())}")
        
        eos_token_id = getattr(fcp.tokenizer, 'eos_token_id', 2)
        generated = input_ids[0].tolist()
        
        temperature = fcp.generation_config.get('temperature', 0.7)
        top_k = fcp.generation_config.get('top_k', 40)
        top_p = fcp.generation_config.get('top_p', 0.9)
        repetition_penalty = fcp.generation_config.get('repetition_penalty', 1.1)
        
        for step in range(max_new_tokens):
            last_token = np.array([[generated[-1]]], dtype=np.int64)
            outputs = fcp.fcp_api.infer(last_token, np.ones_like(last_token, dtype=np.int64))
            
            logits = None
            for key in outputs:
                if 'logits' in key.lower() or 'head' in key.lower():
                    logits = outputs[key]
                    break
            
            if logits is None:
                break
            
            logits_last = logits[0, -1, :] if logits.ndim == 3 else logits[0, -1]
            
            if temperature != 1.0:
                logits_last = logits_last / temperature
            
            if top_k > 0:
                indices = np.argsort(logits_last)[-top_k:]
                mask = np.zeros_like(logits_last)
                mask[indices] = logits_last[indices]
                logits_last = mask
            
            if top_p < 1.0:
                sorted_idx = np.argsort(logits_last)[::-1]
                probs = np.exp(logits_last[sorted_idx])
                cumsum = np.cumsum(probs)
                mask = cumsum > top_p
                logits_last[sorted_idx[mask]] = -np.inf
            
            if repetition_penalty != 1.1:
                for tok in set(generated):
                    if logits_last[tok] > 0:
                        logits_last[tok] /= repetition_penalty
                    else:
                        logits_last[tok] *= repetition_penalty
            
            probs = np.exp(logits_last) / np.sum(np.exp(logits_last))
            next_token = int(np.argmax(probs))
            
            generated.append(next_token)
            
            if next_token == eos_token_id:
                break
        
        response = fcp.tokenizer.decode(generated)
        logger.info(f"[FCP API GEN] Generated {len(generated)} tokens")
        
        return response
        
    except Exception as e:
        logger.error(f"[FCP API GEN] FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return _generate(fcp, prompt, max_new_tokens)


def generate_with_injection(fcp: 'FCPipeline', prompt: str, max_new_tokens: int = 1024,
                            enable_thinking: bool = True, return_metadata: bool = False) -> str:
    """
    Полнослойная инъекция (State API) - legacy метод
    Runtime State Injection: модификация Key и Value тензоров
    """
    logger.info(f"[FCP INJ] ===== START generate_with_injection =====")
    
    if not fcp.pipeline or not fcp.state_injector:
        logger.warning("[FCP INJ] No pipeline or state_injector, using fallback")
        return _generate(fcp, prompt, max_new_tokens)
    
    try:
        import openvino as ov
        
        if not (hasattr(fcp, 'tokenizer') and fcp.tokenizer):
            return _generate(fcp, prompt, max_new_tokens)
        
        input_ids = None
        tokenized = fcp.tokenizer.encode(prompt)
        
        if hasattr(tokenized, 'input_ids'):
            ids = tokenized.input_ids
            try:
                size = ids.get_size()
                shape = ids.get_shape()
                if size > 0:
                    input_ids = np.frombuffer(ids.data, dtype=np.int64, count=size)
                    if shape:
                        input_ids = input_ids.reshape(tuple(shape))
                    if input_ids.ndim == 2 and input_ids.shape[0] == 1:
                        input_ids = input_ids.reshape((1, 1, 1, input_ids.shape[1]))
            except Exception as e:
                logger.error(f"[FCP INJ] Primary method failed: {e}")
        
        if input_ids is None or input_ids.size == 0:
            raise RuntimeError("[FCP INJ] All tokenization methods failed")
        
        if input_ids.ndim == 1:
            input_ids = np.expand_dims(input_ids, axis=0)
        
        seq_len = input_ids.shape[-1]
        
        fcp.state_injector.reset_all_states()
        
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)
        
        try:
            fcp.state_injector.request.infer({
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })
            logger.info("[FCP INJ] Prefill completed")
        except Exception as e:
            logger.error(f"[FCP INJ] Infer failed: {e}")
            return _generate(fcp, prompt, max_new_tokens)
        
        generated_ids = input_ids[0].tolist()
        
        for step in range(max_new_tokens):
            fcp.state_injector.request.infer({"input_ids": np.array([[generated_ids[-1]]])})
            logits = fcp.state_injector.request.get_tensor("logits").data[0, -1]
            next_token = int(np.argmax(logits))
            
            generated_ids.append(next_token)
            
            if next_token == 2:
                break
        
        response = fcp.tokenizer.decode(generated_ids)
        
        if return_metadata:
            metadata = {
                "injection_layers": len(fcp.state_injector._layer_indices),
                "steps": step + 1
            }
            return response, metadata
        
        return response
        
    except Exception as e:
        logger.error(f"[FCP INJ] FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        try:
            return _generate(fcp, prompt, max_new_tokens)
        except Exception:
            return _generate(fcp, prompt, max_new_tokens)


def generate_streaming(fcp: 'FCPipeline', prompt: str, max_new_tokens: int = 4096,
                       enable_thinking: bool = True, enable_injection: bool = True,
                       callback=None, add_to_history: bool = True, **kwargs) -> str:
    """Потоковая генерация с поддержкой инъекции"""
    logger.info(f"[FCP STREAM] Starting: '{prompt[:30]}...'")
    
    if not fcp.pipeline:
        return "Pipeline not initialized"
    
    built_prompt = _build_prompt(fcp, prompt, enable_thinking)
    
    result = ""
    buffer = ""
    reasoning_tags = False
    
    try:
        fcp.pipeline.generate(built_prompt, max_new_tokens=max_new_tokens, streamer=fcp)
        
        for token_text in fcp.streamed_tokens:
            buffer += token_text
            
            if "<reasoning>" in buffer:
                reasoning_tags = True
            
            if reasoning_tags:
                if "</reasoning>" in buffer:
                    reasoning_text = buffer.split("</reasoning>")[0].split("<reasoning>")[1]
                    if fcp.event_queue:
                        fcp.event_queue.put({"type": "reasoning_text", "text": reasoning_text})
                        fcp.event_queue.put({"type": "reasoning_end"})
                    buffer = buffer.split("</reasoning>")[1]
                    reasoning_tags = False
            else:
                if fcp.event_queue:
                    fcp.event_queue.put({"type": "chunk", "text": token_text})
                
                result += token_text
                
                if callback:
                    callback(token_text)
        
    except Exception as e:
        logger.error(f"[FCP STREAM] Error: {e}")
        if fcp.event_queue:
            fcp.event_queue.put({"type": "error", "text": str(e)})
        return f"Generation error: {e}"
    finally:
        if fcp.event_queue:
            fcp.event_queue.put({"type": "done", "timestamp": time.time()})
    
    if add_to_history:
        fcp.conversation_history.append({
            "user": prompt,
            "assistant": result,
            "timestamp": time.time()
        })
    
    return result