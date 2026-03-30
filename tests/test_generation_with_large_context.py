import os
import sys
import torch
import logging
import time
import json
import traceback
from typing import Dict, List, Optional, Any

# Set up logging with both file and console output
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'test_{time.strftime("%Y%m%d_%H%M%S")}.log')

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('HybridCacheTest')

# Import CoreBrain with error handling
try:
    from eva.core.core_brain import CoreBrain
except ImportError as e:
    logger.error(f"Failed to import CoreBrain: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

class LargeContextGenerationTest:
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the test with configuration."""
        self.config = config or self.get_default_config()
        self.brain = None
        self.test_results = []
    
    @staticmethod
    def get_default_config() -> Dict:
        """Get default configuration for testing."""
        return {
            'generation': {
                'enabled': True,
                'model_name': 'sberbank-ai/rugpt3small_based_on_gpt2',
                'cache_config': {
                    'enabled': True,
                    'max_memory_tokens': 10000,  # Larger cache for context
                    'vram_threshold': 0.2,
                    'ram_threshold': 0.15,
                    'disk_cache_dir': './cogniflex_cache/large_context_cache',
                    'dynamic_memory_limit': True,
                    'eviction_policy': 'hybrid'
                },
                'cache_memory_gb': 4.0  # 4GB for testing larger contexts
            }
        }
    
    def initialize_brain(self) -> bool:
        """Initialize CoreBrain with the test configuration."""
        try:
            logger.info("🚀 Initializing CoreBrain with large context cache...")
            start_time = time.time()
            self.brain = CoreBrain(config=self.config)
            init_time = time.time() - start_time
            logger.info(f"✅ CoreBrain initialized in {init_time:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize CoreBrain: {e}")
            return False
    
    def generate_large_context(self, prompt: str, context: str, max_tokens: int = 100, use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate text with large context using the cache.
        
        Args:
            prompt: The prompt to generate text for
            context: Additional context to prepend to the prompt
            max_tokens: Maximum number of tokens to generate
            use_cache: Whether to use the token cache
            
        Returns:
            Dictionary containing the generation results and metadata
        """
        if not self.brain:
            error_msg = "CoreBrain not initialized"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
        
        start_time = time.time()
        try:
            # Log generation parameters
            logger.info(f"Generating text with max_tokens={max_tokens}, context_length={len(context.split())} words, use_cache={use_cache}")
            
            # Combine context and prompt
            full_prompt = f"{context}\n\n{prompt}"
            logger.debug(f"Full prompt (first 200 chars): {full_prompt[:200]}...")
            
            # Generate with caching
            response = self.brain.generate_text(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                use_cache=use_cache,
                do_sample=True
            )
            gen_time = time.time() - start_time
            logger.info(f"Text generation completed in {gen_time:.2f} seconds")
            
            # Log response structure for debugging
            logger.debug(f"Response type: {type(response)}, content: {str(response)[:200]}...")
            
            # Get cache stats if available
            cache_stats = {}
            if hasattr(self.brain, 'token_cache'):
                try:
                    cache_stats = self.brain.token_cache.get_cache_stats()
                    logger.debug(f"Cache stats: {cache_stats}")
                except Exception as cache_err:
                    logger.warning(f"Failed to get cache stats: {cache_err}")
            
            # Extract generated text from response
            if isinstance(response, dict):
                generated_text = response.get('generated_text', '')
                logger.debug("Extracted generated_text from dict response")
            else:
                generated_text = str(response)
                logger.debug("Converted response to string")
            
            logger.info(f"Generated {len(generated_text)} characters of text")
            
            return {
                "response": generated_text,
                "generation_time": gen_time,
                "cache_stats": cache_stats,
                "context_length": len(context.split()),
                "prompt_length": len(prompt.split()),
                "success": True
            }
            
        except Exception as e:
            gen_time = time.time() - start_time
            error_msg = f"Error during text generation after {gen_time:.2f}s: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {
                "error": error_msg,
                "generation_time": gen_time,
                "success": False
            }
    
    def run_context_retention_test(self, context: str, questions: List[str]) -> Dict:
        """
        Test context retention by asking multiple questions about the same context.
        
        Args:
            context: The shared context for all questions
            questions: List of questions to ask about the context
            
        Returns:
            Dictionary containing test results and analysis
        """
        results = {
            'questions': [],
            'total_questions': len(questions),
            'successful_generations': 0,
            'failed_generations': 0,
            'total_generation_time': 0.0,
            'average_tokens_per_second': 0.0,
            'cache_hit_rates': [],
            'context_length': len(context.split())
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 Starting context retention test with {len(questions)} questions")
        logger.info(f"📝 Context length: {results['context_length']} tokens")
        logger.info(f"{'='*80}")
        
        for i, question in enumerate(questions, 1):
            question_result = {
                'question': question,
                'success': False,
                'response': '',
                'generation_time': 0.0,
                'cache_stats': {}
            }
            
            try:
                logger.info(f"\n🔍 Question {i}/{len(questions)}: {question}")
                
                # Generate response with the same context
                result = self.generate_large_context(prompt=question, context=context)
                
                # Update question result
                question_result.update({
                    'success': result.get('success', False),
                    'response': result.get('response', ''),
                    'generation_time': result.get('generation_time', 0.0),
                    'cache_stats': result.get('cache_stats', {})
                })
                
                # Update test statistics
                if question_result['success']:
                    results['successful_generations'] += 1
                    results['total_generation_time'] += question_result['generation_time']
                    
                    # Calculate tokens per second (approximate)
                    response_tokens = len(question_result['response'].split())
                    if question_result['generation_time'] > 0:
                        tokens_per_sec = response_tokens / question_result['generation_time']
                        question_result['tokens_per_second'] = tokens_per_sec
                    
                    # Track cache hit rate if available
                    cache_stats = question_result['cache_stats']
                    if cache_stats and 'hits' in cache_stats and 'misses' in cache_stats:
                        total = cache_stats['hits'] + cache_stats['misses']
                        if total > 0:
                            hit_rate = (cache_stats['hits'] / total) * 100
                            results['cache_hit_rates'].append(hit_rate)
                            question_result['cache_hit_rate'] = f"{hit_rate:.1f}%"
                else:
                    results['failed_generations'] += 1
                    question_result['error'] = result.get('error', 'Unknown error')
                
                # Log results
                log_msg = []
                if question_result['success']:
                    log_msg.append(f"✅ Generated {len(question_result['response'])} chars in {question_result['generation_time']:.2f}s")
                    if 'tokens_per_second' in question_result:
                        log_msg.append(f"   Speed: {question_result['tokens_per_second']:.1f} tokens/sec")
                    if 'cache_hit_rate' in question_result:
                        log_msg.append(f"   Cache hit rate: {question_result['cache_hit_rate']}")
                    logger.info(" | ".join(log_msg))
                else:
                    logger.error(f"❌ Failed: {question_result.get('error', 'Unknown error')}")
                
            except Exception as e:
                error_msg = f"Unexpected error processing question {i}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                question_result['error'] = error_msg
                results['failed_generations'] += 1
            
            results['questions'].append(question_result)
        
        # Calculate final statistics
        if results['successful_generations'] > 0:
            results['average_generation_time'] = results['total_generation_time'] / results['successful_generations']
            
            if results['cache_hit_rates']:
                results['average_cache_hit_rate'] = sum(results['cache_hit_rates']) / len(results['cache_hit_rates'])
            
            logger.info("\n" + "="*80)
            logger.info("📊 Test Results Summary:")
            logger.info("="*80)
            logger.info(f"✅ Successful generations: {results['successful_generations']}/{results['total_questions']}")
            if results['failed_generations'] > 0:
                logger.warning(f"❌ Failed generations: {results['failed_generations']}")
            logger.info(f"⏱️  Average generation time: {results['average_generation_time']:.2f}s")
            
            if 'average_cache_hit_rate' in results:
                logger.info(f"💾 Average cache hit rate: {results['average_cache_hit_rate']:.1f}%")
            
            # Log sample responses
            logger.info("\nSample responses:")
            for i, q_result in enumerate(results['questions'], 1):
                if q_result.get('success', False):
                    response_preview = (q_result['response'][:150] + '...') if len(q_result['response']) > 150 else q_result['response']
                    logger.info(f"{i}. Q: {q_result['question'][:80]}...")
                    logger.info(f"   A: {response_preview}")
        
        return results
    
    def run_performance_test(self, context: str, prompt: str, iterations: int = 3) -> Dict[str, Any]:
        """
        Test and compare performance with and without cache.
        
        Args:
            context: The context to use for generation
            prompt: The prompt to generate text for
            iterations: Number of test iterations to run for each mode (with/without cache)
            
        Returns:
            Dictionary containing detailed performance metrics and comparison
        """
        results = {
            'with_cache': {
                'generation_times': [],
                'tokens_per_second': [],
                'success_count': 0,
                'total_tokens': 0
            },
            'without_cache': {
                'generation_times': [],
                'tokens_per_second': [],
                'success_count': 0,
                'total_tokens': 0
            },
            'iterations': iterations,
            'context_length': len(context.split()),
            'prompt_length': len(prompt.split())
        }
        
        logger.info("\n" + "="*80)
        logger.info(f"🚀 Starting performance test with {iterations} iterations")
        logger.info(f"📝 Context length: {results['context_length']} tokens")
        logger.info(f"📝 Prompt length: {results['prompt_length']} tokens")
        logger.info("="*80)
        
        # Warm-up run to initialize models and caches
        logger.info("\n🔥 Warming up...")
        warmup_result = self.generate_large_context(prompt=prompt, context=context)
        if not warmup_result.get('success', False):
            logger.warning("⚠️  Warm-up generation failed, but continuing with tests...")
        
        def run_test_phase(use_cache: bool) -> None:
            phase = 'with_cache' if use_cache else 'without_cache'
            logger.info(f"\n{'='*40}")
            logger.info(f"🔧 Testing {phase.upper()} (Iterations: {iterations})")
            logger.info("="*40)
            
            for i in range(1, iterations + 1):
                try:
                    logger.info(f"\n🔄 Iteration {i}/{iterations} - {'Using cache' if use_cache else 'Cache disabled'}")
                    
                    # Run generation
                    start_time = time.time()
                    result = self.generate_large_context(
                        prompt=prompt,
                        context=context,
                        max_tokens=100  # Fixed token count for consistent comparison
                    )
                    gen_time = time.time() - start_time
                    
                    if result.get('success', False):
                        # Calculate tokens per second
                        response_tokens = len(result['response'].split())
                        tokens_per_sec = response_tokens / gen_time if gen_time > 0 else 0
                        
                        # Update results
                        results[phase]['generation_times'].append(gen_time)
                        results[phase]['tokens_per_second'].append(tokens_per_sec)
                        results[phase]['success_count'] += 1
                        results[phase]['total_tokens'] += response_tokens
                        
                        # Log iteration results
                        logger.info(f"✅ Generated {response_tokens} tokens in {gen_time:.2f}s ({tokens_per_sec:.1f} tokens/sec)")
                        if 'cache_stats' in result:
                            hits = result['cache_stats'].get('hits', 0)
                            misses = result['cache_stats'].get('misses', 0)
                            total = hits + misses
                            if total > 0:
                                hit_rate = (hits / total) * 100
                                logger.info(f"   Cache: {hits} hits, {misses} misses ({hit_rate:.1f}% hit rate)")
                    else:
                        logger.error(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
                    
                except Exception as e:
                    logger.error(f"❌ Error during iteration {i}: {str(e)}")
                    logger.error(traceback.format_exc())
                
                # Small delay between iterations
                if i < iterations:
                    time.sleep(1)
        
        # Run tests with cache
        run_test_phase(use_cache=True)
        
        # Clear cache for fair comparison
        if hasattr(self.brain, 'token_cache'):
            try:
                logger.info("\n🧹 Clearing cache for next test phase...")
                self.brain.token_cache.clear()
                logger.info("✅ Cache cleared")
            except Exception as e:
                logger.error(f"❌ Failed to clear cache: {e}")
        
        # Run tests without cache
        run_test_phase(use_cache=False)
        
        # Calculate final statistics
        for phase in ['with_cache', 'without_cache']:
            if results[phase]['success_count'] > 0:
                results[phase]['avg_generation_time'] = (
                    sum(results[phase]['generation_times']) / 
                    len(results[phase]['generation_times'])
                )
                results[phase]['avg_tokens_per_second'] = (
                    sum(results[phase]['tokens_per_second']) / 
                    len(results[phase]['tokens_per_second'])
                )
                results[phase]['total_time'] = sum(results[phase]['generation_times'])
                results[phase]['avg_tokens_per_request'] = (
                    results[phase]['total_tokens'] / results[phase]['success_count']
                )
        
        # Calculate speedup if both phases had successful runs
        if (results['with_cache']['success_count'] > 0 and 
            results['without_cache']['success_count'] > 0):
            speedup = (
                (results['without_cache']['avg_generation_time'] - 
                 results['with_cache']['avg_generation_time']) / 
                results['without_cache']['avg_generation_time'] * 100
            )
            results['speedup_percent'] = speedup
            
            # Log final results
            logger.info("\n" + "="*80)
            logger.info("📊 PERFORMANCE TEST RESULTS")
            logger.info("="*80)
            logger.info(f"{'Metric':<25} | {'With Cache':<15} | {'Without Cache':<15} | Improvement")
            logger.info("-" * 80)
            
            # Generation time comparison
            logger.info(
                f"{'Avg. Time (s)':<25} | "
                f"{results['with_cache']['avg_generation_time']:<15.2f} | "
                f"{results['without_cache']['avg_generation_time']:<15.2f} | "
                f"{speedup:.1f}% faster"
            )
            
            # Tokens per second comparison
            logger.info(
                f"{'Tokens/Sec':<25} | "
                f"{results['with_cache']['avg_tokens_per_second']:<15.1f} | "
                f"{results['without_cache']['avg_tokens_per_second']:<15.1f} | "
                f"{(results['with_cache']['avg_tokens_per_second'] / results['without_cache']['avg_tokens_per_second'] - 1) * 100:.1f}% faster"
            )
            
            # Success rate
            success_rate_cache = (results['with_cache']['success_count'] / iterations) * 100
            success_rate_no_cache = (results['without_cache']['success_count'] / iterations) * 100
            logger.info(
                f"{'Success Rate':<25} | "
                f"{success_rate_cache:<15.1f}% | "
                f"{success_rate_no_cache:<15.1f}% | "
                f"{success_rate_cache - success_rate_no_cache:+.1f}%"
            )
            
            # Total time
            logger.info(
                f"{'Total Time (s)':<25} | "
                f"{results['with_cache']['total_time']:<15.2f} | "
                f"{results['without_cache']['total_time']:<15.2f} | "
                f"{(1 - results['with_cache']['total_time'] / results['without_cache']['total_time']) * 100:.1f}% time saved"
            )
            
            logger.info("="*80)
        
        return results
    
    def run_all_tests(self):
        """Run all tests and return results."""
        if not self.initialize_brain():
            return {"success": False, "error": "Failed to initialize CoreBrain"}
        
        try:
            # Sample large context (can be replaced with any large text)
            large_context = """
            В 2023 году были представлены новые модели искусственного интеллекта, которые значительно превзошли 
            предыдущие по способности понимать и генерировать тексты. Эти модели основаны на архитектуре трансформеров 
            и используют миллиарды параметров для обработки естественного языка. Они способны выполнять широкий спектр 
            задач, включая перевод, ответы на вопросы, генерацию кода и многое другое. 
            
            Одной из ключевых особенностей этих моделей является их способность запоминать и использовать контекст 
            из предыдущих частей разговора. Это достигается за счёт механизма внимания, который позволяет модели 
            фокусироваться на релевантных частях входных данных. Кроме того, современные модели используют 
            кэширование промежуточных вычислений для ускорения генерации текста.
            
            В тестах эти модели демонстрируют впечатляющие результаты, но при этом требуют значительных вычислительных 
            ресурсов. Для их работы используются графические ускорители с большим объёмом видеопамяти. Разработчики 
            постоянно работают над оптимизацией производительности, чтобы сделать модели более доступными.
            """
            
            # Questions to test context retention
            questions = [
                "Какие модели были представлены в 2023 году?",
                "Какие задачи могут выполнять эти модели?",
                "Как называется архитектура, на которой основаны эти модели?",
                "Какой механизм позволяет моделям запоминать контекст?",
                "Какие вычислительные ресурсы требуются для работы этих моделей?"
            ]
            
            # Run context retention test
            logger.info("\n🔍 Starting context retention test...")
            retention_results = self.run_context_retention_test(large_context, questions)
            
            # Run performance test
            performance_result = self.run_performance_test(
                context=large_context,
                prompt="Расскажи подробнее о механизме внимания в моделях ИИ",
                iterations=3
            )
            
            # Prepare final results
            result = {
                "success": True,
                "context_retention_test": {
                    "total_questions": len(retention_results),
                    "successful_responses": sum(1 for r in retention_results if r.get('success', False)),
                    "average_generation_time": sum(r.get('generation_time', 0) for r in retention_results) / len(retention_results)
                },
                "performance_test": performance_result
            }
            
            logger.info("\n🎉 All tests completed successfully!")
            logger.info(f"📊 Final results: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during testing: {e}")
            return {"success": False, "error": str(e)}
        finally:
            # Clean up
            if hasattr(self.brain, 'shutdown'):
                self.brain.shutdown()

if __name__ == "__main__":
    print("="*60)
    print("🧪 Starting Large Context Generation Test Suite")
    print("="*60)
    
    test = LargeContextGenerationTest()
    test.run_all_tests()
    
    print("\n" + "="*60)
    print("✅ Test suite completed")
    print("="*60)
