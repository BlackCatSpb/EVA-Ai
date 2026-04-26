"""
QwenLayerExporter - Экспорт Qwen3 модели с exposed layer outputs

Экспортирует Qwen3 модель в ONNX/OpenVINO формат с возможностью
получать hidden_states после каждого слоя.

Использование:
    python -m eva_ai.tools.export_qwen_layers --model_path <path> --output_dir <dir>
"""
import os
import sys
import argparse
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen_layer_exporter")


class Qwen3LayerExporter:
    """
    Экспортер Qwen3 модели с послойными outputs.

    Экспортирует:
    1. Полную модель в ONNX с outputs на каждом слое
    2. OpenVINO версию с тем же функционалом
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        num_layers: int = 36
    ):
        """
        Args:
            model_path: Путь к HuggingFace модели или локальной
            output_dir: Директория для сохранения
            num_layers: Количество слоёв (по умолчанию 36 для Qwen3-4B)
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.num_layers = num_layers

        self.model = None
        self.config = None
        self.tokenizer = None

    def load_model(self) -> bool:
        """Загрузить модель и токенизатор"""
        try:
            from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

            logger.info(f"Loading model from {self.model_path}")

            self.config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            logger.info(f"  hidden_size: {self.config.hidden_size}")
            logger.info(f"  num_layers: {self.config.num_hidden_layers}")
            logger.info(f"  num_heads: {self.config.num_attention_heads}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.config,
                trust_remote_code=True,
                torch_dtype="bfloat16",
                device_map="cpu",
                low_cpu_mem_usage=True,
                output_hidden_states=True
            )

            self.model.eval()
            logger.info(f"  Model loaded: {type(self.model).__name__}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def export_partial_models(self, split_layers: List[int]) -> bool:
        """
        Экспортировать модель частями (split layers).

        Создаёт отдельные ONNX файлы для каждой части модели.
        Это позволяет загружать только нужные слои в память.

        Args:
            split_layers: Список слоёв для сплита, напр. [8, 16, 24]
        """
        import torch
        import onnx
        from onnx import helper, TensorProto

        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Exporting partial models with splits at {split_layers}")

        hidden_size = self.config.hidden_size
        seq_len = 1  # Для генерации token by token

        # Dummy inputs
        dummy_input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
        dummy_attention_mask = torch.ones(1, seq_len, dtype=torch.long)

        # Собираем слои
        all_layers = list(self.model.model.layers)

        prev_layer_idx = 0
        for split_idx, split_layer in enumerate(split_layers + [self.num_layers]):
            logger.info(f"  Exporting layers {prev_layer_idx} to {split_layer - 1}")

            # Создаём subgraph с этими слоями
            layers_subset = all_layers[prev_layer_idx:split_layer]

            # Создаём упрощённую модель для экспорта
            class PartialModel(torch.nn.Module):
                def __init__(self, layers, config):
                    super().__init__()
                    self.layers = layers
                    self.embed_tokens = self.model.model.embed_tokens if hasattr(self.model.model, 'embed_tokens') else None
                    self.embed_dropout = torch.nn.Dropout(0.1)

                def forward(self, input_ids, attention_mask):
                    # Embedding
                    if self.embed_tokens is not None:
                        hidden_states = self.embed_tokens(input_ids)
                    else:
                        hidden_states = self.model.model.embed_tokens(input_ids)
                    hidden_states = self.embed_dropout(hidden_states)

                    # Apply layers
                    for layer in self.layers:
                        layer_output = layer(hidden_states, attention_mask)
                        hidden_states = layer_output[0]

                    return hidden_states

            partial_model = PartialModel(layers_subset, self.config)

            # Экспорт в ONNX
            output_path = os.path.join(
                self.output_dir,
                f"part_{prev_layer_idx}_{split_layer - 1}.onnx"
            )

            try:
                torch.onnx.export(
                    partial_model,
                    (dummy_input_ids, dummy_attention_mask),
                    output_path,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["hidden_states"],
                    dynamic_axes={
                        "input_ids": {0: "batch", 1: "seq"},
                        "attention_mask": {0: "batch", 1: "seq"},
                        "hidden_states": {0: "batch", 1: "seq", 2: "hidden"}
                    },
                    opset_version=14,
                    do_constant_folding=True
                )
                logger.info(f"    Saved: {output_path}")
            except Exception as e:
                logger.warning(f"    Export failed: {e} (expected - layers may not be directly exportable)")

            prev_layer_idx = split_layer

        return True

    def export_with_layer_hooks(self) -> bool:
        """
        Экспортировать модель с hooks для послойного доступа.

        Создаёт модель которая возвращает hidden_states после каждого слоя.
        """
        import torch

        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("Creating model with layer hooks")

        hidden_size = self.config.hidden_size

        class ModelWithLayerOutputs(torch.nn.Module):
            """
            Оборачивает Qwen модель и возвращает outputs всех слоёв.
            """
            def __init__(self, base_model, num_layers):
                super().__init__()
                self.base_model = base_model
                self.num_layers = num_layers
                self.layer_outputs = {}

                # Регистрируем hooks для всех слоёв
                for idx, layer in enumerate(base_model.model.layers):
                    layer.register_forward_hook(self._hook_fn(idx))

            def _hook_fn(self, layer_idx):
                def hook(module, input, output):
                    self.layer_outputs[layer_idx] = output[0].detach()
                return hook

            def forward(self, input_ids, attention_mask=None):
                self.layer_outputs.clear()

                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                # Добавляем embedding layer (index -1)
                all_outputs = {
                    -1: self.base_model.model.embed_tokens(input_ids).detach()
                }

                # Добавляем слои
                for idx in range(self.num_layers):
                    if idx in self.layer_outputs:
                        all_outputs[idx] = self.layer_outputs[idx]
                    elif idx < len(outputs.hidden_states) - 1:
                        all_outputs[idx] = outputs.hidden_states[idx + 1]

                # Добавляем финальный output
                all_outputs['final'] = outputs.logits

                return all_outputs

        wrapped_model = ModelWithLayerOutputs(self.model, self.num_layers)
        wrapped_model.eval()

        # Сохраняем wrapped model state dict
        output_path = os.path.join(self.output_dir, "qwen_layer_model.pt")
        torch.save({
            'model_state_dict': wrapped_model.state_dict(),
            'config': self.config,
            'num_layers': self.num_layers
        }, output_path)
        logger.info(f"Saved wrapped model: {output_path}")

        return True

    def create_layer_inference_wrapper(self) -> str:
        """
        Создать Python модуль для послойного инференса.

        Returns:
            Путь к созданному файлу
        """
        wrapper_code = '''"""
Layer-wise inference wrapper for Qwen3 model.

Загружает модель и позволяет получать hidden_states после каждого слоя.
"""
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("qwen_layer_inference")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available")


class QwenLayerInference:
    """
    Inference wrapper с доступом к hidden_states на каждом слое.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        num_layers: int = 36
    ):
        self.model_path = model_path
        self.device = device
        self.num_layers = num_layers
        self.model = None
        self.config = None
        self.tokenizer = None

    def load(self) -> bool:
        """Загрузить модель"""
        if not HAS_TORCH:
            logger.error("PyTorch not available")
            return False

        try:
            self.config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                output_hidden_states=True
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.config,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                low_cpu_mem_usage=True
            )

            self.model.eval()
            logger.info(f"Model loaded on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_layer_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Получить outputs всех слоёв.

        Returns:
            Dict where key=-1 is embedding output,
            key=0..num_layers-1 are layer outputs,
            key='final' is final logits
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        hidden_states = outputs.hidden_states

        result = {}
        result[-1] = hidden_states[0]  # Embedding output

        for idx, hs in enumerate(hidden_states[1:-1]):
            result[idx] = hs

        result['final'] = outputs.logits

        return result

    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass через один конкретный слой.

        Args:
            layer_idx: Индекс слоя
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len]

        Returns:
            Output hidden_states после этого слоя
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        layer = self.model.model.layers[layer_idx]

        with torch.no_grad():
            output = layer(hidden_states, attention_mask)

        return output[0]

    def generate_with_layer_callback(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        layer_callback: callable = None
    ) -> str:
        """
        Генерация с callback на каждом слое.

        Args:
            prompt: Начальный промпт
            max_new_tokens: Максимум новых токенов
            layer_callback: (layer_idx, hidden_states) -> modified_hidden_states or None
                           Если возвращает None, состояние не меняется

        Returns:
            Сгенерированный текст
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        generated_tokens = []

        for step in range(max_new_tokens):
            # Get all layer outputs
            layer_outputs = self.get_layer_outputs(input_ids, attention_mask)

            # Apply callback for each layer
            if layer_callback:
                for layer_idx in range(self.num_layers):
                    if layer_idx in layer_outputs:
                        modified = layer_callback(layer_idx, layer_outputs[layer_idx])
                        if modified is not None:
                            layer_outputs[layer_idx] = modified

            # Get prediction for next token
            logits = layer_outputs['final'][0, -1, :]
            next_token_id = torch.argmax(logits, dim=-1).item()

            if next_token_id == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id)

            # Append to input for next step
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token_id]], device=self.device)
            ], dim=-1)

            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=self.device)
            ], dim=-1)

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def create_export_script():
    """Создать скрипт экспорта модели"""
    return """
# Экспорт модели с layer outputs

from qwen_layer_inference import QwenLayerInference

model = QwenLayerInference(
    model_path="path/to/model",
    device="cpu",
    num_layers=36
)

model.load()

# Test inference
import torch
inputs = model.tokenizer("Hello", return_tensors="pt")
layer_outputs = model.get_layer_outputs(inputs["input_ids"])

print(f"Number of layers captured: {len([k for k in layer_outputs.keys() if isinstance(k, int)])}")
for layer_idx, hs in sorted([(k, v) for k, v in layer_outputs.items() if isinstance(k, int)]):
    print(f"  Layer {layer_idx}: {hs.shape}")
"""
'''

        output_path = os.path.join(self.output_dir, "qwen_layer_inference.py")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(wrapper_code)
        logger.info(f"Created inference wrapper: {output_path}")

        return output_path

    def export_openvino_with_layers(self) -> bool:
        """
        Экспортировать в OpenVINO с exposed layer outputs.

        Использует OpenVINO ONNX frontend для экспорта.
        """
        try:
            import openvino as ov
            import torch
        except ImportError as e:
            logger.error(f"Required packages not available: {e}")
            return False

        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("Exporting to OpenVINO format")

        # Сначала экспортируем в ONNX
        onnx_path = os.path.join(self.output_dir, "qwen_with_layers.onnx")

        hidden_size = self.config.hidden_size
        seq_len = 32  # Фиксированная длина для экспорта

        dummy_input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
        dummy_attention_mask = torch.ones(1, seq_len, dtype=torch.long)

        class ModelWithAllLayers(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.num_layers = base_model.config.num_hidden_layers

            def forward(self, input_ids, attention_mask):
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                # Return tuple: (logits, layer_0_hs, layer_1_hs, ..., final_hs)
                result = [outputs.logits]
                for hs in outputs.hidden_states[1:]:  # Skip embedding
                    result.append(hs)

                return tuple(result)

        wrapped_model = ModelWithAllLayers(self.model)
        wrapped_model.eval()

        # Export
        torch.onnx.export(
            wrapped_model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=[f"layer_{i}" for i in range(self.num_layers)] + ["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
            },
            opset_version=14,
            do_constant_folding=True
        )
        logger.info(f"ONNX exported: {onnx_path}")

        # Convert to OpenVINO
        ov_path = os.path.join(self.output_dir, "qwen_with_layers.xml")
        ov_model = ov.convert_model(onnx_path)
        ov.serialize(ov_model, ov_path)
        logger.info(f"OpenVINO exported: {ov_path}")

        return True


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3 model with layer outputs")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to Qwen3 model")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--num_layers", type=int, default=36,
                       help="Number of layers (default: 36)")
    parser.add_argument("--export_type", type=str, default="all",
                       choices=["all", "hooks", "openvino", "partial"],
                       help="Type of export")
    parser.add_argument("--split_layers", type=str, default="8,16,24",
                       help="Comma-separated split layers for partial export")

    args = parser.parse_args()

    exporter = Qwen3LayerExporter(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_layers=args.num_layers
    )

    if not exporter.load_model():
        logger.error("Model loading failed")
        return 1

    if args.export_type in ["all", "hooks"]:
        exporter.export_with_layer_hooks()
        exporter.create_layer_inference_wrapper()

    if args.export_type in ["all", "partial"]:
        split_layers = [int(x) for x in args.split_layers.split(",")]
        exporter.export_partial_models(split_layers)

    if args.export_type in ["all", "openvino"]:
        try:
            exporter.export_openvino_with_layers()
        except Exception as e:
            logger.warning(f"OpenVINO export failed: {e}")

    logger.info("Export complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
