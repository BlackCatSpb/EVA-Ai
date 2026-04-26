"""
GNN to OpenVINO Converter

Заимствовано из FCP/src/fcp_gnn/convert_gnn_to_ov.py
Конвертирует PyTorch GNN модели в OpenVINO формат.
"""
import os
import torch
import numpy as np
from typing import Optional, Tuple


def convert_gnn_to_ov(
    pt_path: str,
    onnx_path: str,
    ov_path: str,
    input_dim: int = 384,
    hidden_dim: int = 512,
    output_dim: int = 2560
):
    """
    Конвертировать GNN в OpenVINO.

    Args:
        pt_path: путь к PyTorch модели
        onnx_path: путь для ONNX файла
        ov_path: путь для OpenVINO XML
        input_dim: размерность входа
        hidden_dim: скрытая размерность
        output_dim: размерность выхода
    """
    try:
        import openvino as ov
        HAS_OV = True
    except ImportError:
        HAS_OV = False
        print("OpenVINO не установлен")
        return

    from eva_ai.fcp_gnn.graph_encoder import FractalGraphEncoder

    model = FractalGraphEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )

    if os.path.exists(pt_path):
        model.load_state_dict(torch.load(pt_path, map_location="cpu"))

    model.eval()

    max_nodes = 128
    dummy_x = torch.randn(max_nodes, input_dim)
    dummy_edge = torch.randint(0, max_nodes, (2, max_nodes * 2))

    print(f"Экспорт в ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        (dummy_x, dummy_edge),
        onnx_path,
        input_names=["x", "edge_index"],
        output_names=["graph_vec", "gate_weights"],
        dynamic_axes={"x": {0: "num_nodes"}},
        opset_version=14
    )

    if HAS_OV:
        print(f"Конвертация в OpenVINO: {ov_path}")
        ov_model = ov.convert_model(onnx_path)
        ov.serialize(ov_model, ov_path)
        print(f"GNN exported to {ov_path}")
    else:
        print("Сохранён ONNX (для ручной конвертации в OpenVINO)")


class GNNEncoderOV:
    """
    OpenVINO Runtime обёртка для GNN Encoder.
    """

    def __init__(
        self,
        model_path: str = "graph_encoder.xml",
        device: str = "CPU"
    ):
        self.model_path = model_path
        self.device = device
        self._core = None
        self._compiled = None
        self._request = None

        self._init_runtime()

    def _init_runtime(self):
        """Инициализировать OpenVINO runtime."""
        try:
            import openvino as ov
            self._core = ov.Core()

            if os.path.exists(self.model_path):
                self._compiled = self._core.compile_model(self.model_path, self.device)
                self._request = self._compiled.create_infer_request()
                print(f"GNN loaded: {self.model_path}")
            else:
                print(f"Model not found: {self.model_path}")

        except ImportError:
            print("OpenVINO не установлен")

    def encode(
        self,
        node_embeddings: np.ndarray,
        edge_index: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Инференс GNN.
        """
        if self._request is None:
            return self._fallback_encode(node_embeddings, edge_index)

        import openvino as ov
        self._request.set_tensor("x", ov.Tensor(node_embeddings))
        self._request.set_tensor("edge_index", ov.Tensor(edge_index))
        self._request.infer()

        graph_vec = self._request.get_tensor("graph_vec").data
        gate_weights = self._request.get_tensor("gate_weights").data

        return graph_vec, gate_weights

    def _fallback_encode(
        self,
        x: np.ndarray,
        edge_index: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback через numpy."""
        from eva_ai.fcp_gnn.graph_encoder import GraphEncoderRuntime

        if not hasattr(self, '_runtime'):
            self._runtime = GraphEncoderRuntime()

        return self._runtime.encode(x, edge_index)


class GNNExporter:
    """
    Экспортер GNN моделей.
    """

    def __init__(
        self,
        model,
        input_dim: int = 384,
        hidden_dim: int = 512,
        output_dim: int = 2560
    ):
        self.model = model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def export_onnx(self, path: str):
        """Экспорт в ONNX."""
        import torch

        max_nodes = 128
        dummy_x = torch.randn(max_nodes, self.input_dim)
        dummy_edge = torch.randint(0, max_nodes, (2, max_nodes * 2))

        torch.onnx.export(
            self.model,
            (dummy_x, dummy_edge),
            path,
            input_names=["x", "edge_index"],
            output_names=["graph_vec", "gate_weights"],
            dynamic_axes={"x": {0: "num_nodes"}},
            opset_version=14
        )
        print(f"Exported to ONNX: {path}")

    def export_torchscript(self, path: str):
        """Экспорт в TorchScript."""
        import torch

        self.model.eval()
        scripted = torch.jit.script(self.model)
        scripted.save(path)
        print(f"Exported to TorchScript: {path}")

    def export_openvino(self, onnx_path: str, ov_path: str):
        """Экспорт в OpenVINO."""
        convert_gnn_to_ov(
            "",
            onnx_path,
            ov_path,
            self.input_dim,
            self.hidden_dim,
            self.output_dim
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert GNN to OpenVINO")
    parser.add_argument("--input", type=str, required=True, help="Path to input .pt file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    onnx_path = os.path.join(args.output, "graph_encoder.onnx")
    ov_path = os.path.join(args.output, "graph_encoder.xml")

    convert_gnn_to_ov(args.input, onnx_path, ov_path)
