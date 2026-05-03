import torch
import nncf
from typing import Optional, Callable

import logging
logger = logging.getLogger("FCP.UES")


class QATTrainer:
    """
    Обучает модель (например, GNN) с имитацией INT8-квантования,
    чтобы сохранить точность при последующем сжатии.
    """
    
    @staticmethod
    def quantize_model(model: torch.nn.Module, 
                       calibration_dataset: Optional[Callable] = None) -> torch.nn.Module:
        """Применяет QAT к модели."""
        if calibration_dataset is None:
            # Создаём фиктивный калибровочный датасет (для GNN)
            class DummyDataset:
                def __init__(self, input_shape, num_samples=10):
                    self.input_shape = input_shape
                    self.num_samples = num_samples
                def __iter__(self):
                    for _ in range(self.num_samples):
                        yield torch.randn(*self.input_shape)
            
            calibration_dataset = DummyDataset((1, 128, 384))
        
        # Конвертируем в DataLoader для NNCF
        calib_loader = torch.utils.data.DataLoader(
            list(calibration_dataset), batch_size=1
        )
        
        # Применяем QAT
        quantized_model = nncf.quantize(
            model,
            calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            target_device=nncf.TargetDevice.CPU,
            model_type=nncf.ModelType.TRANSFORMER
        )
        return quantized_model
    
    @staticmethod
    def fine_tune(model: torch.nn.Module, 
                  train_loader: torch.utils.data.DataLoader,
                  epochs: int = 2,
                  lr: float = 1e-5):
        """Дообучает квантованную модель для восстановления точности."""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    x, y = batch[0], batch[1]
                else:
                    x = batch
                    y = torch.zeros(x.shape[0])
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"QAT fine-tuning epoch {epoch}: loss={total_loss/len(train_loader):.4f}")
        return model
