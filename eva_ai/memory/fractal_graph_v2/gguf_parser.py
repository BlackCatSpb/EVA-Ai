"""Dummy GGUF parser - GGUF models not used (only OpenVINO)."""
import logging
logger = logging.getLogger("eva_ai.fractal_graph_v2.gguf_parser")

def parse_gguf_model(*args, **kwargs):
    logger.warning("GGUF models not supported (only OpenVINO used)")
    return None

def extract_to_graph(*args, **kwargs):
    logger.warning("GGUF models not supported (only OpenVINO used)")
    return None
