from eva.core.core_brain import CoreBrain
import os, time

print("ENV:", {k: os.environ.get(k) for k in [
    "COGNIFLEX_MAP_REDUCE","COGNIFLEX_MAP_WINDOW_TOKENS","COGNIFLEX_MAP_OVERLAP",
    "COGNIFLEX_MAP_PARALLEL_WORKERS","COGNIFLEX_MAP_MAXLEN",
    "COGNIFLEX_PARALLEL_UNIFIED","COGNIFLEX_PARALLEL_SAMPLES","COGNIFLEX_PARALLEL_WORKERS","COGNIFLEX_GPU_PARALLEL_LIMIT"
]})

brain = CoreBrain({})
rg = brain.response_generator
status = rg.get_status()
print("status:", status)

ctx = " ".join(["абзац контекста" for _ in range(9000)])

t0 = time.time()
res = rg.generate_response(
    "Сделай краткое резюме по контексту.",
    max_length=120,
    task="text-generation",
    context=ctx
)
print("meta:", res.get("metadata"))
print("text_len:", len(res.get("text","")))
print("time_s:", round(time.time()-t0, 3))
