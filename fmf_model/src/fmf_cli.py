"""
FMF EVA Container - CLI Interface
 универсальный формат для запуска EVA на любой системе
"""
import sys
import os
import argparse
import json
from pathlib import Path

CONTAINER_ROOT = Path(__file__).parent
MODEL_PATH = CONTAINER_ROOT / "model.ov"
GRAPH_PATH = CONTAINER_ROOT / "data" / "graph.db"
LORA_PATH = CONTAINER_ROOT / "lora"

sys.path.insert(0, str(CONTAINER_ROOT / "src"))

def cmd_run(args):
    """Интерактивный запрос"""
    from fmf_interactive import FMFGeneratorInteractive
    
    gen = FMFGeneratorInteractive(str(MODEL_PATH), str(GRAPH_PATH), args.device)
    
    print(f"[FMF] Generating: {args.query}")
    result = gen.generate(args.query, max_tokens=args.max_tokens, enable_thinking=args.thinking)
    
    print("\n" + "="*50)
    print(result.get("response", ""))
    print("="*50)
    print(f"Latency: {result.get('latency_ms', 0)}ms")

def cmd_serve(args):
    """Запуск API сервера"""
    from fmf_server import run_server
    
    print(f"[FMF] Starting server on {args.host}:{args.port}")
    run_server(str(MODEL_PATH), str(GRAPH_PATH), args.host, args.port)

def cmd_chat(args):
    """Интерактивный чат"""
    from fmf_interactive import FMFGeneratorInteractive
    
    gen = FMFGeneratorInteractive(str(MODEL_PATH), str(GRAPH_PATH), args.device)
    
    print("[FMF] Chat mode. Type 'exit' to quit.\n")
    
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit', 'выход']:
            break
        
        result = gen.generate(query, max_tokens=args.max_tokens)
        print(f"\nFMF: {result.get('response', '')}\n")

def cmd_status(args):
    """Проверка статуса"""
    print("=== FMF EVA Status ===")
    print(f"Container: {CONTAINER_ROOT}")
    print(f"Model: {MODEL_PATH} - {'OK' if MODEL_PATH.exists() else 'MISSING'}")
    print(f"Graph: {GRAPH_PATH} - {'OK' if GRAPH_PATH.exists() else 'EMPTY'}")
    print(f"LoRA: {LORA_PATH} - {len(list(LORA_PATH.glob('*')))} files")

def main():
    parser = argparse.ArgumentParser(description="FMF EVA Container")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    subparsers.add_parser("status", help="Check status")
    
    run_parser = subparsers.add_parser("run", help="Single query")
    run_parser.add_argument("query", help="Query text")
    run_parser.add_argument("--max-tokens", type=int, default=2048)
    run_parser.add_argument("--thinking", action="store_true")
    run_parser.add_argument("--device", default="CPU")
    
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=7860)
    
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument("--max-tokens", type=int, default=2048)
    chat_parser.add_argument("--device", default="CPU")
    
    args = parser.parse_args()
    
    if args.command == "run":
        cmd_run(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
