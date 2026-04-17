import importlib.util
import os
import glob
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
import sys

PARSER_DIR = os.path.dirname(os.path.abspath(__file__))

def load_reasoning_parsers():
    parser_files = glob.glob(os.path.join(PARSER_DIR, "*_reasoning_parser.py"))
    
    for file_path in parser_files:
        module_name = os.path.basename(file_path).replace(".py", "")
        
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                print(f"Successfully loaded reasoning parser: {module_name}")
            except Exception as e:
                print(f"Failed to load {module_name}: {e}")

load_reasoning_parsers()