import pickle
import re
import multiprocessing
import os
from tqdm import tqdm
import sys
import unittest

# 确保脚本能找到你的工具函数
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入你原来的、安全的代码执行器
from utils.helpers.code.functions import unsafe_execute_worker, EXECUTION_TIMEOUT

def re_evaluate_item_with_all_fixes(item):
    """
    对单个条目重新运行评估，同时处理 unittest.mock 和 os.path.isfile 的问题。
    """
    # 我们只修复那些之前因为代码评估失败的条目
    if item.get("final_answer_correct") is True:
        return item
    
    # 确认这确实是一个 'code' 任务的条目
    if "os.listdir" not in item.get('question', ''): # Heuristic to identify code tasks
        return item

    unit_test_str = item.get("other_metadata", {}).get("test", "")
    model_code = "N/A"
    if item.get("model_response") and item["model_response"].strip():
        match = re.search(r"```python\s*([\s\S]*?)\s*```", item["model_response"])
        if match:
            model_code = match.group(1).strip()

    # --- 最终修复逻辑：同时处理两个问题 ---
    
    # 1. 修复 unittest.mock 的导入问题
    if "unittest.mock.patch" in unit_test_str:
        if "import unittest" not in unit_test_str:
            unit_test_str = "import unittest\n" + unit_test_str
        unit_test_str = "from unittest.mock import patch\n" + unit_test_str.replace("unittest.mock.patch", "patch")

    # 2. 如果模型代码检查了文件是否存在，就在测试中模拟这个行为
    if "os.path.isfile" in model_code:
        if "from unittest.mock import patch" not in unit_test_str:
             unit_test_str = "from unittest.mock import patch\n" + unit_test_str
        
        modified_lines = []
        for line in unit_test_str.splitlines():
            # 在每个测试函数定义前加上 isfile 的 patch
            if line.strip().startswith("def test_"):
                modified_lines.append("@patch('os.path.isfile', return_value=True)")
            modified_lines.append(line)
        unit_test_str = "\n".join(modified_lines)
    # --- 修复结束 ---

    # 使用一个独立的、安全的进程来执行测试
    with multiprocessing.Manager() as manager:
        result_dict = manager.dict()
        process = multiprocessing.Process(
            target=unsafe_execute_worker,
            args=(model_code, unit_test_str, result_dict)
        )
        process.start()
        process.join(timeout=EXECUTION_TIMEOUT)
        
        is_correct, new_explanation = False, "Patching failed: Unknown error."
        if process.is_alive():
            process.terminate()
            process.join()
            new_explanation = f"Execution timed out after {EXECUTION_TIMEOUT} seconds."
        else:
            is_correct = result_dict.get('model_pass', False)
            new_explanation = result_dict.get('explanation', f"Crashed or no explanation. Exit code: {process.exitcode}")

        item["final_answer_correct"] = is_correct
        item["final_answer_explanation"] = new_explanation
    
    return item

def patch_file(input_path):
    """加载文件，重新评估，并保存修复后的版本。"""
    if not os.path.exists(input_path):
        print(f"File not found, skipping: {input_path}")
        return

    directory, filename = os.path.split(input_path)
    new_filename = filename.replace('.pkl', '_patched.pkl')
    output_path = os.path.join(directory, new_filename)

    with open(input_path, 'rb') as f:
        data_to_fix = pickle.load(f)

    print(f"\nPatching file: {input_path}")

    # 使用一个简单的、稳定的循环，避免所有多进程冲突
    corrected_data = []
    for item in tqdm(data_to_fix, desc="Re-evaluating items"):
        corrected_item = re_evaluate_item_with_all_fixes(item)
        corrected_data.append(corrected_item)

    with open(output_path, 'wb') as f:
        pickle.dump(corrected_data, f)
    
    print(f"✅ Patched file saved to: {output_path}")

if __name__ == "__main__":
    # --- 把所有评估错误的 code 任务的结果文件路径都加到这里 ---
    files_to_patch = [
        'data/eval_results/code/injection_evaluated/base_500_gpt-4.1-mini_1.pkl',
        # ...把你所有需要修复的code任务的结果文件都加进来
    ]

    for f_path in files_to_patch:
        patch_file(f_path)