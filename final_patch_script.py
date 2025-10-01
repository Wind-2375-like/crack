import pickle
import re
import multiprocessing
import os
from tqdm import tqdm
import sys
import unittest

# 确保脚本能找到你的工具函数 (utils.helpers.code)
# 这段代码会自动将项目根目录添加到Python的搜索路径中
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    # 导入我们已经修复好的代码执行器
    from utils.helpers.code.functions import unsafe_execute_worker, EXECUTION_TIMEOUT
except ImportError:
    print("❌ 错误：无法找到 'utils/helpers/code/functions.py'。")
    print("请确保你已经用我之前提供的最终正确版本替换了该文件，并且此补丁脚本位于你的项目根目录中。")
    sys.exit(1)


def re_evaluate_item(item):
    """
    对单个条目重新运行评估。
    """
    explanation = item.get('final_answer_explanation', '')
    
    # 我们只修复那些之前因为 'unittest.mock' 错误而失败的条目
    if "module 'unittest' has no attribute 'mock'" not in explanation:
        return item

    unit_test_str = item.get("other_metadata", {}).get("test", "")
    model_code = "N/A"
    if item.get("model_response") and item["model_response"].strip():
        match = re.search(r"```python\s*([\s\S]*?)\s*```", item["model_response"])
        if match:
            model_code = match.group(1).strip()

    # unsafe_execute_worker 函数本身现在已经包含了所有的修复逻辑
    # 我们只需要直接调用它即可
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

        # 用新的、正确的结果更新条目
        item["final_answer_correct"] = is_correct
        item["final_answer_explanation"] = new_explanation
    
    return item

def patch_file(input_path):
    """加载文件，逐个重新评估，并保存修复后的版本。"""
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
        corrected_item = re_evaluate_item(item)
        corrected_data.append(corrected_item)

    with open(output_path, 'wb') as f:
        pickle.dump(corrected_data, f)
    
    print(f"✅ Patched file saved to: {output_path}")

if __name__ == "__main__":
    # --- 在这里列出所有评估结果错误的文件 ---
    files_to_patch = [
        'data/eval_results/code/injection_evaluated/original_500_gpt-4.1-mini_1.pkl',
        'data/eval_results/code/injection_evaluated/original_500_o4-mini_1.pkl',
        'data/eval_results/code/injection_evaluated/base_500_gpt-4.1-mini_1.pkl',
        'data/eval_results/code/injection_evaluated/base_500_o4-mini_1.pkl',
        'data/eval_results/code/injection_evaluated/base_500_gpt-4.1-mini_10.pkl',
        'data/eval_results/code/injection_evaluated/base_500_o4-mini_10.pkl',
        # ... 把你所有需要修复的 code 任务的结果文件都加进来
    ]

    print("--- Starting evaluation patch process ---")
    for f_path in files_to_patch:
        patch_file(f_path)
    print("\n--- All specified files have been patched. ---")