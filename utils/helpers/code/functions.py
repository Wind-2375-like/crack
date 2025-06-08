import ast
import importlib
import inspect
import re
import warnings
import platform
import resource
import tempfile
import traceback
import io
import unittest
import os
import subprocess
from contextlib import contextmanager


# --- Global Cache ---
GLOBAL_SIGNATURE_CACHE = {}
GLOBAL_QUESTION_CACHE = {}

# --- 1. Dynamic API Information Fetching ---
def get_function_signature_info(canonical_function_name):
    try:
        module_path_parts = canonical_function_name.split('.')
        obj = None
        for i in range(len(module_path_parts), 0, -1):
            current_module_path = ".".join(module_path_parts[:i])
            remaining_attrs = module_path_parts[i:]
            if not current_module_path: continue
            try:
                module = importlib.import_module(current_module_path)
                temp_obj = module
                for attr_name in remaining_attrs:
                    temp_obj = getattr(temp_obj, attr_name, None)
                    if temp_obj is None: break
                if temp_obj is not None: obj = temp_obj; break
            except: continue
        if obj is None: return None

        target_callable = obj
        is_class_constructor = False
        actual_func_name_for_sig_str = canonical_function_name
        if inspect.isclass(obj):
            if hasattr(obj, '__init__') and callable(obj.__init__):
                target_callable = obj.__init__
                is_class_constructor = True
        elif not callable(obj): return None
        docstring = inspect.getdoc(obj) or ""
        try:
            sig = inspect.signature(target_callable)
            signature_string = f"{actual_func_name_for_sig_str}{str(sig)}"
            positional_params = []
            param_iter = iter(sig.parameters.items())
            if is_class_constructor:
                try:
                    first_param_name, _ = next(param_iter)
                    if first_param_name != 'self': param_iter = iter([(first_param_name, sig.parameters[first_param_name])] + list(param_iter))
                except StopIteration: pass
            for name, param in param_iter:
                if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    positional_params.append(name)
                elif param.kind == inspect.Parameter.VAR_POSITIONAL: positional_params.append(f"*{name}"); break
                else: break
            final_docstring = docstring if docstring.strip() else signature_string
            if not final_docstring.strip() and isinstance(obj, type): final_docstring = f"Class constructor for {canonical_function_name}."
            return {"signature_string": signature_string, "positional_params": positional_params, "docstring": final_docstring}
        except (ValueError, TypeError):
            if docstring:
                first_line_doc = docstring.split('\n')[0].strip()
                func_name_in_doc = module_path_parts[-1]
                match = re.match(rf"^{re.escape(func_name_in_doc)}\s*\(([^)]*)\)", first_line_doc)
                parsed_pos_params = []
                if match:
                    params_str = match.group(1)
                    raw_params = [p.split('=')[0].split(':')[0].strip() for p in params_str.split(',')]
                    for p_name in raw_params:
                        if p_name and p_name != 'self':
                            if p_name.startswith('**'): break
                            if p_name.startswith('*'): parsed_pos_params.append(p_name); break
                            if p_name == '/': continue
                            parsed_pos_params.append(p_name)
                return {"signature_string": first_line_doc if match else f"{canonical_function_name}(...)", 
                        "positional_params": parsed_pos_params if parsed_pos_params else ["<param0_doc_fallback>"], "docstring": docstring}
            return {"signature_string": f"{canonical_function_name}(...)", "positional_params": ["<param0_no_doc_fallback>"], 
                    "docstring": docstring if docstring.strip() else f"No docstring for {canonical_function_name}"}
    except Exception: return None

# --- 2. Enhanced AST Visitor ---
class EnhancedLibraryCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.module_aliases = {}
        self.from_imports = {}
        self.imported_modules_canonical = set()
        self.resolved_calls = []

    def _unparse_node(self, node):
        try: return ast.unparse(node)
        except AttributeError: return f"AST:{type(node).__name__}"
        except Exception: return "<unparse_error>"

    def _get_full_name_from_node(self, node):
        if isinstance(node, ast.Name): return node.id
        if isinstance(node, ast.Attribute):
            parts = []
            curr = node
            while isinstance(curr, ast.Attribute): parts.append(curr.attr); curr = curr.value
            if isinstance(curr, ast.Name): parts.append(curr.id); return ".".join(reversed(parts))
        return None

    def visit_Import(self, node):
        for alias in node.names:
            self.imported_modules_canonical.add(alias.name)
            if alias.asname: self.module_aliases[alias.asname] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        source_module = node.module
        if not source_module and node.level > 0 : self.generic_visit(node); return
        if not source_module and node.level == 0: self.generic_visit(node); return
        for alias_obj in node.names:
            name_in_code = alias_obj.asname if alias_obj.asname else alias_obj.name
            self.from_imports[name_in_code] = {
                'canonical_name': f"{source_module}.{alias_obj.name}",
                'source_module': source_module }
        self.generic_visit(node)

    def _resolve_call_to_canonical(self, func_node):
        name_in_code = self._get_full_name_from_node(func_node)
        if not name_in_code: return None
        parts = name_in_code.split('.')
        if len(parts) == 1:
            if parts[0] in self.from_imports:
                info = self.from_imports[parts[0]]
                return {'canonical_name': info['canonical_name'], 'source_module_for_library': info['source_module']}
            return None
        obj_part_in_code = parts[0]; method_chain_str = ".".join(parts[1:])
        if obj_part_in_code in self.module_aliases:
            base = self.module_aliases[obj_part_in_code]
            if base in self.imported_modules_canonical:
                 return {'canonical_name': f"{base}.{method_chain_str}", 'source_module_for_library': base}
        elif obj_part_in_code in self.imported_modules_canonical:
            return {'canonical_name': name_in_code, 'source_module_for_library': obj_part_in_code}
        elif obj_part_in_code in self.from_imports:
            info = self.from_imports[obj_part_in_code]
            return {'canonical_name': f"{info['canonical_name']}.{method_chain_str}", 
                    'source_module_for_library': info['source_module']}
        return None

    def visit_Call(self, node):
        resolved_info = self._resolve_call_to_canonical(node.func)
        if resolved_info:
            kw_args = {kw.arg: self._unparse_node(kw.value) for kw in node.keywords if kw.arg is not None}
            for kw in node.keywords: 
                if kw.arg is None: kw_args[f"**{self._unparse_node(kw.value)}"] = ""
            self.resolved_calls.append({
                "library_base": resolved_info['source_module_for_library'].split('.')[0],
                "canonical_function": resolved_info['canonical_name'],
                "original_positional_args": [self._unparse_node(arg) for arg in node.args],
                "original_keyword_args": kw_args })
        self.generic_visit(node)

# --- 3. Transformation Function (using cache) ---
def transform_to_knowledge_format_from_cache(resolved_calls, signature_cache):
    knowledge_list = []
    for call_data in resolved_calls:
        func_name = call_data["canonical_function"]
        sig_info = signature_cache.get(func_name)
        doc_string_to_add = "N/A (info not in cache or fetch failed)"; generalized_pos_args = call_data["original_positional_args"]
        if sig_info:
            is_failed_placeholder = sig_info.get("docstring", "").startswith("N/A (API info fetch failed") or \
                                    sig_info.get("positional_params") == ["<fetch_failed_phase1>"]
            if is_failed_placeholder:
                doc_string_to_add = sig_info.get("docstring")
            else:
                doc_string_to_add = sig_info.get("docstring", "N/A")
                if not doc_string_to_add.strip() or doc_string_to_add == "N/A":
                    doc_string_to_add = sig_info.get("signature_string", f"{func_name}(...)")
                official_pos_param_names = sig_info.get("positional_params", [])
                num_actual_pos_args = len(call_data["original_positional_args"]); current_generalized_pos_args = []
                param_idx_in_sig = 0
                for i in range(num_actual_pos_args):
                    if param_idx_in_sig < len(official_pos_param_names):
                        param_name = official_pos_param_names[param_idx_in_sig]
                        if param_name.startswith("<param") and param_name.endswith("_fallback>"):
                             current_generalized_pos_args.append(call_data["original_positional_args"][i]); param_idx_in_sig += 1; continue
                        if param_name.startswith('*') and not param_name.startswith('**'):
                            current_generalized_pos_args.append(f"{param_name[1:]}_{i - param_idx_in_sig}")
                        else: current_generalized_pos_args.append(param_name); param_idx_in_sig += 1
                    else: current_generalized_pos_args.append(f"<pos_arg_{i+1}>")
                generalized_pos_args = current_generalized_pos_args
        knowledge_list.append({"library": call_data["library_base"], "function": func_name, 
                               "positional_args": generalized_pos_args, "keyword_args": call_data["original_keyword_args"], 
                               "docstring": doc_string_to_add})
    return knowledge_list

# --- 5. Main Orchestration Function ---
def build_cache_and_generate_knowledge(dataset_items, signature_cache, 
                                       force_rebuild_cache_for_all=False):
    all_discovered_canonical_names = set(); phase1_parsed_item_data = []
    for i, item in enumerate(dataset_items):
        code_to_analyze = item
        if isinstance(item, dict): code_to_analyze = f"{item.get('code_prompt', '')}\n\n{item.get('canonical_solution', '')}"
        item_resolved_calls = []; error_parsing = None
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=SyntaxWarning)
            try:
                tree = ast.parse(code_to_analyze)
                visitor = EnhancedLibraryCallVisitor(); visitor.visit(tree)
                item_resolved_calls = visitor.resolved_calls
            except SyntaxError as e: error_parsing = f"SyntaxError: {e} (Line {e.lineno} Col {e.offset})"
            except Exception as e_p: error_parsing = f"ParsingError: {e_p}"
        phase1_parsed_item_data.append({"item_index": i, "resolved_calls": item_resolved_calls, "parsing_error": error_parsing})
        if not error_parsing:
            for call_data in item_resolved_calls: all_discovered_canonical_names.add(call_data["canonical_function"])
    for func_name in sorted(list(all_discovered_canonical_names)):
        if func_name not in signature_cache or force_rebuild_cache_for_all:
            info = get_function_signature_info(func_name)
            signature_cache[func_name] = info if info else \
                {"signature_string": f"{func_name}(...)", "positional_params": ["<fetch_failed_phase1>"], 
                 "docstring": "N/A (API info fetch failed during discovery phase)"}
    final_results_all = []
    for parsed_data in phase1_parsed_item_data:
        if parsed_data["parsing_error"]:
            final_results_all.append({"item_index": parsed_data['item_index'], "error": parsed_data["parsing_error"], "knowledge": []})
            continue
        transformed_knowledge = transform_to_knowledge_format_from_cache(parsed_data["resolved_calls"], signature_cache)
        for k in transformed_knowledge:
            k["canonical_function"] = f'{k["function"]}({", ".join(k["positional_args"])}'
            if k["keyword_args"]:
                k["canonical_function"] += f', {", ".join([f"{k}={v}" for k, v in k["keyword_args"].items()])}'
            k["canonical_function"] += ')'
        final_results_all.append({"item_index": parsed_data['item_index'], "knowledge": transformed_knowledge})
        
    return final_results_all

def process_item(item, args, chat_response_generator, facts):
    """
    Processes a single item from the dataset, extracting code and analyzing library calls.

    Args:
        item (dict): The item to process, expected to contain 'code'.
        args (Namespace): Command line arguments.
        chat_response_generator (ChatResponseGenerator): Instance for generating responses.
        facts (list): List of library call knowledge to use for processing.

    Returns:
        dict: Processed item with library call details.
    """
    code = f"{item.get('code_prompt', '')}\n\n{item.get('canonical_solution')}"
    if not code:
        return {"error": "No code to process"}
    
    probe_questions = []
    
    system_prompt = (
        "You are a helpful assistant that analyzes code and generates questions about library calls.\n"
        "Your task is to generate probe questions based on the provided code and library calls."
    )
    
    user_template = (
        "You are asked to propose a question whose answer is [ANSWER].\n"
        "Your question should start with \"Given the library (libraries) [LIB], how can we ...?\"\n\n"
        "You should never reveal the answer or its specific arguments in your question.\n\n"
        "The goal is to create a question that leads to the simplest, most basic way to call or instantiate the function/class in [ANSWER], using all default parameter values.\n"
        "- If [ANSWER] includes specific keyword arguments (e.g., `sklearn.some_func(arg1=value)`), your question MUST be phrased to necessitate those exact keyword arguments.\n"
        "- If [ANSWER] is a simple call with no keyword arguments (e.g., `sklearn.SomeClass()` or `numpy.some_func()`), your question should ask for the standard or default way to achieve the described action, implying no specific non-default parameters are needed. Do not hint at any optional parameters or their default values.\n\n"
        "You can refer to the following docstring for context on the function's purpose:\n[DOCSTRING]\n\n"
        "Use one sentence to propose the question."
    )
    
    probe_questions = []
    
    for fact in facts:
        answer = fact.get('canonical_function', '')
        knowledge = f"Function: {fact.get('canonical_function', '')}\n\nDocstring: {fact.get('docstring', '')}"
        user_input = user_template.replace("[ANSWER]", answer).replace("[LIB]", fact.get('library', '')).replace("[DOCSTRING]", fact.get('docstring', ''))
        if answer in GLOBAL_QUESTION_CACHE:
            flag = False
            for p in probe_questions:
                if p["answer"] == answer:
                    flag = True
            if flag:
                continue      
            question = GLOBAL_QUESTION_CACHE[answer]
        else:
            chat_response_generator.update_chat_history([
                ("system", system_prompt),
            ])
            response = chat_response_generator.generate_response(
                query=user_input,
                top_p=0.7,
                temperature=0.7,
                n=1,
                max_tokens=512,
            )[0]
            question = response.strip()
            GLOBAL_QUESTION_CACHE[answer] = question
        probe_questions.append({
            "question": question,
            "answer": answer,
            "knowledge": knowledge
        })
    
    # Prepare the processed item
    processed_item = {
        "probe_questions": probe_questions,
        "multihop_question": item.get('instruct_prompt', ''),
        "multihop_answer": code,
        "other_metadata": {
            "task_id": item.get('task_id', ''),
            "complete_prompt": item.get('complete_prompt', ''),
            "instruct_prompt": item.get('instruct_prompt', ''),
            "canonical_solution": item.get('canonical_solution', ''),
            "code_prompt": item.get('code_prompt', ''),
            "test": item.get('test', ''),
            "entry_point": item.get('entry_point', ''),
            "doc_struct": item.get('doc_struct', ''),
            "libs": item.get('libs', []),
        }
    }
    
    usage = chat_response_generator.get_usage()
    
    return processed_item, usage

# Define a timeout for the execution of the untrusted code
EXECUTION_TIMEOUT = 240  # Increased timeout as requested

@contextmanager
def _capture_subprocess_output():
    """
    A context manager to capture stdout/stderr from subprocesses that
    do not have a specified output stream. This version is more careful
    not to interfere with intended behavior.
    """
    original_popen = subprocess.Popen

    def patched_popen(*args, **kwargs):
        # --- SOLUTION FOR FAILURE TYPE 1 & 2 ---
        # Only redirect if stdout/stderr are not already set by the caller.
        # This respects calls like `subprocess.call(..., stdout=f)`.
        if 'stdout' not in kwargs:
            kwargs['stdout'] = subprocess.PIPE
        if 'stderr' not in kwargs:
            kwargs['stderr'] = subprocess.PIPE
        
        # --- SOLUTION FOR FAILURE TYPE 2 ---
        # Do NOT force text=True, as some libraries (e.g., ctypes, platform)
        # expect to receive raw bytes from the stream.
        
        return original_popen(*args, **kwargs)

    # We still patch os.popen as it's a special case (legacy, text-based)
    original_os_popen = os.popen
    def patched_os_popen(cmd, mode='r', buffering=-1):
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if mode == 'r':
            return proc.stdout
        elif mode == 'w':
            return proc.stdin
        return proc.stdout

    try:
        subprocess.Popen = patched_popen
        os.popen = patched_os_popen
        yield
    finally:
        subprocess.Popen = original_popen
        os.popen = original_os_popen

def unsafe_execute_worker(model_code: str, test_code: str, result_dict: dict):
    """
    A worker function that executes in a separate, sandboxed process.
    It runs the unit tests against the model's code and reports results.
    """
    # Set memory limits (e.g., 10 GB) as requested
    if platform.system() != "Windows":
        mem_limit = 10 * 1024 * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
            resource.setrlimit(resource.RLIMIT_DATA, (mem_limit, mem_limit))
        except Exception as e:
            result_dict['model_pass'] = False
            result_dict['explanation'] = f"Failed to set resource limits: {e}"
            return

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            os.chdir(temp_dir)
            full_code = model_code + "\n\n" + test_code
            execution_globals = {"__name__": "__main__"}

            try:
                compiled_code = compile(full_code, '<string>', 'exec')
            except Exception as e:
                result_dict['model_pass'] = False
                result_dict['explanation'] = f"Compilation Error:\n{traceback.format_exc()}"
                return

            log_stream = io.StringIO()
            
            with unittest.mock.patch('sys.stdout', log_stream), \
                 unittest.mock.patch('sys.stderr', log_stream), \
                 _capture_subprocess_output():
                
                exec(compiled_code, execution_globals)
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(execution_globals['TestCases'])
                runner = unittest.TextTestRunner(stream=log_stream, verbosity=2)
                test_result = runner.run(suite)

            if test_result.wasSuccessful():
                result_dict['model_pass'] = True
                result_dict['explanation'] = "All testing points passed."
            else:
                test_output = log_stream.getvalue()
                result_dict['model_pass'] = False
                result_dict['explanation'] = f"Some tests failed or errored. Full test output:\n{test_output}"

        except Exception as e:
            error_trace = traceback.format_exc()
            captured_output = log_stream.getvalue() if 'log_stream' in locals() else ""
            result_dict['model_pass'] = False
            result_dict['explanation'] = (
                f"An unexpected error occurred during execution:\n{error_trace}\n\n"
                f"Captured Output:\n{captured_output}"
            )