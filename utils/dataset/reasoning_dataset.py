import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import pickle
from datasets import Dataset
import pyarrow as pa
from collections import defaultdict

class ReasoningEvalDataset(Dataset):
    """
    A dataset class that processes raw data (complex questions) and probe data 
    (individual fact probes with pre-formatted knowledge strings) to create 
    items for reasoning evaluation.
    """

    def __init__(self, raw_path: str, probe_path: str):
        """
        Args:
            raw_path (str): Path to the raw_data pickle file. 
                            (Contains multihop_question, multihop_answer)
            probe_path (str): Path to the probe_data pickle file.
                            (Contains individual probe results, including complex_question_id, 
                             the pre-formatted 'knowledge' string, 'knowledgable', 
                             and 'knowledge_confidence' for each fact).
        """

        raw_data_list = self._s_load_data_from_pkl(raw_path)
        probe_data_list = self._s_load_data_from_pkl(probe_path)
        
        # No longer needs task_name
        dataset_records = self._s_prepare_dataset_records(
            raw_data_list,
            probe_data_list
        )

        arrow_table = self._s_convert_records_to_arrow_table(dataset_records)
        super().__init__(arrow_table=arrow_table)

    @staticmethod
    def _s_load_data_from_pkl(path: str):
        """Loads data from a pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data if isinstance(data, list) else []

    @staticmethod
    def _s_prepare_dataset_records(raw_data_list: list, probe_data_list: list):
        """
        Processes raw_data (complex questions) and probe_data (probed facts)
        to create the final list of records for the dataset.
        Assumes 'knowledge' string is already present in probe_data_list items.
        """
        processed_records = []

        # 1. Group probe_data items by their 'complex_question_id'
        # Each item in probe_data_list is expected to have:
        # 'complex_question_id', 'knowledge',
        # 'knowledgable', 'knowledge_confidence'.
        probes_grouped_by_complex_id = defaultdict(list)
        for probe_item in probe_data_list:
            cq_id = probe_item.get('complex_question_id')
            if cq_id is not None: # Ensure complex_question_id exists
                probes_grouped_by_complex_id[cq_id].append(probe_item)
        
        # 2. Iterate through raw_data_list (each item represents one complex question)
        for i, raw_item_data in enumerate(raw_data_list):
            complex_q_id = i 
            
            multihop_q_text = raw_item_data.get('multihop_question')
            multihop_a_text = raw_item_data.get('multihop_answer')

            if multihop_q_text is None or multihop_a_text is None:
                # print(f"Warning: Skipping raw_item at index {complex_q_id} due to missing 'multihop_question' or 'multihop_answer'.")
                continue

            # Get all probe results associated with this complex_q_id
            associated_probes = probes_grouped_by_complex_id.get(complex_q_id, [])
            
            required_knowledge_output_list = []
            for probe_detail in associated_probes:
                knowledge_str = probe_detail.get('knowledge') # Crucial: This is now expected to be pre-formatted.
                knowledgable_status = probe_detail.get('knowledgable')
                confidence = probe_detail.get('knowledge_confidence')

                # Ensure all necessary fields from the probe_detail are present
                if knowledge_str is not None and knowledgable_status is not None and confidence is not None:
                    required_knowledge_output_list.append({
                        "knowledge": str(knowledge_str),
                        "knowledgable": bool(knowledgable_status),
                        "knowledge_confidence": float(confidence)
                    })
                # else:
                    # print(f"Warning: Probe item for complex_q_id {complex_q_id} is missing 'knowledge', 'knowledgable', or 'knowledge_confidence'. Probe detail: {probe_detail}")
        
            processed_records.append({
                "id": complex_q_id, # This ID refers to the complex question's ID
                "question": str(multihop_q_text),
                "answer": str(multihop_a_text),
                "required_knowledge": required_knowledge_output_list
            })
            
        return processed_records

    @staticmethod
    def _s_convert_records_to_arrow_table(records: list):
        """Converts a list of records (dictionaries) to a PyArrow Table."""
        if not records:
            schema = pa.schema([
                ('id', pa.int64()),
                ('question', pa.string()),
                ('answer', pa.string()),
                ('required_knowledge', pa.list_(
                    pa.struct([
                        ('knowledge', pa.string()),
                        ('knowledgable', pa.bool_()),
                        ('knowledge_confidence', pa.float32())
                    ])
                ))
            ])
            return pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)

        try:
            knowledge_struct_type = pa.struct([
                pa.field('knowledge', pa.string()),
                pa.field('knowledgable', pa.bool_()),
                pa.field('knowledge_confidence', pa.float32())
            ])
            
            schema = pa.schema([
                pa.field('id', pa.int64()),
                pa.field('question', pa.string()),
                pa.field('answer', pa.string()),
                pa.field('required_knowledge', pa.list_(knowledge_struct_type))
            ])
            
            ids = pa.array([r.get('id') for r in records], type=pa.int64())
            questions = pa.array([r.get('question') for r in records], type=pa.string())
            answers = pa.array([r.get('answer') for r in records], type=pa.string())
            
            required_knowledge_data = []
            for r in records:
                knowledge_list = r.get('required_knowledge', [])
                current_item_knowledge = []
                for k_item in knowledge_list:
                    current_item_knowledge.append({
                        'knowledge': k_item.get('knowledge', ""),
                        'knowledgable': k_item.get('knowledgable', False),
                        'knowledge_confidence': float(k_item.get('knowledge_confidence', 0.0))
                    })
                required_knowledge_data.append(current_item_knowledge)

            required_knowledge_pa_array = pa.array(required_knowledge_data, type=pa.list_(knowledge_struct_type))
            
            return pa.Table.from_arrays([ids, questions, answers, required_knowledge_pa_array], schema=schema)

        except Exception as e:
            # print(f"Error converting records to Arrow table: {e}. Records example: {records[:1]}")
            empty_schema = pa.schema([
                ('id', pa.int64()), ('question', pa.string()), ('answer', pa.string()),
                ('required_knowledge', pa.list_(pa.struct([
                    ('knowledge', pa.string()), ('knowledgable', pa.bool_()), ('knowledge_confidence', pa.float32())
                ])))
            ])
            return pa.Table.from_arrays([pa.array([], type=f.type) for f in empty_schema], schema=empty_schema)