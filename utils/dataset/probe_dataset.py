import pickle
from datasets import Dataset # Main class to inherit from
import pyarrow as pa       # For creating Arrow tables

class ProbeDataset(Dataset):
    """
    A class to represent a probe dataset.
    Reads from .pkl file containing dictionaries with a key `probe_questions`
    (list of dictionaries with keys `question` and `answer`),
    Flattens it to a list of dictionaries with keys
    - `id`: index of the original complex question structure in the input list
    - `question`: the question text
    - `answer`: the answer text
    - `complex_question_id`: the 'id' field from the original complex question structure
    """

    def __init__(self, path: str):
        # 1. Load and flatten the data using static helper methods
        # These methods prepare a list of dictionaries.
        raw_data_list = ProbeDataset._s_load_raw_data_from_pkl(path)
        flattened_records = ProbeDataset._s_flatten_raw_data(raw_data_list)

        # 2. Convert the list of dictionaries to a PyArrow Table
        arrow_table = ProbeDataset._s_convert_records_to_arrow_table(flattened_records)

        # 3. Call the parent Dataset's __init__ method with the Arrow table.
        # This properly initializes the Dataset object with your data.
        super().__init__(arrow_table=arrow_table)

        # Attributes like `self.data` (from the original code) or `self.flattened_data`
        # are no longer needed here as instance variables. The data is managed
        # internally by the parent `datasets.Dataset` class.
        # The `__len__` and `__getitem__` methods are inherited from `datasets.Dataset`
        # and will operate correctly on the Arrow table.

    @staticmethod
    def _s_load_raw_data_from_pkl(path: str):
        """Loads raw data from a pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            return []
        return data

    @staticmethod
    def _s_flatten_raw_data(raw_data_list: list):
        """
        Flattens the raw data structure into a list of records (dictionaries).
        Each record corresponds to a single probe question.
        """
        flattened_records = []
        count = 0
        for i, item in enumerate(raw_data_list):
            if not isinstance(item, dict):
                # Skip items not in expected dictionary format
                # print(f"Warning: Item at index {i} is not a dictionary, skipping.")
                continue

            # Use .get() for safer access to dictionary keys, providing defaults if necessary
            complex_id_val = i
            probe_questions_list = item.get('probe_questions', [])

            if not isinstance(probe_questions_list, list):
                # Skip if 'probe_questions' is not a list
                # print(f"Warning: 'probe_questions' for item with id {complex_id_val} is not a list, skipping.")
                continue

            for question_data in probe_questions_list:
                if not isinstance(question_data, dict):
                    # Skip malformed question data entries
                    # print(f"Warning: A question entry for id {complex_id_val} is not a dictionary, skipping.")
                    continue

                question = question_data.get('question')
                answer = question_data.get('answer')

                # Ensure essential fields are present for a valid record
                if question is None or answer is None:
                    # print(f"Warning: Missing 'question' or 'answer' for id {complex_id_val}, skipping entry.")
                    continue

                flattened_records.append({
                    'id': count,  # This is the index of the item in the original outer list from the pkl
                    'question': str(question),  # Ensure string type
                    'answer': str(answer),      # Ensure string type
                    'complex_question_id': complex_id_val,  # This is the index of the item in the original outer list from the pkl
                                                # Convert original ID to string for consistency in Arrow table;
                                                # or ensure it's of a consistent type (e.g., int) if applicable.
                })
                count += 1
        return flattened_records

    @staticmethod
    def _s_convert_records_to_arrow_table(flattened_records: list):
        """Converts a list of records (dictionaries) to a PyArrow Table."""
        if not flattened_records:
            # Define a schema for an empty table to avoid errors if data is empty
            # Adjust types if 'complex_question_id' is, for example, always an integer.
            schema = pa.schema([
                ('id', pa.int64()),
                ('question', pa.string()),
                ('answer', pa.string()),
                ('complex_question_id', pa.int64())
            ])
            # Create an empty table with this schema
            empty_arrays = [pa.array([], type=field.type) for field in schema]
            return pa.Table.from_arrays(empty_arrays, schema=schema)

        # PyArrow can often infer the schema from a list of dictionaries.
        # However, for robustness, especially with mixed types or many None values,
        # explicitly defining the schema or constructing columns one by one might be better.
        try:
            arrow_table = pa.Table.from_pylist(flattened_records)
        except pa.ArrowInvalid as e:
            # print(f"PyArrow type inference failed: {e}. Attempting explicit column construction.")
            # Fallback: More robustly build the table if from_pylist fails
            # This example assumes all records have the same keys as the first record.
            if not flattened_records: return pa.Table.from_pylist([]) # Should be caught above

            data_dict = {key: [dic.get(key) for dic in flattened_records] for key in flattened_records[0]}
            
            # Define schema and arrays, ensuring type consistency
            # (This is a simplified fallback; production code might need more nuanced type handling)
            pa_arrays = []
            fields = []

            # Example: Define types explicitly
            # Adjust types as necessary for your data.
            # 'id' is from enumerate, so int64 is good.
            # 'question' and 'answer' are text.
            # 'complex_question_id' was cast to string or None.
            type_map = {
                'id': pa.int64(),
                'question': pa.string(),
                'answer': pa.string(),
                'complex_question_id': pa.string() # Table needs a consistent type; nulls are okay.
            }

            for col_name, col_values in data_dict.items():
                pa_type = type_map.get(col_name, pa.string()) # Default to string if key not in map
                pa_arrays.append(pa.array(col_values, type=pa_type))
                fields.append(pa.field(col_name, pa_type))
            
            arrow_table = pa.Table.from_arrays(pa_arrays, schema=pa.schema(fields))
            
        return arrow_table