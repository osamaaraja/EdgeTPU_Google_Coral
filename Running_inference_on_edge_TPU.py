import numpy as np
import pandas as pd
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
import time

EMG_data_path = '..//..//EMG_data'
file_path = f'{EMG_data_path}/new_participant_4.0_data_1Hz.csv'


# Load the model
interpreter = make_interpreter("SAE_quantized_edgetpu.tflite")
interpreter.allocate_tensors()

# Load and prepare the input data

data = pd.read_csv(file_path, low_memory=False, nrows=num_rows)
numeric_data = np.array(data.filter(like="EMG"))

numeric_data = (numeric_data/10) * 127
input_data = numeric_data.astype(np.int8)

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Measure inference time
start_time = time.time()

for i in range(len(input_data)):
    print(f"Sample {i}:")
    # Reshape each sample to (1, 16) to match the expected input shape
    sample = np.expand_dims(input_data[i], axis=0)

    # Directly set the input tensor
    input_tensor_index = interpreter.get_input_details()[0]['index']
    interpreter.tensor(input_tensor_index)()[:, :] = sample

    interpreter.invoke()

    # Fetch outputs
    reconstructed_data = common.output_tensor(interpreter, 1).copy()
    encoded_features = common.output_tensor(interpreter, 0).copy()


end_time = time.time()
print(f"Inference time for {num_rows} samples:", end_time - start_time, "seconds")
