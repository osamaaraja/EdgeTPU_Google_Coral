import numpy as np
import pandas as pd
import tensorflow as tf


EMG_data_path = '..//..//EMG_data'
file_path = f'{EMG_data_path}/new_participant_1.0_data_1Hz.csv'

def representative_dataset_gen():

    data = pd.read_csv(file_path, low_memory=False)
    numeric_data = np.array(data.filter(like="EMG")).astype(np.float32)
    for i in range(len(numeric_data)):
        # Get sample, ensuring correct shape and type for TFLite (batch size, input_dim)
        sample = numeric_data[i:i + 1]
        # Yielding as a list of input tensors for TFLite converter
        yield [sample]

# Initialize TFLite converter
converter = tf.lite.TFLiteConverter.from_saved_model("SAE_tf")

# Enable quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set the representative dataset for quantization
converter.representative_dataset = representative_dataset_gen

# Ensure the model uses only operations supported by the Edge TPU
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Specify the input and output tensors to be int8
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert the model
tflite_quantized_model = converter.convert()

# Save the quantized model
with open("SAE_quantized.tflite", "wb") as f:
    f.write(tflite_quantized_model)

print("SAE_quantized.tflite file saved in the current directory.")
