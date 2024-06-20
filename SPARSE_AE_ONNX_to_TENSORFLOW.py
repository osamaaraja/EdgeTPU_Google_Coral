import onnx
from onnx_tf.backend import prepare

# Load the ONNX model
onnx_model = onnx.load("SAE.onnx")

# Convert to TensorFlow
tf_rep = prepare(onnx_model)

# Export the TensorFlow model
tf_rep.export_graph("SAE_tf")

print("TensorFlow model saved.")

