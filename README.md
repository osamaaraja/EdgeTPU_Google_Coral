# Model Conversion and Quantization

### Description

This repository contains files that convert the original Deep learning model implemented in Pytorch code into TensorFlow using ONNX. The process includes converting the model to TensorFlow Lite format and applying post-training quantization to convert the model into int8 format. Finally, the quantized TensorFlow Lite model is converted to a format compatible with Coral Edge TPU using publicly available Google Colab code.

You can read more about Coral Edge TPU [here](https://coral.ai/docs/accelerator/get-started).

### Workflow

1. **Conversion to ONNX:**
   - Convert the original Pytorch model to ONNX format.

2. **Conversion to TensorFlow:**
   - Convert the ONNX model to TensorFlow format.

3. **TensorFlow Lite Conversion with Quantization:**
   - Convert the TensorFlow model to TensorFlow Lite format and apply post-training quantization to convert the model to int8 format.

4. **Edge TPU Compilation:**
   - Use Google Colab code to compile the quantized TensorFlow Lite model for Coral Edge TPU compatibility.

### Resources

- [ONNX](https://onnx.ai/)
- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Coral Edge TPU](https://coral.ai/)


