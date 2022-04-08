# Using LSTMs with dynamic sequence length in TFLite

Although TensorFlow Lite natively supports RNNs, including LSTMs, GRUs, and biderectional LSTMs, creating a TFLite model is not straightforward (with TF 2.8).
An LSTM model usually has a input shape with two dynamic dimensions: the batch dimension, and the sequence dimension.
For example, an LSTM model taking into input a sequence of 3 features (let's say a red pixel, a green pixel, and a blue pixel), has the following shape (None, None, 3).

Using the basic TFLite conversion method, we obtain a TFLite model with input shape (1,1,3) and signature (-1,-1,3).
However, when we use a sequence of length 10 (i.e., a tensor with shape (1,10,3)), TFLite complains that the second dimension is not the expected size.
From experience, it seems that TFLite still only supports at most one dynamic dimension. Resizing a TFLite tensor when two dimensions are unknown leads to segmentation faults (tf 2.8, python 3.9).

To solve this, we have to fix the batch dimension.
With a fixed batch dimension, we can easily reshape TFLite's tensor and take inputs with varying sequence length.


The following code takes as input a Keras model (Sequential or functional), a string representing the path where the tflite model will be saved, and returns the tflite model.
It fixes the batch dimension (to 1 by default), while keeping the sequence dimension dynamic.

Version without quantization, all weights are tf.float32:

```
# A function that converts a tensorflow model with LSTMs to TFLite model
def convert_LSTM_to_float_TFLite(model_tf, TFLite_target_filename, batch_size=1):
    # We need to clearly set the input signature of the Keras model
    # As of now, only one dimension can be dynamic. We must fix the batch size
    input_signature = model.layers[0].input_shape
    input_signature = (batch_size,) + input_signature[:1]
    if np.sum(np.array(input_signature)==None)!=1:
        print(f"Signature {input_signature} mus have at most one None!")
        raise Exception
    
    # Create a lambda function of the model
    run_model = tf.function(lambda x: model(x))
    # Create a concrete_function signature for the model
    # We specify a input signature where only the sequence is dynamic, the batch size is fixed
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([BATCH_SIZE, None, N_STEPS, N_FEATURES],model.inputs[0].dtype))
    # We must save the model with the signature on disk
    MODEL_DIR = "temp_model_to_convert"
    model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)
    
    # create TFLite converter from teh model saved on disk
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    # Set the optimization flag to use default quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Use default TFLite and TF operators
    converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS,tf.lite.OpsSet.TFLITE_BUILTINS]
    # use float as input and output
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    # Convert model
    model_tflite = converter.convert()
    
    open(TFLite_target_filename, "wb").write(model_tflite)
    print(f"TFLite model size: {os.path.getsize(TFLite_target_filename)}")
    return model_tflite
```


Version with quantization, weights, inputs and outputs are tf.int8:
```
# A function that converts a tensorflow model with LSTMs to TFLite model
def convert_LSTM_to_float_TFLite(model_tf, TFLite_target_filename,  X_train, batch_size=1):
    # We need to clearly set the input signature of the Keras model
    # As of now, only one dimension can be dynamic. We must fix the batch size
    input_signature = model.layers[0].input_shape
    input_signature = (batch_size,) + input_signature[:1]
    if np.sum(np.array(input_signature)==None)!=1:
        print(f"Signature {input_signature} mus have at most one None!")
        raise Exception
    
    # Create a lambda function of the model
    run_model = tf.function(lambda x: model(x))
    # Create a concrete_function signature for the model
    # We specify a input signature where only the sequence is dynamic, the batch size is fixed
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([BATCH_SIZE, None, N_STEPS, N_FEATURES],model.inputs[0].dtype))
    # We must save the model with the signature on disk
    MODEL_DIR = "temp_model_to_convert"
    model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)
    
    # create TFLite converter from teh model saved on disk
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    # Set the optimization flag to use default quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Use quantzied INT8 weights 
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    def generate_representative_dataset():
        for i in range(len(X_train)):
            yield([np.float32(X_train[i]).reshape((1,)+X_train[0].shape)])
    # Converter will use the above function to optimize quantization
    converter.representative_dataset = generate_representative_dataset
    
    # Convert model
    model_tflite = converter.convert()
    
    open(TFLite_target_filename, "wb").write(model_tflite)
    print(f"TFLite model size: {os.path.getsize(TFLite_target_filename)}")
    return model_tflite
```

Predict function, takes into input a string representing the model path, and a list X, where each element of X is a input sample

```
def predict_TFLite(model, X):
    x_data = X.copy() # the function quantizes the input, so we must make a copy
    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    outputs = []
    
    # Quantize input if needed
    input_scale, input_zero_point = input_details["quantization"]
    if (input_scale, input_zero_point) != (0.0, 0):
        for x in x_data:
            x = x / input_scale + input_zero_point
            x = x.astype(input_details["dtype"])
    
    
    for x in x_data:
        # We need to resize the input shape to fit the dynamic sequence (batch size must be equal to 1)
        interpreter.resize_tensor_input(input_details['index'], (1,)+x.shape, strict=True)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details["index"], [x])
        interpreter.invoke()
        outputs.append(np.copy(interpreter.get_tensor(output_details["index"])))
    
    
    # Dequantize output
    outputs = np.array(outputs)
    output_scale, output_zero_point = output_details["quantization"]
    if (output_scale, output_zero_point) != (0.0, 0):
        outputs = outputs.astype(np.float32)
        outputs = (outputs - output_zero_point) * output_scale
    # todo reshape output into array for each exit
    return outputs
```


