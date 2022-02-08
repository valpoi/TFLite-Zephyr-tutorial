# How to Use Tensorflow Lite for Microcontrollers

### Install the requirements

Download and install Tensorflow and Tensorflow Lite

```
git clone https://github.com/tensorflow/tflite-micro
pip install tensorflow tensorflow-model-optimization #alternatively pip3
```

If you plan to use TFLite for Microcontrollers, you might want to use the Zephyr OS. Please follow the installation steps there: <https://docs.zephyrproject.org/latest/getting_started/index.html>

### Python: From TensorFlow to TFLite (and TFLM)

When creating an embedded model, we follow a typical pipeline:

1. Train a normal Keras model (suggestion: a Sequential model or functional API model are best, the sub-classing API is not well supported for quantization-aware training)
2. Once you have trained your TF model, use the quantization-aware retraining to create a quantize-ready and fine-tuned model
3. Convert quantize-aware model to TFLite
4. Convert TFLite model to TFLite Micro CPP-style file
5. (Optional) Visualize the quantized model
6. Export your pre-processing steps

##### 2. TF: Quantization-aware retraining

* Recommended: follow the tutorial: [https://www.tensorflow.org/model_optimization/guide/quantization/training_example](https://www.tensorflow.org/model_optimization/guide/quantization/training_example)

Import the tensorflow optimization package

```
import tensorflow as tf
import tensorflow_model_optimization as tfmot
```

(Optional) Copy the model (to be able to compare before and after quantization)

```
model_q = tf.keras.models.clone_model(model)
model_q.set_weights(model.get_weights())
```

Quantize the model and fine-tune it by retraining for a few iterations. Without this fine-tuning, quantization causes a 5-10% accuracy loss.

```
model_q = tfmot.quantization.keras.quantize_model(model_q)
# Compile model (as example, using SGD and sparse cross-entropy loss fn)
sgd = tf.optimizers.SGD(learning_rate=1e-3,momentum=0.9,decay=1e-6,nesterov=True,)
model_q.compile(loss='sparse_categorical_crossentropy',optimizer=sgd,metrics=tf.keras.metrics.SparseCategoricalAccuracy())
# retrain model to re-gain the accuracy lost due to quantization
model_q.fit(train_ds,epochs=3,validation_data=test_ds,validation_freq=1)
```

(Optional) Quantization-aware retraining does not support hand-crafted layers, you might need to force skip those layers. The following code can help you do that. You need to annotate layers while copying the weights from the original model to the cloned model.

```
class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  def get_weights_and_quantizers(self, layer):
    return []
  def get_activations_and_quantizers(self, layer):
    return []
  def set_quantize_weights(self, layer, quantize_weights):
    pass
  def set_quantize_activations(self, layer, quantize_activations):
    pass
  def get_output_quantizers(self, layer):
    return []
  def get_config(self):
    return {}

def annotate_layers(layer):
    if isinstance(layer, tf.keras.layers.Multiply):
        return tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=NoOpQuantizeConfig())
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        return tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=NoOpQuantizeConfig())
    # if isinstance(tf.keras.layers.core.Concatenate):
    #     return tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=NoOpQuantizeConfig())
    return layer

# Annotate layers when copying
model_q = tf.keras.models.clone_model(model,clone_function=annotate_layers,)
model_q.set_weights(model.get_weights())
# Specify scope if you use weird Layers (functional API)
with tfmot.quantization.keras.quantize_scope({'Multiply': tf.keras.layers.Multiply}):
    model_q = tfmot.quantization.keras.quantize_model(model_q)
```

##### 3. TF to TFLite

Recommended: follow the tutorial: <https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb>

A TFLite is not a normal Keras (or TF) model. It emulates the way TFLite runs on an embedded device, by creating input and output pointers, buffers for computation, etc.

We need to convert from TF to TFLite by using the TFLite converter. Note that to optimally quantize weights, we need to know the usual input distribution. We need to pass a subset (or all) X_train to the function.

**Warning:** Do not use X_test here! This would optimize the model with regard to your evaluation set and provide better results, but that would be the same as training on the evaluation set directly.

**Note:** You can set your input and output types to either quantized (tf.int8), or unquantized (tf.float32). For large input (e.g., images) I recommend to use int8 input. For classification taks, the input is between 0 and 255 in quantized uint8, and 0.0f and 1.0f for unquantized.

```
def convert_TF_to_TFLite(model_tf, X_train, TFLite_target_filename):
    # Create TFLite model from the original TF model
    converter = tf.lite.TFLiteConverter.from_keras_model(model_tf)
    # Set the optimization flag to use default quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Enforce integer only quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Also Quantize input and output (not mandatory, replace with tf.float32 to keep floating representations)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    # Provide a representative dataset to optimize quantization with respect to expected data distribution
    def generate_representative_dataset():
        for i in range(len(X_train)):
            yield([np.float32(X_train[i]).reshape(1, len(X_train[0]))])
    # Converter will use the above function to optimize quantization
    converter.representative_dataset = representative_dataset
    #convert to TFLite
    model_tflite = converter.convert()
    open(TFLite_target_filename, "wb").write(model_tflite)
    # If you want to read the on-disk size (good proxy for on device size)
    # size = os.path.getsize(TFLite_target_filename)
    return model_tflite
```

TFLite models do not provide the same API as Keras models: you cannot use .predict() or .evaluate(). The following function runs a TFLite model on an input, and returns the predicted output.

Note: X is a numpy matrix

```
def predict_TFLite(model, X):
    x_data = np.copy(X) # the function quantizes the input, so we must make a copy
    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    # Inputs will be quantized
    input_scale, input_zero_point = input_details["quantization"]
    if (input_scale, input_zero_point) != (0.0, 0):
        x_data = x_data / input_scale + input_zero_point
        x_data = x_data.astype(input_details["dtype"])
    # Invoke the interpreter
    predictions = np.empty(x_data.size, dtype=output_details["dtype"])
    for i in range(len(x_data)):
        interpreter.set_tensor(input_details["index"], [x_data[i]])
        interpreter.invoke()
        predictions[i] = interpreter.get_tensor(output_details["index"])[0]
    # Dequantize output
    output_scale, output_zero_point = output_details["quantization"]
    if (output_scale, output_zero_point) != (0.0, 0):
        predictions = predictions.astype(np.float32)
        predictions = (predictions - output_zero_point) * output_scale
    # todo reshape output into array for each exit
    return predictions
```

##### 4. TFLite to TFLite Micro

**Warning:** This section is a work in progress!

TensorFlow Lite for Microcontrollers (shorted as TFLite Micro, TFLM) is the embedded engine to run inference on resource-constrained devices. It is implemented in C++, and uses a (slightly) different model format. To run a model on an embedded platform, we need to convert a TFLite model to a C++ array. The following code takes a TFLite model ***saved on disk*** and create a .cc file using xxd (xxd needs to be installed on your machine!).

**Warning:** Please go through the generated file, and check that the model is defined as a \`alignas(16) const unsigned char model[]\`! We need to ensure the alignment is correct.

```
def convert_TFLite_to_TFLM(TFLite_filename, TFLM_target_filename):
    # Read a TFLite saved model, convert it to TFLite Micro
    # save TFLite model
    TFLite_target = f"{TFLM_target_filename}_tflite"
    open(TFLite_target, "wb").write(model_tflite)
    # Convert to a C source file, i.e, a TensorFlow Lite for Microcontrollers model
    !xxd -i {TFLite_target} > {TFLM_target_filename}
    # Update variable names
    REPLACE_TEXT = TFLite_target.replace('/', '_').replace('.', '_')
    !sed -i 's/'{REPLACE_TEXT}'/g_model/g' {TFLM_target_filename}
```

**Warning:** The following code has not been tested!

It is also possible to manually create the file. The following code writes the TFLite model into a Hpp header.

```
# Alternative approach?
# 
# !rm -f ./constants.cc
# Create C++ header for the model
# model must be a char array, with 16-bit alignment
model_str = "alignas(16) const unsigned char TFLM_model[] = "
# Read TFLite model, write it into CPP-style
with open(tflite_filename, 'r') as file:
    data = file.read();
    model_str += data[data.index("{"): len(data)].replace("unsigned", "const")
with open(target_filename, "w") as file:
    file.write(model_str) 
```

You should obtain a .cc that looks like this:

```
alignas(16) const unsigned char model[] = {
  0x1c, 0x00, ...., ....,
  ...., 0x00, 0x00, 0x00
};
const int model_len = 65104;
```

Where model_len is the model size in flash.

You should also contain a .h header file containing the following

```
extern const unsigned char model[];
extern const int model_len;
```

If not, create it yourself.

##### 5. Visualize the embedded model

Once a TFLite model has been generated, you do not have access to the .summary() function provided by the Keras API. To make sure that all layers are quantized, and what kind of input/output your embedded model expects, you can run the following python command.

```
python -m tensorflow.lite.tools.visualize model.tflite visualized_model.html
# alternatively
cd PATH_TO_TFLM/tensorflow/lite/tools/visualize && python visualize.py model.tflite
```

where *model.tflite* is the model saved by TFLite after conversion. The script creates a file named *visualized_model.html*.

**todo:** explain the main parts of the visualization document.

##### 6. Export your pre-processing steps

if you apply pre-processing on your data, you might want to export it as well. Typically, images are standardized (for example, x= x-mean)/std) to ensure that the input is centered around 0 with a standard deviation of 1). You will need to apply the same pre-processing on the embedded device, don't forget to export them! I, personally, export the mean and std computed over the entire training set, and apply them on the device before running the inference.

# Zephyr: Run an embedded model on a microcontroller

We assume here you have Zephyr and its toolchain installed. Follow the tutorial here to do it: <https://docs.zephyrproject.org/latest/getting_started/index.html>

Make sure that the path to your arm-gcc is set.

```
export GNUARMEMB_TOOLCHAIN_PATH=your/path/to/armgcc
```

**The first time,** you need to compile TFLM yourself. This ensures that all external libs are downloaded (especially the flatbuffers lib).

```
# For Linux
cmake PATH_TO_TFLM/tensorflow/lite/micro/tools/make
# For MacOS
brew install gmake
gmake PATH_TO_TFLM/tensorflow/lite/micro/tools/make
```

This might take time.

We now set our Zephyr application project to include TFLM. To do so, we need to modify the CMakeLists.txt and prj.conf files.

##### CMakeLists.txt

The file will load TFLM as an external library, build it for the specified platform and link it.

Note: you can modify the floating point representation (soft or hard ABI) in this file. Compiling for a new platform takes time, and does not print any output. This is normal.

You need to set cmake or gmake based on your OS (see set(submake gmake) line below)

You can include CMSIS-NN optimized kernels for ARM platforms. This does not make a lot of difference for the binary size.

```
cmake_minimum_required(VERSION 3.13.1)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(external_lib) # todo can we change name?
FILE(GLOB app_sources src/*.cc) # find all sources in src/ folder
target_sources(app PRIVATE ${app_sources}) # include them
zephyr_include_directories(src)
#zephyr_cc_option(-lstdc++)
# The external static library that we are linking with does not know
# how to build for this platform so we export all the flags used in
# this zephyr build to the external build system.
list(GET ZEPHYR_RUNNER_ARGS_jlink 1 MCPU_FLAG ) #Here we assume that '-mcpu' will be the second argument. This might be wrong. 
string(REPLACE "=" ";" MCPU_FLAG_LIST ${MCPU_FLAG})
list(GET MCPU_FLAG_LIST 0 MCPU)
# set target platform
set(TARGET cortex_m_generic) #alternatively set(TARGET ${BOARD})
set(TARGET_ARCH cortex-m4) # for the nRF52840!
# define the path to your TFLite Micro cloned repository
set(TF_SRC_DIR /Users/valentin/toolchains/tflite-micro)
# precise the path towards the makefile
set(TF_MAKE_DIR ${TF_SRC_DIR}/tensorflow/lite/micro/tools/make)
# west will call TFLM's makefile, that will generate the TFLM binary here
set(TF_LIB_DIR ${TF_MAKE_DIR}/gen/${TARGET}_${TARGET_ARCH}_default/lib)
# Precise floating point representation, you need to recompile (=delete the old version) of TFLM if you change this line
set(extra_project_flags "-mcpu=${TARGET_ARCH} -mthumb -mno-thumb-interwork -mfpu=fpv5-sp-d16 -mfloat-abi=softfp")
# include for the C language
zephyr_get_include_directories_for_lang_as_string(       C C_includes)
zephyr_get_system_include_directories_for_lang_as_string(C C_system_includes)
zephyr_get_compile_definitions_for_lang_as_string(       C C_definitions)
zephyr_get_compile_options_for_lang_as_string(           C C_options)
# you can add -DCMSIS_NN if you do not want the CMSIS-optimized kernels (size change is negligible)
set(external_project_cflags
  "${C_includes} ${C_definitions} ${optC_optionsions} ${C_system_includes} ${extra_project_flags}"
)
# include for C++
zephyr_get_include_directories_for_lang_as_string(       CXX CXX_includes)
zephyr_get_system_include_directories_for_lang_as_string(CXX CXX_system_includes)
zephyr_get_compile_definitions_for_lang_as_string(       CXX CXX_definitions)
zephyr_get_compile_options_for_lang_as_string(           CXX CXX_options)
# you can add -DCMSIS_NN if you do not want the CMSIS-optimized kernels (size change is negligible)
set(external_project_cxxflags
  "${CXX_includes} ${CXX_definitions} ${CXX_options} ${CXX_system_includes} ${extra_project_flags}"
)
include(ExternalProject) # todo can we change that name?
# For MacOS users
set(submake gmake)
# For Linux users
# set(submake cmake)

ExternalProject_Add(
  tf_project # Name for custom target
  #PREFIX     ${mylib_build_dir} # Root dir for entire project
  SOURCE_DIR ${TF_SRC_DIR}
  BINARY_DIR ${TF_SRC_DIR} # This particular build system is invoked from the root
  CONFIGURE_COMMAND ""    # Skip configuring the project, e.g. with autoconf
  BUILD_COMMAND
  ${submake} -f tensorflow/lite/micro/tools/make/Makefile
  TARGET=${TARGET}
  TARGET_ARCH=${TARGET_ARCH}
  # OPTIMIZED_KERNEL_DIR=cmsis_nn # you can add if you want the optimized kernels
  TARGET_TOOLCHAIN_ROOT=${GNUARMEMB_TOOLCHAIN_PATH}/bin/
  TARGET_TOOLCHAIN_PREFIX=arm-none-eabi-
  #PREFIX=${mylib_build_dir}
  CC=${CMAKE_C_COMPILER}
  CXX=${CMAKE_CXX_COMPILER}
  AR=${CMAKE_AR}
  CCFLAGS=${external_project_cflags} 
  CXXFLAGS=${external_project_cxxflags} 
  microlite 
  INSTALL_COMMAND ""      # This particular build system has no install command
  BUILD_BYPRODUCTS ${TF_LIB_DIR}/libtensorflow-microlite.a
  )
# Create a wrapper CMake library that our app can link with
add_library(tf_lib STATIC IMPORTED GLOBAL)
add_dependencies(
  tf_lib
  tf_project
  )
set_target_properties(tf_lib PROPERTIES IMPORTED_LOCATION             ${TF_LIB_DIR}/libtensorflow-microlite.a)
set_target_properties(tf_lib PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TF_SRC_DIR};${TF_SRC_DIR}/tensorflow/lite/micro;${TF_MAKE_DIR}/downloads/flatbuffers/include") 
target_link_libraries(app PUBLIC tf_lib)
```

##### prj.conf

We must ensure that the C++ compiler is called. We need the FPU to speed up computation (but this might consume more energy!). We also ensure that we have a heap large enough to malloc intermediate buffers during computation.

```
CONFIG_CPLUSPLUS=y
CONFIG_STD_CPP11=y
CONFIG_NEWLIB_LIBC=y
CONFIG_FPU=y
CONFIG_FP_SOFTABI=y
CONFIG_LIB_CPLUSPLUS=y
CONFIG_HEAP_MEM_POOL_SIZE=16384
CONFIG_MAIN_STACK_SIZE=8192
CONFIG_SYSTEM_WORKQUEUE_STACK_SIZE=8192

CONFIG_LOG=n
CONFIG_GPIO=y
CONFIG_UART_CONSOLE=n
#enabale RTT logging
CONFIG_USE_SEGGER_RTT=y
CONFIG_RTT_CONSOLE=y
CONFIG_LOG_PRINTK=y
```

##### Main .cpp file

Mandatory includes

```
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
// You'll often see lite/version included, but it does not exist anymore in recent TFLM libs
//#include "tensorflow/lite/version.h"

// Your own model file
#include "model.h"

// if you want to run unit tests
// #include "tensorflow/lite/micro/testing/micro_test.h"

// if you want to print
#include <sys/printk.h>
#include <math.h>

#include <zephyr.h>
```

Set up the TFLM components, prepare the buffers for the model execution (defined outside of the main function)

```
namespace
{
tflite::ErrorReporter *error_reporter = nullptr;
const tflite::Model *model = nullptr;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// Create an area of memory to use for input, output, and intermediate arrays.
// const int kModelArenaSize = 6064;
// Extra headroom for model + alignment + future interpreter changes.
// const int kExtraArenaSize = 560 + 16 + 160;
const int tensor_arena_size = 2048; //kModelArenaSize + kExtraArenaSize;
static uint8_t tensor_arena[tensor_arena_size];
} 
```

Setup phase.

```
// Create error reporter
static tflite::MicroErrorReporter micro_error_reporter;
error_reporter = &micro_error_reporter;

// load your model
model = tflite::GetModel(blueseer_model);

// Check that TFLM library si similar to the one used to generate model
if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
			"Model provided is schema version %d not equal "
			"to supported version %d.",
			 model->version(), TFLITE_SCHEMA_VERSION);
    printk("model not supported\n");
    return;
}

// Build an interpreter to run the model with.
static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena,
                                     tensor_arena_size, error_reporter);
interpreter = &static_interpreter;

// Allocate memory from the tensor_arena for the model's tensors.
TfLiteStatus allocate_status = interpreter->AllocateTensors();
if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    printk("tensor allocation failed\n");
    return;
}

// Obtain pointers to the model's input and output tensors.
input = interpreter->input(0);
output = interpreter->output(0);

//print expected and actual input size and used memory 
//TF_LITE_MICRO_EXPECT_NE(nullptr, input);
// The property "dims" tells us the tensor's shape. It has one element for
// each dimension. Our input is a 2D tensor containing 1 element, so "dims"
// should have size 2.
//TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
// The value of each element gives the length of the corresponding tensor.
// We should expect two single element tensors (one is contained within the
// other).
//TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
//TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
// The input is a 32 bit floating point value
//TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);
int expected = input->dims->data[1];
printk("Expected input: %d, used tensor bytes: %d\n", expected, interpreter->arena_used_bytes());
```

Run inference

```
// Preprocess data if you need to! You need to create your own function
// preprocess_input(input_data, INPUT_SIZE, mean_value, std_value);
// Copy input to input pointer
for (int i = 0;i<INPUT_SIZE; ++i){
    input->data.f[i] = input_data[i];
}
//execute model
interpreter->Invoke();
// copy output to our own variable
for (int i = 0;i<NUM_CLASSES; ++i){
    output[i] = output->data.f[i];
}
// find predicted class
float max_value=0.0f;
float pred;
int selected_class=0;
for (int i = 0; i<NUM_CLASSES; i++){
    pred = output[i];
    if(pred>max_value){
        max_value=pred;
        selected_class=i;
    }
}
```

ddd.

##### Reducing the binary size by only including relevant TFLM functions

replace the resolver with:

```
// Replace the Resolver with micro resolver
// remove #include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// remove tflite::AllOpsResolver resolver;
static tflite::MicroMutableOpResolver<N> resolver;
resolver.AddFullyConnected();
resolver.AddSoftmax();
resolver.AddRelu();
resolver.AddQuantize();
resolver.AddDequantize();
```

Where you can find the functions you need to include by checking the visualization tool presented in Python, step 5. N is the number of operations added.

##### Advanced: Understanding the TFLite array

TFLite uses Flatbuffers to store the model. This makes it compelx to udnerstand what is really stored in there. Luckily, it is possible to transform a TFLite model (raw binary) into a readable JSON file.

Install the flatbuffers executables (flatc). On MacOS:

```
brew install flatbuffers
```

You then need to download the FlatBuffer format used by TFLite.

```
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs
```

Now, you can transpose the binary TFLite model into a JSON file

```
flatc --raw-binary -t schema.fbs -- model.tflite 
```

This creates a model.json file.

## Useful Links

<https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb>

<https://www.tensorflow.org/lite/guide/faq>

<https://www.tensorflow.org/modeloptimization/guide/quantization/training_example>

Understanding memory management in TFLM:

<https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/memory_management.md>

<https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/online_memory_allocation_overview.md>

<https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/offline_memory_plan.md>

<https://blog.tensorflow.org/2020/10/optimizing-tensorflow-lite-runtime.html>
