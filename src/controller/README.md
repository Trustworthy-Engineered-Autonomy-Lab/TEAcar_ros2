# TEAcar_ros2: Inferencer and Neural Network Controller

This document explains how the neural network steering controller (`nn_controller`) interacts with the neural network backend plugins (`inferencer`), and answers common questions about configuration.

## How do I specify which model to use?

The model is specified via the `model_file` parameter sent to the `nn_controller` node. 
In the `bringup` package's launch files (like `autopilot.launch.py`), you can see the node configuration:

```python
Node(
    package="controller",
    executable="nn_controller",
    name="nn_controller_node",
    parameters=[
        {"model_file": os.path.join(bringup_share, "models", "my_custom_model.onnx")},
        # ... other params ...
    ],
)
```

Change the `"model_file"` string to the absolute path of your ONNX or TensorRT `.engine` file.

## How do I specify which inferencer plugin?

The `nn_controller` uses dynamic loading (`dlopen`) to load a backend plugin library at runtime in the format `lib<backend>_inferencer.so`. 

You specify the backend using the `"backend"` parameter on the node:
```python
{"backend": "tensorrt"}  # Loads libtensorrt_inferencer.so
# or
{"backend": "dummy"}     # Loads libdummy_inferencer.so
```

When you write a new inferencer backend, you just need to implement the C API defined in `inferencer_c.h` and compile it as a shared library named `libYOURNAME_inferencer.so`. Then pass `{"backend": "YOURNAME"}`.

## Are the dummy components separate?

Yes! The dummy components provided for testing are entirely independent of the real TensorRT pipeline:

1. **`libdummy_inferencer.so` (`dummy_inferencer.cpp`)**: A completely fake backend plugin. If you set `{"backend": "dummy"}`, it will load this `.so` file, ignore whatever model file you pass, and output a hardcoded sine wave steering pattern. It requires no GPU or ONNX files.
2. **`dummy_model.onnx` (`generate_dummy_model.py`)**: A totally valid ONNX model that calculates the average brightness of the image to steer. If you set `{"backend": "tensorrt"}` and pass this model, the **real TRT backend** (`libtensorrt_inferencer.so`) will parse it, optimize it, write a `.engine` file, and run it on the GPU, thoroughly testing the entire TensorRT runtime pipeline. If you cover the camera (dark), it steers left (-1.0). If you point it at a light source (bright), it steers right (+1.0).

## How do I control the throttle while running autopilot?

The **`nn_controller` only publishes steering commands.** It explicitly publishes a throttle value of 0.0. 

This architecture allows a human operator to control the throttle manually using a joystick while the neural network handles the steering. 

In `autopilot.launch.py`, the launch file includes `drive.launch.py` entirely, which runs the `joy_node` and the `param_controller` (joystick controller).
Both the `nn_controller` and the `param_controller` will publish to the `/motion_cmd` topic simultaneously:
- The `nn_controller` publishes `{steer: <net_output>, throttle: 0.0}`
- The `param_controller` publishes `{steer: <joy_steer>, throttle: <joy_throttle>}`

The actuator node seamlessly takes the throttle from the joystick (since you press the trigger) and the steer from the NN. If you wish to override the NN steering, both nodes are publishing conflicting steer values, but typically the joystick commands override when pressed. For a true override system, a dedicated multiplexer node (MUX) should be used.
