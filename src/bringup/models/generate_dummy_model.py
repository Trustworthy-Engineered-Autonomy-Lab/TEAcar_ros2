#!/usr/bin/env python3
# DUMMY: This entire script is for testing only — remove before publication.
#
# generate_dummy_model.py
# Generates a trivial ONNX model that takes an image tensor and outputs
# a constant steer value of 0.0. Used to test the TensorRT inferencer
# pipeline without a real trained model.
#
# Usage:
#   pip3 install onnx
#   python3 generate_dummy_model.py
#
# Output: dummy_model.onnx in the same directory

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def main():
    # Input: "image" — float32 [1, 3, 224, 224]
    image_input = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 224, 224])

    # Output: "steer" — float32 [1]
    steer_output = helper.make_tensor_value_info("steer", TensorProto.FLOAT, [1])

    # We'll make the ONNX model steer based on the overall brightness of the image.
    # Bright image -> steer right (+1.0)
    # Dark image -> steers left (-1.0)
    # This makes it visibly distinct from the C++ dummy sine wave steering.
    
    # 1. ReduceMean over all spatial and channel axes [1, 2, 3]
    reduce_mean_node = helper.make_node(
        "ReduceMean",
        inputs=["image"],
        outputs=["mean_val"],
        axes=[1, 2, 3],
        keepdims=1,
    )

    # 2. Divide by 127.5 (assuming input pixels are in [0, 255.0])
    div_const = numpy_helper.from_array(np.array([127.5], dtype=np.float32), name="div_const")
    div_const_node = helper.make_node("Constant", inputs=[], outputs=["div_const_val"], value=div_const)
    div_node = helper.make_node("Div", inputs=["mean_val", "div_const_val"], outputs=["scaled_val"])

    # 3. Subtract 1.0 to shift to [-1.0, 1.0]
    sub_const = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="sub_const")
    sub_const_node = helper.make_node("Constant", inputs=[], outputs=["sub_const_val"], value=sub_const)
    
    # ONNX Sub node
    sub_node = helper.make_node("Sub", inputs=["scaled_val", "sub_const_val"], outputs=["steer"])

    graph = helper.make_graph(
        [reduce_mean_node, div_const_node, div_node, sub_const_node, sub_node],
        "dummy_brightness_steer_model",
        [image_input],
        [steer_output],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    output_path = __import__("os").path.join(
        __import__("os").path.dirname(__import__("os").path.abspath(__file__)),
        "dummy_model.onnx",
    )
    onnx.save(model, output_path)
    print(f"DUMMY: Saved dummy ONNX model to {output_path}")

if __name__ == "__main__":
    main()
