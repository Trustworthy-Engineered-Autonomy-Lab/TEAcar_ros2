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

    # Constant tensor: always 0.0
    constant_value = numpy_helper.from_array(
        np.array([0.0], dtype=np.float32), name="const_steer"
    )

    # A single Constant node that ignores the input and outputs 0.0
    # We also add an Identity node so the input is "used" (some runtimes
    # optimise away unused inputs).
    identity_node = helper.make_node("Identity", inputs=["image"], outputs=["image_identity"])
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["steer"],
        value=constant_value,
    )

    graph = helper.make_graph(
        [identity_node, constant_node],
        "dummy_steer_model",
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
