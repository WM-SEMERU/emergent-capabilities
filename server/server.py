import os
import sys
from flask import Flask, request, send_from_directory, jsonify
import random
# add parent directory to visibile import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from model_wrapper import ModelFamily
from run_battery import *

app = Flask(__name__)

@app.route("/")
def serve_html():
    return send_from_directory(".", "model-ui.html")
@app.route("/model-ui.js")
def serve_js():
    return send_from_directory(".", "model-ui.js")


@app.route("/load_cases", methods=["POST"])
def load_cases():
    data = request.get_json()
    if not data:
        return jsonify({
            "error": "Invalid input"
        }), 400
    
    #family = data["family"]
    task = data["task"]
    
    try:
        config = getattr(BatteryConfigs, task)
    except:
        return jsonify({
            "error": "Could not find task " + task
        }), 400
    
    runner = BatteryRunner.of(config)
    runner.load_cases()
    
    response = {
        "prompts": runner.prompts,
        "test_cases": runner.battery,
    }

    return jsonify(response)

# cache the model for consecutive runs
model = None
@app.route("/run_single", methods=["POST"])
def run_single():
    global model
    data = request.get_json()
    if not data:
        return jsonify({
            "error": "Invalid input"
        }), 400
    
    family = data["family"]
    model_input = data["input"]

    model_name = ModelFamily.CodeGen1.multi[family]
    if model and model.name != model_name:
        model.free()
        model = None

    if model is None:
        model = Model(model_name)
        model.configure(time=True)

    output = model.generate_until(model_input, stops=["\n"])
    del model.inputs

    return jsonify({
        "output": output,
        "time_taken": "idk yet"
    })

if __name__ == "__main__":
    app.run(port=3337)