import os
import torch
from model_wrapper import Model, ModelFamily
from timehelp import with_progress, display_header
import matplotlib.pyplot as plt
from render_output import OutputRenderer
import json

class BatteryRunner:
    def __init__(self, case_count, task, prompts, battery_path, questions_file=None, truth_file=None, *, meta_count=None, json_battery=False):
        self.task = task
        self.output_dir_base = f"./output/{task}"
        self.prompts = prompts
        self.case_count = case_count
        self.task = task
        self.prompts = prompts
        self.battery_path = battery_path
        self.json_battery = json_battery
        if json_battery:
            # {name:, format:, cases: [ {prompt:, truth: }] }
            #self.questions_path = None
            #self.truth_path = None
            pass
        else:
            self.questions_path = os.path.join(self.battery_path, questions_file)
            self.truth_path = os.path.join(self.battery_path, truth_file)
        self.battery = []
        self.meta_count = meta_count


    def load_cases(self):
        if self.json_battery:
            with open(self.battery_path, "r") as battery:
                test_cases = json.loads(battery.read())["cases"][:self.case_count]
                self.battery = [ obj["prompt"].strip() for obj in test_cases ]
        else:
            with open(self.questions_path, "r") as battery:
                self.battery = [
                    line.strip()
                    for line
                    in battery.readlines()[:self.case_count]
                ]
        
        print(f"Loaded {len(self.battery)} cases!")

    
    def run_battery(self, family, prompt_indices=None, prompt_index=None, *args, **kwargs):
        assert len(self.battery) > 0, "Must have at least 1 test case loaded"
        if prompt_indices is None:
            if prompt_index is None:
                prompt_indices = list(range(len(self.prompts)))
            else:
                prompt_indices = [ prompt_index ]
        
        
        assert len(prompt_indices) > 0, "Must run on at least one prompt/prompt index"
        
        for test_prompt_index in prompt_indices:
            assert test_prompt_index is not None, "Prompt index must not be None"
            
            prompt_dir = f"prompt{test_prompt_index}"
            output_dir = os.path.join(self.output_dir_base, prompt_dir)
            os.makedirs(output_dir, exist_ok=True)
            prompt = self.prompts[test_prompt_index]

            display_header(f"Testing prompt index {test_prompt_index}")
            print("Prompt to be tested:")
            print(prompt)
            
            self.single_battery(
                family=family,
                prompt=prompt,
                output_dir=output_dir,
                *args,
                **kwargs,
            )


    def single_battery(self, family, prompt, output_dir, family_name=None, *args, **kwargs):
        # e.g. family=ModelFamily.CodeGen1.multi
        if family_name is None:
            family_name = ModelFamily.name_for(family)
        
        for key, model_name in family.items():
            display_header(f"Loading {key} ({model_name})", depth=2)
            # optimization: don't load the model if we don't need to make more cases
            # reading files twice is way more time-efficient than loading a model we don't need to
            iterate_structure = []
            for i in range(self.meta_count or 1):
                if self.meta_count is None:
                    base_name = f"{family_name}-{key}.output"
                else:
                    base_name = f"{family_name}-{key}-mc{i}.output"
                output_path = os.path.join(output_dir, base_name)
                # creates the file if it doesn't exist
                with open(output_path, "a+") as output_file:
                    output_file.seek(0)
                    to_skip = len(output_file.readlines())

                # only record file if its missing outputs
                if to_skip < len(self.battery):
                    iterate_structure.append([ output_path, to_skip ])

            if len(iterate_structure) == 0:
                print("No new cases necessary to generate, not loading model")
                continue
            
            torch.cuda.empty_cache()
            model = Model(model_name)
            model.configure(time=True)
            model.verbose = False

            @with_progress(len(self.battery))
            def iterate(output_file, *, step=None):
                test_case = self.battery[step]
                specific_prompt = prompt.format(prompt=test_case)
                output = model.generate_until(specific_prompt, stops=["\n"], **kwargs)

                # output is now returned as a string
                decoded = output.strip()
                #if output is None:
                #    print("Warning: Model returned no output (prompt may have been too large)")
                #    decoded = ""
                #else:
                #    decoded = model.decode(output).strip()

                if "\n" in decoded:
                    print("!! WARNING !! newline found in output")
                    print("Input: ", test_case)
                    print("Prompt: ", repr(specific_prompt))
                    print("Decoded: (next line)")
                    print(repr(decoded))
                    decoded = decoded.split("\n")[0]
                
                output_file.write(decoded + "\n")
        
                del model.inputs, output
        
            for output_path, to_skip in iterate_structure:
                print(f"Opening {output_path}...")
                with open(output_path, "a+") as output_file: 
                    if to_skip > 0:
                        print(f"{to_skip} entries found already, skipping that many...")
                    iterate(output_file, skip=to_skip)
        
            model.free()

    
    def init_render(self, family, family_name=None):
        if family_name is None:
            family_name = ModelFamily.name_for(family)

        if self.json_battery:
            with open(self.battery_path, "r") as battery:
                test_cases = json.loads(battery.read())["cases"][:self.case_count]
                self.answer_key = [ obj["truth"].strip() for obj in test_cases ]
        else:
            with open(self.truth_path, "r") as truth_file:
                self.answer_key = truth_file.readlines()
        
        prompt_family_answers = []
        for prompt_index in range(len(self.prompts)):
            output_dir = os.path.join(self.output_dir_base, f"prompt{prompt_index}")
            family_answers = {}
            for key, model_name in family.items():
                assert self.meta_count is None, "Cannot render meta_count yet"
                # meta_count: base_name = f"{family_name}-{key}-mc{i}.output"
                base_name = f"{family_name}-{key}.output"
                output_path = os.path.join(output_dir, base_name)
                with open(output_path, "r") as output_file:
                    answers = output_file.readlines()
                family_answers[key] = answers
            prompt_family_answers.append(family_answers)

        self.prompt_family_answers = prompt_family_answers
        
    
    def render_metric(self, metric, save=None):
        by_prompt = {}
        for idx, family_answers in enumerate(self.prompt_family_answers):
            series = []
            series_name = f"prompt{idx}"
            for key, answers in family_answers.items():
                grade = metric.grade(self.answer_key[:len(answers)], answers)
                series.append(grade)
            by_prompt[series_name] = series

        self.renderer = OutputRenderer(
            baseline=metric.baseline,
            metric=metric.name
        )

        self.renderer.render(ys=by_prompt, save=save)
        return by_prompt
