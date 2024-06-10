import os
import os.path
import torch
from model_wrapper import Model, ModelFamily
from timehelp import with_progress, display_header
import matplotlib.pyplot as plt
from render_output import OutputRenderer
import json
import numpy as np

class BatteryConfigs:
    Code2Code = dict(
        case_count=100,
        meta_count=None,
        task="code2code-trans",
        display_name="CodeTrans",
        prompts=[
            "// original code.java\n{prompt}\n// code.cs version of code.java\n",
            "// code.java\n{prompt}\n// code.cs\n",
            "// This code is written in Java. Reproduce the same exact code in C#.\n{prompt}\n",
            "// original code.java\n{prompt}\n\n// code.cs version of code.java\n",
        ],
        battery_path="./data/CodeXGLUE/Code-Code/code-to-code-trans/data",
        questions_file="test.java-cs.txt.java",
        truth_file="test.java-cs.txt.cs",
    )

    Bugs2Fix = dict(
        case_count=100,
        meta_count=None,
        task="bugs2fix",
        display_name="Bugs2fix",
        prompts=[
            "// the buggy version of the code\n{prompt}\n// the fixed version of the code\n",
            "// You are given a piece of buggy code. Your task is to fix the error, and generate the corrected code. Fix the following code:\n{prompt}\n",
        ],
        battery_path="./data/CodeXGLUE/Code-Code/code-refinement/data/small",
        questions_file="test.buggy-fixed.buggy",
        truth_file="test.buggy-fixed.fixed",
    )


class BatteryRunner:
    def __init__(self, case_count, task, prompts, battery_path, questions_file=None, truth_file=None, *, meta_count=None, json_battery=False, **kwargs):
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


    @staticmethod
    def of(kwargs):
        return BatteryRunner(**kwargs)

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

    
    def run_battery(self, family, prompt_indices=None, prompt_index=None, quiet=False, *args, **kwargs):
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

            if not quiet:
                display_header(f"Testing prompt index {test_prompt_index}")
                print("Prompt to be tested:")
                print(prompt)
            
            self.single_battery(
                family=family,
                prompt=prompt,
                output_dir=output_dir,
                quiet=quiet,
                *args,
                **kwargs,
            )


    def single_battery(self, family, prompt, output_dir, family_name=None, quiet=False, *args, **kwargs):
        # e.g. family=ModelFamily.CodeGen1.multi
        if family_name is None:
            family_name = ModelFamily.name_for(family)
        
        for key, model_name in family.items():
            if not quiet:
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
                if not quiet:
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
                if output is None:
                    output = ""
                decoded = output.strip()
                #if output is None:
                #    print("Warning: Model returned no output (prompt may have been too large)")
                #    decoded = ""
                #else:
                #    decoded = model.decode(output).strip()

                if "\n" in decoded:
                    if not quiet:
                        print("!! WARNING !! newline found in output")
                        print("Input: ", test_case)
                        print("Prompt: ", repr(specific_prompt))
                        print("Decoded: (next line)")
                        print(repr(decoded))
                    decoded = decoded.split("\n")[0]
                
                output_file.write(decoded + "\n")
        
                del model.inputs, output
        
            for output_path, to_skip in iterate_structure:
                if not quiet:
                    print(f"Opening {output_path}...")
                with open(output_path, "a+") as output_file: 
                    if to_skip > 0 and not quiet:
                        print(f"{to_skip} entries found already, skipping that many...")
                    iterate(output_file, skip=to_skip)
        
            model.free()

    
    def init_cases(self, family, family_name=None):
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

    
    def init_render(self, *args, **kwargs):
        return self.init_cases(*args, **kwargs)

    
    def calculate_metrics(self, metric, limit=None, cache=True):
        if cache:
            cache_file_path = os.path.join("./output", self.task, "metrics.json")
            with open(cache_file_path, "a+", encoding="utf-8") as cache_file:
                cache_file.seek(0)
                data = cache_file.read()
                if len(data) == 0:
                    cache_obj = {
                        "case_count": self.case_count,
                        "results": {}
                    }
                else:
                    cache_obj = json.loads(data)
            
            if cache_obj["case_count"] == self.case_count and metric.shortname in cache_obj["results"]:
                return cache_obj["results"][metric.shortname]
        else:
            cache_obj = None
        
        by_prompt = {}
        for idx, family_answers in enumerate(self.prompt_family_answers):
            series = []
            series_name = f"prompt{idx}"
            for key, answers in family_answers.items():
                if limit is None:
                    limit = len(answers)
                
                grade = metric.grade(self.answer_key[:limit], answers[:limit])
                series.append(grade)
            by_prompt[series_name] = series

        if cache:
            cache_obj["case_count"] = self.case_count
            cache_obj["results"][metric.shortname] = by_prompt
            with open(cache_file_path, "w", encoding="utf-8") as cache_file:
                cache_file.write(json.dumps(cache_obj))
        
        return by_prompt
        
    
    def render_metric(self, metric, by_prompt=None, *args, **kwargs):
        if by_prompt is None:
            by_prompt = self.calculate_metrics(metric)

        self.renderer = OutputRenderer(
            baseline=metric.baseline,
            metric=metric.name
        )

        self.renderer.render(ys=by_prompt, *args, **kwargs)
        return by_prompt

    
    def calculate_iterative_metric(self, metric, limit=None, quiet=False):
        if limit is None:
            max_case = self.case_count
        else:
            max_case = limit
        datapoints = {}
        for i in range(1, max_case + 1):
            if not quiet:
                print(f"{i = }/{max_case}...", end="\r")
            by_prompt = self.calculate_metrics(metric, limit=i)
            for key, data in by_prompt.items():
                if key not in datapoints:
                    datapoints[key] = []
                datapoints[key].append(data)
        for key in datapoints.keys():
            datapoints[key] = np.transpose(datapoints[key])

        return datapoints
    
    
    def render_iterative_metric(self, metric, limit=None, datapoints=None, quiet=False, save=None, cmap="viridis"):
        if datapoints is None:
            datapoints = self.calculate_iterative_metrics(metric, quiet=quiet, limit=limit)
        
        prompts = datapoints.keys()
        prompt_count = len(prompts)
        assert prompt_count > 1, "Unsure if prompt_count == 1 works; TODO: test"
        
        names = ["350M", "2B", "6B", "16B"]
        
        width = int(np.ceil(np.sqrt(prompt_count)))
        height = int(np.ceil(prompt_count / width))
        figs, axes = plt.subplots(height, width, figsize=(10 * width, 6 * height))
        linear_axes = []
        
        for i in range(prompt_count):
            if len(axes.shape) == 1:
                ax = axes[i]
            else:
                ax = axes[i // width, i % width]
            linear_axes.append(ax)

        cmap_object = plt.get_cmap(cmap)
        colors = cmap_object(np.linspace(1, 0, len(names)))
        
        for idx, prompt in enumerate(prompts):
            ax = linear_axes[idx]
        
            data = datapoints[prompt]
            for i in range(data.shape[0]):
                ax.plot(
                    range(1, data.shape[1] + 1),
                    data[i],
                    label=names[i],
                    color=colors[i],
                    linewidth=3,
                    alpha=0.8
                )
            
            ax.set_xlabel("Test Case #")
            ax.set_ylabel(metric.name)
            ax.set_title(f"{metric.name} over time for {prompt}")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        if save:
            plt.save_fig(save, bbox_inches="tight")
        plt.show()


    def calculate_bootstrap_metric(self, metric, sample_size=50, iterations=500, quiet=False, use_cache=True, seed=None):
        cache_config = {
            "seed": seed,
            "sample_size": sample_size,
            "iterations": iterations,
            "results": {}
        }
        bootstrap_cache = f"./output/{self.task}/bootstrap-{metric.shortname}.json"
        cache_file = None
        if os.path.exists(bootstrap_cache):
            cache_file = open(bootstrap_cache, "r+")
        else:
            cache_file = open(bootstrap_cache, "w+")
        
        cache = json.loads(cache_file.read() or "{}")
        cache_file.close()
        
        if use_cache and cache:
            # attempt to read from the cache
            matches_props = all(
                cache_config[prop] == cache[prop]
                for prop in ["seed", "sample_size", "iterations"]
            )
            if matches_props:
                return cache["results"]
        
        max_size = self.case_count
        by_prompt = {}

        if seed is not None:
            np.random.seed(seed)
        
        for i in range(iterations):
            if not quiet:
                print(f"i = {i + 1}/{iterations}...", end="\r")
            subset_indices = np.random.choice(np.arange(max_size), sample_size, replace=False)
            
            for idx, family_answers in enumerate(self.prompt_family_answers):
                series = []
                series_name = f"prompt{idx}"
                for key, answers in family_answers.items():
                    answer_key_subset = list(np.array(self.answer_key)[subset_indices])
                    answers_subset = list(np.array(answers)[subset_indices])
                    grade = metric.grade(answer_key_subset, answers_subset)
                    series.append(grade)
                if series_name not in by_prompt:
                    by_prompt[series_name] = []
                by_prompt[series_name].append(series)
        
        for key in by_prompt.keys():
            by_prompt[key] = np.transpose(by_prompt[key]).tolist()

        cache_config["results"] = by_prompt
        if use_cache:
            with open(bootstrap_cache, "w+") as cache_file:
                cache_file.write(json.dumps(cache_config))
        
        return by_prompt