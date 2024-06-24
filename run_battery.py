import os
import os.path
import torch
from model_wrapper import Model, ModelFamily
from timehelp import with_progress, display_header
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
from render_output import OutputRenderer, index_axis
import json
import numpy as np
import scipy

PATCH_SEPARATOR = "=" * 30 + "\n"
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

def open_relative(name, *args, **kwargs):
    return open(os.path.join(BASE_DIRECTORY, name), *args, **kwargs)

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

    """
    Code2CodeChecklist = dict(
        case_count=100,
        meta_count=None,
        task="code2code-trans_checklist",
        display_name="CodeTrans (Checklist)",
        prompts=[
            "// original code.java\n{prompt}\n// code.cs version of code.java\n",
            "// code.java\n{prompt}\n// code.cs\n",
            "// This code is written in Java. Reproduce the same exact code in C#.\n{prompt}\n",
            "// original code.java\n{prompt}\n\n// code.cs version of code.java\n",
        ],
        battery_path="./data/checklist/Code2Code",
        questions_file="test.java-cs.txt.java",
        truth_file="test.java-cs.txt.cs",
    )
    """

    Bugs2Fix = dict(
        case_count=100,
        meta_count=None,
        task="bugs2fix",
        display_name="Bugs2fix",
        prompts=[
            "// the buggy version of the code\n{prompt}\n// the fixed version of the code\n",
            "// You are given a piece of buggy code. Your task is to fix the error, and generate the corrected code. Fix the following code:\n{prompt}\n",
            "// You are given a piece of buggy code. Your task is to fix the error, and generate the corrected code. Fix the following code:\n{prompt}\n// The following code is correct:\n",
        ],
        battery_path="./data/CodeXGLUE/Code-Code/code-refinement/data/small",
        questions_file="test.buggy-fixed.buggy",
        truth_file="test.buggy-fixed.fixed",
    )

    Bugs2FixChecklist = dict(
        case_count=100,
        meta_count=None,
        task="bugs2fix_checklist",
        display_name="Bugs2fix (Checklist)",
        prompts=[
            "// the buggy version of the code\n{prompt}\n// the fixed version of the code\n",
            "// You are given a piece of buggy code. Your task is to fix the error, and generate the corrected code. Fix the following code:\n{prompt}\n",
            "// You are given a piece of buggy code. Your task is to fix the error, and generate the corrected code. Fix the following code:\n{prompt}\n// The following code is correct:\n",
        ],
        battery_path="./data/checklist/Bugs2fix",
        questions_file="test.buggy-fixed.buggy",
        truth_file="test.buggy-fixed.fixed",
        base="Bugs2Fix",
    )

    CommitMessageGeneration = dict(
        case_count=100,
        meta_count=None,
        task="commit",
        prompts=[
            "/* diff of changes\n{prompt}\n*/\n// a summary of the above diff is:\n// -"
        ],
        battery_path="./data/commits/commit_message_generation_codisum.json",
        json_battery=True,
    )


def sample_cmap(cmap, count=4, lower=0, upper=1):
    cmap_object = plt.get_cmap(cmap)
    colors = cmap_object(np.linspace(upper, lower, count))
    return colors


def init_lazy_model(model_name):
    model = None
    def inner_model(load=True):
        nonlocal model
        if load and model is None:
            model = Model(model_name)
            model.configure(time=True)
            model.verbose = False
        return model
    return inner_model


def clean_model_output(line):
    return line.replace("<|endoftext|>", "").strip()


class BatteryRunner:
    def __init__(self, case_count, task, prompts, battery_path, questions_file=None, truth_file=None, *, meta_count=None, json_battery=False, base=None, **kwargs):
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
        if base is None:
            self.base = None
        else:
            self.base = BatteryRunner.of(getattr(BatteryConfigs, base))


    @staticmethod
    def of(kwargs):
        return BatteryRunner(**kwargs)

    def load_cases(self):
        if self.json_battery:
            with open_relative(self.battery_path, "r") as battery:
                test_cases = json.loads(battery.read())["cases"][:self.case_count]
                self.battery = [ obj["prompt"].strip() for obj in test_cases ]
        else:
            with open_relative(self.questions_path, "r") as battery:
                self.battery = [
                    line.strip()
                    for line
                    in battery.readlines()[:self.case_count]
                ]
        
        print(f"Loaded {len(self.battery)} cases!")
        if self.base is not None:
            print("Loading base config (does not run base config battery)...")
            self.base.load_cases()

    
    def run_battery(self, family, prompt_indices=None, prompt_index=None, quiet=False, patch=False, *args, **kwargs):
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
                patch=patch,
                *args,
                **kwargs,
            )
            #print("Breaking out of prompt iteration index early...")
            #break


    def single_battery(self, family, prompt, output_dir, family_name=None, quiet=False, patch=False, *args, **kwargs):
        # e.g. family=ModelFamily.CodeGen1.multi
        if family_name is None:
            family_name = ModelFamily.name_for(family)
        
        for key, model_name in family.items():
            if not quiet:
                display_header(f"Loading {key} ({model_name})", depth=2)
            # optimization: don't load the model if we don't need to make more cases
            # reading files twice is way more time-efficient than loading a model we don't need to
            iterate_structure = []
            existing_lines = []
            for i in range(self.meta_count or 1):
                if self.meta_count is None:
                    base_name = f"{family_name}-{key}.output"
                else:
                    base_name = f"{family_name}-{key}-mc{i}.output"
                output_path = os.path.join(output_dir, base_name)
                # creates the file if it doesn't exist
                with open_relative(output_path, "a+") as output_file:
                    output_file.seek(0)
                    existing_lines = [line.strip() for line in output_file.readlines()]
                    to_skip = len(existing_lines)

                # if we're in patch mode, we need to investigate the file anyway
                if patch:
                    to_skip = 0
                
                # only record file if its missing outputs
                if to_skip < len(self.battery):
                    iterate_structure.append([ output_path, to_skip ])

            if len(iterate_structure) == 0:
                if not quiet:
                    print("No new cases necessary to generate, not loading model")
                continue
            
            torch.cuda.empty_cache()
            # lazily access it, so we don't actually need to load the model unless we generate
            get_model = init_lazy_model(model_name)

            @with_progress(len(self.battery))
            def iterate(output_file, *, step=None):
                output = None
                test_case = self.battery[step]
                specific_prompt = prompt.format(prompt=test_case)
                patching = False
                if patch:
                    existing_output = existing_lines[step]
                    if existing_output != "":
                        output = existing_output
                    else:
                        patching = True
                        print(f"Regenerating empty test case index {step} for {repr(test_case)} with prompt {repr(specific_prompt)}...")
                    
                if output is None:
                    output = get_model().generate_until(specific_prompt, stops=["\n"], **kwargs)

                # output is now returned as a string
                if output is None:
                    output = ""
                decoded = output.strip()
                #if output is None:
                #    print("Warning: Model returned no output (prompt may have been too large)")
                #    decoded = ""
                #else:
                #    decoded = get_model().decode(output).strip()

                if "\n" in decoded:
                    if not quiet:
                        print("!! WARNING !! newline found in output")
                        print("Input: ", test_case)
                        print("Prompt: ", repr(specific_prompt))
                        print("Decoded: (next line)")
                        print(repr(decoded))
                    decoded = decoded.split("\n")[0]

                if patching:
                    print(f"Patched output: {repr(decoded)}")
                
                output_file.write(decoded + "\n")
                
                if get_model(load=False) is not None and get_model().inputs is not None:
                    del get_model().inputs
                    # ensure we don't null-read
                    get_model().inputs = None
        
            for output_path, to_skip in iterate_structure:
                if not quiet:
                    print(f"Opening {output_path}...")
                with open_relative(output_path, "a+") as output_file: 
                    if to_skip > 0 and not quiet:
                        print(f"{to_skip} entries found already, skipping that many...")
                    elif patch:
                        print(f"Preparing file for patching...")
                        output_file.write(PATCH_SEPARATOR)
                    iterate(output_file, skip=to_skip)

            if patch:
                with open_relative(output_path, "r") as source:
                    parts = source.read().split(PATCH_SEPARATOR)
                with open_relative(output_path, "w") as dest:
                    dest.write(parts[-1])

            if get_model(load=False) is not None:
                get_model().free()
            #print("Breaking out of model family iteration loop early...")
            #return

    
    def init_cases(self, family, family_name=None):
        if family_name is None:
            family_name = ModelFamily.name_for(family)

        if self.json_battery:
            with open_relative(self.battery_path, "r") as battery:
                test_cases = json.loads(battery.read())["cases"][:self.case_count]
                self.answer_key = [ obj["truth"].strip() for obj in test_cases ]
        else:
            with open_relative(self.truth_path, "r") as truth_file:
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
                with open_relative(output_path, "r") as output_file:
                    answers = [
                        clean_model_output(answer)
                        for answer in output_file.readlines()
                    ]
                family_answers[key] = answers
            prompt_family_answers.append(family_answers)

        self.prompt_family_answers = prompt_family_answers

        if self.base is not None:
            self.base.init_cases(family, family_name)

    
    def init_render(self, *args, **kwargs):
        return self.init_cases(*args, **kwargs)

    
    def calculate_metrics(self, metric, limit=None, cache=True):
        if limit is not None:
            cache = False
        
        if cache:
            cache_file_path = os.path.join("./output", self.task, "metrics.json")
            with open_relative(cache_file_path, "a+", encoding="utf-8") as cache_file:
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
            with open_relative(cache_file_path, "w", encoding="utf-8") as cache_file:
                cache_file.write(json.dumps(cache_obj))
        
        return by_prompt
        
    
    def render_metric(
        self,
        metric,
        by_prompt=None,
        render_to=None,
        *args,
        **kwargs
    ):
        if by_prompt is None:
            by_prompt = self.calculate_metrics(metric)

        self.renderer = OutputRenderer(
            baseline=metric.baseline,
            metric=metric.name,
        )

        self.renderer.render(ys=by_prompt, render_to=render_to, *args, **kwargs)
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
            linear_axes.append(index_axis(axes, i))

        i += 1
        while i < width * height:
            # turn off extra subplots
            index_axis(axes, i).axis("off")
            i += 1
        
        colors = sample_cmap(cmap, count=len(names))
        
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
            plt.savefig(save, bbox_inches="tight")
        plt.show()


    def calculate_perturbations(self, metric, series_name="prompt0"):
        assert self.base is not None, "Cannot calculate perturbations for model without base battery"

        # TODO: optional extras
        extras_path = os.path.join(self.battery_path, "extras.json")
        with open_relative(extras_path, "r") as file:
            extras = json.loads(file.read())
            
        # stored as e.g. plot_xs["prompt0"]["350M"] i.e. plot_xs[series_name][model_key]
        plot_data = {}
        for idx, family_answers in enumerate(self.prompt_family_answers):
            series_name = f"prompt{idx}"
            result = {}
            plot_data[series_name] = {}
            for model_idx, (model_key, answers) in enumerate(family_answers.items()):
                plot_data[series_name][model_key] = model_plot = {
                    "xs": [],
                    "ys": [],
                    "diff_ys": [],
                }
                for answer_idx, answer in enumerate(answers):
                    extra = extras[answer_idx]
                    truth = self.answer_key[answer_idx]
                    base_answer = self.base.prompt_family_answers[idx][model_key][answer_idx]
                    distance = extra["lev"]
                    # TODO: make this work with newlines in output
                    try:
                        score = metric.grade_single(truth, answer, silent=True)
                    except:
                        print(
                            "Warning: perturbed score could not be calculated, skipping. Origin: (",
                            series_name, model_key, answer_idx,
                            ")...",
                            end=""
                        )
                        next
                    try:
                        score_baseline = metric.grade_single(truth, base_answer, silent=True)
                    except:
                        print(
                            "Warning: baseline score could not be calculated, skipping. Origin: (",
                            series_name, model_key, answer_idx,
                            ")...",
                            end=""
                        )
                        next
                    model_plot["xs"].append(distance)
                    model_plot["ys"].append(score)
                    model_plot["diff_ys"].append(score - score_baseline)
        
        return plot_data

    def _render_perturbations(
        self,
        metric,
        plot_data,
        series_name,
        save=None,
        colors=None,
        y_target="ys",
        ylim=(0, 1),
        center_axis=False,
        xlabel="Levenshtein Distance",
        ylabel=None,
        title=None,
    ):
        if colors is None:
            BASE_COLORS = sample_cmap("viridis", count=4, lower=0, upper=0.9)
        else:
            BASE_COLORS = colors
        
        height = 2
        width = 2
        fig, axes = plt.subplots(
            height, width,
            figsize=(10, 10)
            #, sharex=True, sharey=True
        )
        
        for idx, (model_key, target_data) in enumerate(plot_data[series_name].items()):
            ax = index_axis(axes, idx)
            ax.set_ylim(*ylim)
            xs = np.array(target_data["xs"])
            ys = np.array(target_data[y_target])
            color = BASE_COLORS[idx]
            marker = "o"
            
            ax.scatter(xs, ys, color=color, label=model_key, alpha=1, marker=marker)
            linreg = scipy.stats.linregress(xs, ys)
            # we only graph unique sorted xs cuz we don't want to draw multiple lines atop each other
            unique_xs = np.unique(np.sort(xs))
            # y = mx + b
            trend_ys = linreg.slope * unique_xs + linreg.intercept
            #trendline = np.poly1d(np.polyfit(xs, ys, 1))
            series_single, = ax.plot(
                unique_xs,
                trend_ys,
                color="red",
                #color=color,
                linestyle="--",
                label=f"{model_key} trend"
            )

            if center_axis:
                ax.spines["left"].set_position(("data", 0))
                ax.spines["bottom"].set_position(("data", 0))
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            
            ax.legend()
            print(f"{series_name} & {model_key} & {linreg.slope:.5f} & {linreg.rvalue:.5f} \\\\")
            ax.set_xlabel(f"{model_key}; $m={linreg.slope:.3f}$, $R^2={linreg.rvalue**2:.3f}$")

            if center_axis:
                ax.xaxis.set_label_coords(x=0.5, y=0)
        
        # hack to get labels properly aligned around the entire graph
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)
        plt.xlabel(xlabel, labelpad=20)
        plt.ylabel(ylabel)
        plt.suptitle(title)
        plt.tight_layout()

        if save:
            plt.savefig(save, bbox_inches="tight")
        
        plt.show()
    
    def render_perturbations(self, metric, plot_data=None, series_name="prompt0", save=None, colors=None):
        if plot_data is None:
            plot_data = self.calculate_perturbations(metric)

        return self._render_perturbations(
            metric=metric,
            plot_data=plot_data,
            series_name=series_name,
            save=save,
            colors=colors,
            y_target="ys",
            ylim=(0, 1),
            ylabel=f"{metric.name} after Perturbation",
            title=f"{series_name}: Perturbed {metric.name} vs Levenshtein Distance"
        )


    def render_perturbations_relative(self, metric, plot_data=None, series_name="prompt0", save=None, colors=None):
        if plot_data is None:
            plot_data = self.calculate_perturbations(metric)

        return self._render_perturbations(
            metric=metric,
            plot_data=plot_data,
            series_name=series_name,
            save=save,
            colors=colors,
            y_target="diff_ys",
            ylim=(-1, 1),
            center_axis=True,
            ylabel=f"Improvement {metric.name} between base and Perturbation",
            title=f"{series_name}: Change in {metric.name} after Perturbation vs Levenshtein Distance",
        )

    
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
            cache_file = open_relative(bootstrap_cache, "r+")
        else:
            cache_file = open_relative(bootstrap_cache, "w+")
        
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
            with open_relative(bootstrap_cache, "w+") as cache_file:
                cache_file.write(json.dumps(cache_config))
        
        return by_prompt

    
    def free(self):
        pass