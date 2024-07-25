# Emergent Capabilities of LLMs for Software Engineering
A growing interest for Large Language Models (LLMs) is how increasing their size might result in changes to their behavior not predictable from relatively smaller-scaled models. Analyzing these emergent capabilities is therefore crucial to understanding and developing LLMs. Yet, whether LLMs exhibit emergence, or possess emergent capabilities, is a contested question. Furthermore, most research into LLM emergence has focused on natural language processing tasks and models suited for them.

We focus on investigating emergence in the context of software engineering, and recontextualize the discussion of emergence in the context of prior research. We propose a multifaceted pipeline for evaluating and reasoning about emergent capabilities of LLMs in any context and instantiate this pipeline to analyze the emergent capabilities of the CodeGen1-multi model across four scales ranging from 350M parameters to 16.1B parameters. We examine the model's performance on the software engineering tasks of automatic bug fixing, code translation, and commit message generation. We find no evidence of emergent growth at this scale on these tasks and consequently discuss the future investigation of emergent capabilities.

## How to Replicate

### Installing the remote tests
Our `pull-tests.ipynb` includes many candidate avenues of research, but the only one to install is under the header `CoDiSum's data4CopynetV3.zip`. Otherwise, `cd data && git clone https://github.com/microsoft/CodeXGLUE.git` instantiates the other tasks' data.

Each software engineering task has a corresponding `.ipynb` file which is responsible for loading the models (defined in `model_wrapper.py`), tasks (defined in `run_battery.py`) and metrics (defined in `metric.py`). We include `bugs2fix.ipynb`, `bugs2fix-checklist.ipynb`, `code2code-trans.ipynb`, and `commit-message.ipynb` to run the models on the tasks and grade them according to our metrics, generating the results graphs. We generate the additional graphs (i.e. bootstrapping) with `metric-progress.ipynb`. We generate our tables with `tabulate-results.ipynb`.

In general, our software engineering tasks follow the following code format:

```py
from run_battery import BatteryRunner, BatteryConfigs

runner = BatteryRunner.of(BatteryConfigs.TaskName) # replace with correct config name
runner.load_cases()

# Generate results
from model_wrapper import ModelFamily
runner.run_battery(
    family=ModelFamily.CodeGen1.multi, # e.g.
    patch=False, # change to True if you want to fill in blank lines, shouldn't be necessary
)

# Render graphs
import metric
runner.init_render(family=ModelFamily.CodeGen1.multi) # e.g.
runner.render_metric_multi(
    [ metric.ExactMatch, metric.BLEU, metric.CodeBLEUJava ],
    save="./figs/OUTPATH-path-all.png",
)
```


## Repository Structure

### Primary Files
- `bugs2fix.ipynb` generates the graphs for the Bugs2Fix code repair task.
- `bugs2fix-checklist.ipynb` generates the graphs for the Bugs2Fix (Checklist) code repair task.
- `code2code-trans.ipynb` generates the graphs for the Code2Code code translation tastk.
- `commit-message.ipynb` generates the graphs for the commit message generation task.
- `tabulate-results.ipynb` generates the tables and pulls together the information.

- `pull-tests.ipynb` installs the datasets from BIG and other various places. (Note: CodeXGLUE was not installed this way - the repository was simply cloned to `data/CodeXGLUE`.)

- `bleu.py` is code adapted from CodeXGLUE which calculates the BLEU metric.
- `metric.py` is a wrapper around the various metrics we used in this project.
- `model_wrapper.py` is a wrapper around (specifically) the CodeGen extended family of models.
- `render_output.py` is a wrapper around matplotlib suited for our usecase.
- `run_battery.py` is helper code which streamlines the testcase running process.
- `timehelp.py` is helper code which is responsible for timing operations and formatting them.

### Scaffolding Files

- `accelerate-test.ipynb` is scaffolding code which became the basis for `model_wrapper.py`, testing GPU loading & caching of the codegen models.
- `codexglue-test.ipynb` is a scratchpad for initial testing of various prompts.
- `consecutive.ipynb` is proof of concept code loading the models in sequence without leaking GPU memory.
- `metric-progress.ipynb` investigates the change in metric according to number of test cases processed.
- `softmax.ipynb` is verifying the correspondence between softmax and logits.
- `testing.ipynb` is used for miscellaneous testing, but primarily the parsing of multiple choice questions.
- `verify-result.ipynb` is debugging code used to examine questionable input/output pairs and assess what caused them (in this case, a bug/false assumption in the generation code).
- `wrapper-test.ipynb` is a simple testing file for making sure the model wrapper works correctly.
- `test.py` is an old testing file.
- `trim-tokens.ipynb` was a planned experiment to normalize token lengths across experiments.