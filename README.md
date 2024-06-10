# emergent-capabilities

## Primary Files
- `bugs2fix.ipynb` generates the graphs for the Bugs2Fix code repair task.
- `code2code-trans.ipynb` generates the graphs for the Code2Code code translation tastk.
- `commit-message.ipynb` generates the graphs for the commit message generation task.
- `tabulate-results.ipynb` generates the tables and pulls together the information.

- `pull-tests.ipynb` installs the datasets from BIG and other various places. (I'm pretty sure CodeXGLUE was not installed this way - the repository was simply cloned to `data/CodeXGLUE`.)
- `trim-tokens.ipynb` (***TODO***) is to uniformly trim output lines to ensure all lines are at most 500 tokens long (useful because various configurations were used during the testing process).

- `bleu.py` is code adapted from CodeXGLUE which calculates the BLEU metric.
- `metric.py` is a wrapper around the various metrics we used in this project.
- `model_wrapper.py` is a wrapper around (specifically) the CodeGen extended family of models.
- `render_output.py` is a wrapper around matplotlib suited for our usecase.
- `run_battery.py` is helper code which streamlines the testcase running process.
- `timehelp.py` is helper code which is responsible for timing operations and formatting them.

## Scaffolding Files

- `accelerate-test.ipynb` is scaffolding code which became the basis for `model_wrapper.py`, testing GPU loading & caching of the codegen models.
- `codexglue-test.ipynb` is a scratchpad for initial testing of various prompts.
- `consecutive.ipynb` is proof of concept code loading the models in sequence without leaking GPU memory.
- `metric-progress.ipynb` investigates the change in metric according to number of test cases processed.
- `softmax.ipynb` is verifying the correspondence between softmax and logits.
- `testing.ipynb` is used for miscellaneous testing, but primarily the parsing of multiple choice questions.
- `verify-result.ipynb` is debugging code used to examine questionable input/output pairs and assess what caused them (in this case, a bug/false assumption in the generation code).
- `wrapper-test.ipynb` is a simple testing file for making sure the model wrapper works correctly.
- `test.py` is an old testing file.