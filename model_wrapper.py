import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from timehelp import time_start, time_end, display_header, with_progress
import re
import gc
import os
from math import exp
from enum import Enum

class ModelFamily:
    # fun fact: CodeGen.nl came first; CodeGen.multi was built atop that, and mono atop multi 
    class CodeGen1:
        mono = {}
        multi = {}
        nl = {}
        
    CodeGen2 = {}
    
    class CodeGen2p5:
        mono = "Salesforce/codegen25-7b-mono"
        multi = "Salesforce/codegen25-7b-multi"
        instruct = "Salesforce/codegen25-7b-instruct"

    @staticmethod
    def name_for(family):
        if family == ModelFamily.CodeGen1.mono:
            return "codegen1-mono"
        if family == ModelFamily.CodeGen1.multi:
            return "codegen1-multi"
        if family == ModelFamily.CodeGen1.nl:
            return "codegen1-multi"
        if family == ModelFamily.CodeGen2:
            return "codegen2"
        # CodeGen2.5 is "weird" in that its the only model here with only one model size per kind (i.e. 7B)
        if family == ModelFamily.CodeGen2p5:
            return "codegen2p5"
        if family == ModelFamily.CodeGen2p5.mono:
            return "codegen2p5-mono"
        if family == ModelFamily.CodeGen2p5.multi:
            return "codegen2p5-multi"
        if family == ModelFamily.CodeGen2p5.instruct:
            return "codegen2p5-instruct"

        assert False, f"Cannot provide nice name for {family}"

# these model sizes are what huggingface uses, but are actually (perhaps unsurprisingly) rounded from their true values
for size in [ "350M", "2B", "6B", "16B" ]:
    for label in [ "mono", "multi", "nl" ]:
        model_name = f"Salesforce/codegen-{size}-{label}"
        getattr(ModelFamily.CodeGen1, label)[size] = model_name

for size in [ "1B", "3.7B", "7B", "16B" ]:
    safe_size = size.replace(".", "_")
    model_name = f"Salesforce/codegen2-{safe_size}"
    ModelFamily.CodeGen2[size] = model_name

MultipleChoiceStrategy = Enum("MultipleChoiceStrategy", [
    "MULTIPLY",
    "LOGIT_AVERAGE",
    "FIRST_BRANCH",
])

def abbreviate_string(s, start=30, end=30):
    if len(s) <= start + end:
        return s
    return s[:start] + f" [ ... {len(s) - start - end} bytes abbreviated ... ] " + s[-end:]


# Note: Aggressive use of `del` to try and coax CUDA memory back to be freed.
# Note: This is probably not necessary. I think I was just running out of memory
# because my max_size for generate_until was too large to prevent CUDA memory overflow
# for larger models - presumably cuz the tensors are larger? I have no idea tbh.
# max=1000 worked fine for 2B but not 6B; likewise the max=500 might not work for
# 16B, but I will cross that bridge when I get there.

# MAJOR TODO: truncate all existing output files to 500 tokens for consistency in grading!

def find_contiguous_subtensor_index(a, b):
    if b.numel() == 0:
        return 0
    if b.numel() > a.numel():
        return None
    if b.numel() == 1:
        indices = torch.nonzero(a == b.item(), as_tuple=False)
        if indices.numel() > 0:
            item = indices[0].item()
        else:
            item = None
        del indices
        return item
    
    for i in range(start, a.numel() - b.numel() + 1):
        tensor_slice = a[i:i + b.numel()]
        if torch.equal(tensor_slice, b):
            del tensor_slice
            return i
        del tensor_slice
        
    return None

def find_contiguous_subtensor_index_after_content(a, b):
    if b.numel() == 0:
        return 0
    if b.numel() > a.numel():
        return None
    if b.numel() == 1:
        indices = torch.nonzero(a == b.item(), as_tuple=False)
        if indices.numel() > 0:
            index_pointer = 0
            # ignore found instances at the head of the search range
            while index_pointer < indices.numel() and indices[index_pointer].item() == index_pointer:
                index_pointer += 1
            
            if index_pointer >= indices.numel():
                item = None
            else:
                item = indices[index_pointer].item()
            
            del indices
            return item
        else:
            del indices
            return None
    
    assert False, "Have not yet handled multi-token needle for find_contiguous_subtensor_index_after_content"

def find_stop_index_after_content(base, stop):
    assert len(stop) == 1, "Assuming stops are single characters"

    stripped = base.lstrip(stop)
    stop_index = stripped.find(stop)
    if stop_index == -1:
        return None
    
    stop_index += len(base) - len(stripped)
    return stop_index

class Model:
    CACHE_DIR = "/workspaces/emergent-capabilities/datax"
    DEFAULT_SOFTMAX = torch.nn.Softmax(dim=-1)

    @staticmethod
    def clean_cache_dir(confirm=False):
        if not confirm:
            print("Please confirm! Call as `Model.clean_cache_dir(confirm=True)`!")
            return
        
        import os
        os.system(f"rm -rfv {CACHE_DIR}")

    
    @staticmethod
    def prob_from_logit(logit):
        assert False, "Do not use this function"
    
    def __init__(self, name, cache_dir=None, device_name=None, verbose=True, softmax=None):
        self.verbose = verbose
        self.name = name
        self.tokenizer = None
        self.model = None
        self.cache_dir = cache_dir or Model.CACHE_DIR
        self.device_name = device_name
        self.device = None
        self.softmax = softmax or Model.DEFAULT_SOFTMAX
        self._tokenized_eos_token = None
        # used to cache the inputs given, if transformed e.g.
        self.inputs = None
    
    
    def yap(self, *args, **kwargs):
        if not self.verbose:
            return

        print(*args, **kwargs)

    
    def configure_device(self):
        assert self.device is None, "Device already exists, cannot re-configure"
        self.yap("Configuring torch device...")
        
        if self.device_name is None:
            self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
            if self.device_name == "cpu":
                self.yap("Warning: Model is running on CPU")
        self.device = torch.device(self.device_name)
        
        self.yap("Using device:", self.device_name, "aka", self.device)

    
    def configure_tokenizer(self):
        assert self.tokenizer is None, "Tokenizer already exists, cannot re-configure"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name,
            cache_dir=self.cache_dir,
            device_map=self.device,
            # padding_side="left",
        )
        # for padding; doesn't quite work, though
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # alternative suggestion for padding:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    @property
    def tokenized_eos_token(self):
        if self._tokenized_eos_token is None:
            self._tokenized_eos_token = self.tokenize(self.tokenizer.eos_token)["input_ids"]

        return self._tokenized_eos_token
    
    def configure_model(self, force_device=False):
        assert self.model is None, "Model already exists, cannot re-configure"
        # torch.cuda.empty_cache()
        self.yap("Obtaining model...")

        if force_device:
            self.yap("Warning: Forcing the device will not allow the model to use non-GPU resources in some cases")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            cache_dir=self.cache_dir,
            device_map=self.device if force_device else "auto"
        )
        
        if force_device:
            self.yap("Forcing model on requested device", use_device, "...")
            self.model = self.model.to(use_device)

        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

    
    def configure(self, time=False):
        if time:
            time_start("model.device")
        self.configure_device()
        if time:
            time_end()
            time_start("model.tokenizer")
        self.configure_tokenizer()
        if time:
            time_end()
            time_start("model.model")
        self.configure_model()
        if time:
            time_end()

    
    def tokenize(self, prompt, time=False):
        if time:
            time_start("model.tokenize")
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
        prompt_tokens = prompt_tokens.to(self.device)
        if time:
            time_end()
        # self.yap("Token count in input:", prompt_tokens["input_ids"].shape[1])
        return prompt_tokens

    
    # TODO: figure out if this is necessary
    def model_no_grad(self, *args, **kwargs):
        value = None
        with torch.no_grad():
            value = self.model(*args, **kwargs)
        return value

    # e.g., max_length=128
    def generate(self, inputs, time=False, auto_tokenize=True, *args, **kwargs):
        if isinstance(inputs, str):
            assert auto_tokenize, "Cannot generate given string input prompt when auto_tokenize=False"
            inputs = self.tokenize(inputs, time=time)
        self.inputs = inputs

        self.yap("Generating...")
        if time:
            time_start("model.generate")

        sample = self.model.generate(*args, **inputs, **kwargs)

        if time:
            time_end()
        
        return sample

    # max_size default set comfortably below model max (2048)
    def generate_until(
        self,
        inputs,
        stops=[],
        per_step=50,
        # TODO: configure max_size based on model family?
        max_size=500,
        truncate=True,
        auto_tokenize=True,
        time=False,
        *args, **kwargs
    ):
        """
        stops is a list of string
        returns a string
        """
        assert truncate, "truncate=False is not supported"
        
        if isinstance(inputs, str):
            assert auto_tokenize, "Cannot generate given string input prompt when auto_tokenize=False"
            inputs = self.tokenize(inputs, time=time)
        
        original_inputs = inputs

        #stops = [
        #    self.tokenize(stop)["input_ids"] if isinstance(stop, str)
        #    else stop
        #    for stop in stops
        #]

        base_size = inputs["input_ids"].size(dim=1)
        
        tokens = None
        result_string = None
        force_stop = False
        # it = 0
        while True:
            next_size = inputs["input_ids"].size(dim=1) + per_step
            # print("GEN ITER", it, next_size, "!!!!!!!!")
            # it += 1
            
            if next_size > max_size:
                print("!! max size might be exceeded !!")
                print("inputs so far:", abbreviate_string(
                    self.decode(inputs["input_ids"]),
                    start=400,
                    end=100
                ))
                output = inputs["input_ids"]
                force_stop = True
            else:
                output = self.generate(inputs, max_new_tokens=per_step, auto_tokenize=False)
            # remove input given so far from output 
            
            output_trimmed = output[:, original_inputs["input_ids"].size(dim=1):]
            del output
            output = output_trimmed

            if force_stop:
                result_string = self.decode(output)
                del output
                break
                
            
            if tokens is None:
                tokens = output
            else:
                tokens_together = torch.cat((tokens, output), dim=1)
                del tokens
                tokens = tokens_together

            # effectively left-strips the input of stop subsequences before searching
            # for a stop index
            # NOTE: since the program takes a string representation of stops,
            # we are not actually concerned about tokens; e.g.,
            # token "\n" (198) != token "\n\n" (628)
            # SO, we must examine the decoded string representation
            decoded = self.decode(output)
            # print(f"{decoded = }")
            stop_indices = [
                index
                for stop in stops
                if (
                    index := find_stop_index_after_content(decoded, stop)
                ) is not None
            ]
            if len(stop_indices) == 0:
                stop_index = None
            else:
                stop_index = min(stop_indices)
            ###print("decoded:", decoded)
            ###print("stops:", stop_indices, ";", stops)
            #stop_index = next(
            #    (
            #        inner_index
            #        for stop in stops
            #        if (inner_index :=
            #            find_contiguous_subtensor_index_after_content(
            #                tokens[0],
            #                stop,
            #            )
            #        ) is not None
            #    ),
            #    None
            #)
            ###print("Stop index (before eos search)", stop_index)

            # if the stop was not found, look for the eos token to make sure we
            # do not generate past it
            if stop_index is None:
                eos_index = find_contiguous_subtensor_index(
                    tokens[0],
                    self.tokenized_eos_token
                )

                if eos_index is not None:
                    tokens_truncated = tokens[:, :stop_index]
                    del tokens
                    tokens = tokens_truncated
                    result_string = self.decode(tokens_truncated)
                    break

            else:
                # truncate to stop index
                result_string = decoded[:stop_index]
                break
                ###print("Tokens before truncation", tokens)
                # tokens_truncated = tokens[:, :stop_index]
                # del tokens
                # tokens = tokens_truncated
                # break
            
            next_inputs = self.concatenate_tokens(inputs, output)
            del inputs, output
            inputs = next_inputs

        # free running input; we don't need it anymore
        del inputs
        for _ in range(len(stops)):
            del stops[0]

        self.inputs = original_inputs
        return result_string
        
        if truncate:
            return tokens
        else:
            result = torch.cat((original_inputs["input_ids"], tokens), dim=1)
            del tokens
            return result
    
    def multiple_choice_token(self, inputs, targets, time=False):
        assert len(targets) >= 2, "Expected at least 2 targets" 
        if isinstance(inputs, str):
            inputs = self.tokenize(inputs, time=time)

        if time:
            time_start("model.generate_single")
        
        output = self.model_no_grad(input_ids=inputs["input_ids"])
        logits = output.logits[:, -1, :]

        if all(isinstance(target, str) for target in targets):
            target_ids = self.tokenizer.convert_tokens_to_ids(targets)
        else:
            assert all(
                isinstance(target, int) or isinstance(target, tensor)
                for target in targets
            ), "Expected string or (tensor/int) array for target"
            target_ids = targets
        subset_logits = logits[:, target_ids]
        predicted_idx = torch.argmax(subset_logits, dim=-1).item()
        predicted_token = targets[predicted_idx]

        if time:
            time_end()

        return predicted_idx, predicted_token
        
    
    def append_token(self, source, extra):
        """Non-mutating. Combines an object-formatted token listing with a single token id"""
        input_ids = source["input_ids"]
        attention_mask = source["attention_mask"]
        
        extra_token = torch.tensor([[extra]], device=self.device)
        extra_attention = torch.tensor([[1]], device=self.device)
        
        result = {
            "input_ids": torch.cat((input_ids, extra_token), dim=1),
            "attention_mask": torch.cat((attention_mask, extra_attention), dim=1),
        }
        
        del extra_token, extra_attention

        return result

    
    def concatenate_tokens(self, source, extra):
        """Non-mutating. Combines an object-formatted token listing with 2D tensor of extra tokens"""
        input_ids = source["input_ids"]
        attention_mask = source["attention_mask"]
        
        extra_attention = torch.tensor([[1] * extra.size(dim=1)], device=self.device)
        
        result = {
            "input_ids": torch.cat((input_ids, extra), dim=1),
            "attention_mask": torch.cat((attention_mask, extra_attention), dim=1),
        }

        del extra_attention

        return result
        
    
    def _multiple_choice_prompts_first_branch(self, input_tokens, target_tokens, time=False):
        """
        Private helper function.
        Prefer model.multiple_choice_prompts(..., strategy=MultipleChoiceStrategy.FIRST_BRANCH)
        Inputs:
         - input_tokens is list of tokens
         - target_tokens is list of tuples (idx, list of tokens)
        Returns:
         - idx of the best target (indexabale into target_tokens)
        """
        longest = max(len(tokens) for idx, tokens in target_tokens)
        # walk through each token until we only have one possible target left
        for token_idx in range(longest):
            # select those that extend this far
            target_tokens = [
                (target_idx, tokens)
                for target_idx, tokens in target_tokens
                if token_idx < tokens.shape[1]
            ]
            
            # self.yap(f"@{token_idx}:", "Targets:", target_tokens)
            
            # save a copy of the inputs we can extend off of
            my_inputs = input_tokens
            # take a cross-section of the tokens which extend this far
            interim_targets = [*{
                # convert the tensor to an int so we can index using it
                int(tokens[0, token_idx])
                for _, tokens in target_tokens
            }]

            assert len(interim_targets) != 0, \
                "Expected at least one toke in cross-section " + repr(interim_targets)

            if len(interim_targets) == 1:
                # we have, by process of elimination, found our target
                break

            # which of the tokens in the cross section is most likely?
            _, predicted_token = self.multiple_choice_token(
                my_inputs,
                targets=interim_targets,
                time=time
            )
            
            # append the predicted token to our context so far
            my_inputs = self.append_token(my_inputs, predicted_token)

            # select only those responses whose current cross section matches the predicted token
            target_tokens = [
                (target_idx, tokens)
                for target_idx, tokens in target_tokens
                if tokens[0, token_idx] == predicted_token
            ]
            
            if len(target_tokens) == 1:
                break

        assert len(target_tokens) == 1, \
            f"Expected only 1 result left, got {len(target_tokens)}"

        
        best_idx, best_tokens = target_tokens[0]
        return best_idx

    
    def _multiple_choice_prompts_logit_average(self, input_tokens, target_tokens, time=False):
        """
        Private helper function.
        Prefer model.multiple_choice_prompts(..., strategy=MultipleChoiceStrategy.LOGIT_AVERAGE)
        Inputs:
         - input_tokens is list of tokens
         - target_tokens is list of tuples (idx, list of tokens)
        Returns:
         - idx of the best target (indexabale into target_tokens)
        """
        base_output = self.model_no_grad(input_ids=input_tokens["input_ids"])
        base_logits = base_output.logits[:, -1, :]
        best_score = float("-inf")
        best_option_idx = None
        
        for idx, tokens in target_tokens:
            self.yap(idx, tokens, "!!!!")
            score = base_logits[:, tokens[0, 0]].item()
            running_inputs = input_tokens["input_ids"]
    
            self.yap("initial score =", score)
            
            for j in range(1, tokens.shape[1]):
                token = tokens[0, j]
                token_formatted = token.unsqueeze(0).unsqueeze(0)
                running_inputs = torch.cat((running_inputs, token_formatted), dim=-1)
                
                output = self.model_no_grad(input_ids=running_inputs)
                next_logits = output.logits[:, -1, :]
                self.yap("Inner score:", next_logits[:, token])
                score += next_logits[:, token].item()
    
            self.yap("Final score =", score)
            # so i'm told, we can normalize logits like this
            score /= tokens.shape[1]
            self.yap("Normalized =", score)
            
            if best_option_idx is None or score > best_score:
                best_score = score
                best_option_idx = idx
        
        return best_option_idx
        
    def _multiple_choice_prompts_multiply(self, input_tokens, target_tokens, time=False):
        """
        Private helper function.
        Prefer model.multiple_choice_prompts(..., strategy=MultipleChoiceStrategy.MULTIPLY)
        Inputs:
         - input_tokens is list of tokens
         - target_tokens is list of tuples (idx, list of tokens)
        Returns:
         - idx of the best target (indexabale into target_tokens)
        """
        
        base_output = self.model_no_grad(input_ids=input_tokens["input_ids"])
        base_logits = base_output.logits[:, -1, :]
        best_score = float("-inf")
        best_option_idx = None
        idx = None

        # in general, if a word A has parts a0 a1 ... aN, we can calculate
        #   P(A|H) = P(a0|H) * P(a1|H.a0) * ... * P(aN|H.a0.a1...a(N-1))
        for idx, tokens in target_tokens:
            # goal: calculate P(tokens|H) = 𝚷 P(aj|H.∑ak 0<=k<j) 0<=j<=N
            first_token = tokens[0, 0]
            # logit_score = base_logits[:, tokens[0, 0]].item()
            # P(a0|H)
            # total_prob = Model.prob_from_logit(logit_score)
            initial_distribution = self.softmax(base_logits)
            total_prob = initial_distribution[:, first_token].item()
            print(f"init = {total_prob * 100:.4f}%")
            # H
            running_inputs = input_tokens["input_ids"]

            # compute product
            for j in range(1, tokens.shape[1]):
                token = tokens[0, j]
                token_formatted = token.unsqueeze(0).unsqueeze(0)
                running_inputs = torch.cat((running_inputs, token_formatted), dim=-1)
                
                output = self.model_no_grad(input_ids=running_inputs)
                next_logits = output.logits[:, -1, :]
                # self.yap("NEXT LOGITS:", next_logits)

                distribution = self.softmax(next_logits)
                # self.yap("SOFTMAX:", distribution)

                # logit_score = next_logits[:, token].item()
                prob = distribution[:, token].item()
                # self.yap(f"Token {j}: P={prob}, logit={logit_score}")
                print(f"prob = {prob * 100:.4f}%")
                total_prob *= prob
                # self.yap("Running P:", total_prob)
            
            score = total_prob
            print(f"overall = {total_prob * 100:.120f}%")
            
            if best_option_idx is None or score > best_score:
                best_score = score
                best_option_idx = idx
            # self.yap()
            
        return idx


    def _tokenize_label(self, targets, time=False):
        """
        Tokenizes the strings in the list of targets given.
        Pairs each list of tokens with a corresponding index.
        """
        return [
            (
                idx,
                self.tokenize(target, time=time)["input_ids"] if isinstance(target, str)
                else target
            )
            for idx, target in enumerate(targets)
        ]
    
    def multiple_choice_prompts(self, inputs, targets, time=False, strategy=MultipleChoiceStrategy.MULTIPLY):
        """
        Given a prompt context, computes which of the provided targets is most likely.
        Inputs:
         - inputs, either a string or list of tokens, representing the context given for the prompt.
            NOTE: strings will be tokenized using the model's tokenizer.
         - targets, either a list of strings or list of list of tokens.
            NOTE: there should be no duplicate target options
            NOTE: string elements will be tokenized using the model's tokens. 
         - time, a boolean corresponding to whether timing information should be displayed
            NOTE: propogates to all called methods
         - strategy, an enum from MultipleChoiceStrategy, corresponding to the method the function uses to assess likelihood. defaults to MULTIPLY.
        Returns:
         - idx corresponding to the most likely prompt
        """
        if isinstance(inputs, str):
            # self.yap("Tokenizing input prompt...")
            inputs = self.tokenize(inputs, time=time)

        # # TODO: deduplicate testing for target tokens
        # assert len({*targets}) == len(targets), "Cannot have duplicate targets"
    
        target_tokens = self._tokenize_label(targets, time=time)
        
        if strategy == MultipleChoiceStrategy.MULTIPLY:
            idx = self._multiple_choice_prompts_multiply(inputs, target_tokens, time=time)
            
        elif strategy == MultipleChoiceStrategy.LOGIT_AVERAGE:
            idx = self._multiple_choice_prompts_logit_average(inputs, target_tokens, time=time)
            
        elif strategy == MultipleChoiceStrategy.FIRST_BRANCH:
            idx = self._multiple_choice_prompts_first_branch(inputs, target_tokens, time=time)

        else:
            assert False, f"Unknown/unhandled multiple choice strategy {strategy}"
        
        return idx

    
    def decode(self, tokens, inputs=None):
        """If inputs provided (series of tokens), strips inputs from the tokens."""
        if inputs:
            tokens = tokens[:, inputs["input_ids"].shape[1]:][0]
        else:
            tokens = tokens[0]
        
        return self.tokenizer.decode(tokens)


    def free(self):
        """Frees associated GPU memory"""
        # TODO: use a `with` context
        del self.model, self.tokenizer, self.device
        gc.collect()
        torch.cuda.empty_cache()
