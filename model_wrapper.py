import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from timehelp import time_start, time_end
import re
import gc
from math import exp
from enum import Enum

class ModelFamily:
    class CodeGen1:
        mono = {}
        multi = {}
        nl = {}
        
    CodeGen2 = {}
    
    class CodeGen2p5:
        mono = "Salesforce/codegen25-7b-mono"
        multi = "Salesforce/codegen25-7b-multi"
        instruct = "Salesforce/codegen25-7b-instruct"

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
        assert false, "Do not use this function"
        # TODO: check if this is the correct way to scale CodeGen logits;
        # it stands to reason that the falloff could be steeper or less steep

        # code from
        # https://sebastiansauer.github.io/convert_logit2prob/
        odds = exp(logit)
        prob = odds / (1 + odds)
        return prob

    
    def __init__(self, name, cache_dir=None, device_name=None, verbose=True, softmax=None):
        self.verbose = verbose
        self.name = name
        self.tokenizer = None
        self.model = None
        self.cache_dir = cache_dir or Model.CACHE_DIR
        self.device_name = device_name
        self.device = None
        self.softmax = softmax or Model.DEFAULT_SOFTMAX

    
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, cache_dir=self.cache_dir, device_map=self.device)
        # for padding; doesn't quite work, though
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # alternative suggestion for padding:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    
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

    def generate(self, inputs, time=False, *args, **kwargs):
        if isinstance(inputs, str):
            # self.yap("Tokenizing input prompt...")
            inputs = self.tokenize(inputs, time=time)

        self.yap("Generating...")
        if time:
            time_start("model.generate")

        sample = self.model.generate(*args, **inputs, **kwargs)

        if time:
            time_end()
        
        return sample

    
    def multiple_choice_token(self, inputs, targets, time=False):
        assert len(targets) >= 2, "Expected at least 2 targets" 
        if isinstance(inputs, str):
            # self.yap("Tokenizing input prompt...")
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
        
        return {
            "input_ids": torch.cat((input_ids, extra_token), dim=1),
            "attention_mask": torch.cat((attention_mask, extra_attention), dim=1),
        }

    
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
    
    
    def multiple_choice_prompts(self, inputs, targets, time=False, strategy=MultipleChoiceStrategy.MULTIPLY):
        """
        Given a prompt context, computes which of the provided targets is most likely.
        Inputs:
         - inputs, either a string or list of tokens.
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

        # TODO: deduplicate testing for input target tokens
        assert len({*inputs}) == len(inputs), "Cannot have duplicate targets"
    
        # tokenize each of the targets if necessary
        target_tokens = [
            (
                idx,
                self.tokenize(target, time=time)["input_ids"] if isinstance(target, str)
                else target
            )
            for idx, target in enumerate(targets)
        ]

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