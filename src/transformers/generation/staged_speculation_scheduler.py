import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, List, Union
import torch
from ..utils import logging
logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from .configuration_utils import GenerationConfig
    from .logits_process import LogitsProcessorList
    
from .candidate_generator import (
    AssistedCandidateGenerator,
    CandidateGenerator,
    CandidateVerifier,
    CascadeCandidateVerifier,
    PromptLookupCandidateGenerator,
    _crop_past_key_values,
    _prepare_attention_mask,
    _prepare_token_type_ids,
)

from .stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

class SpeculationScheduler:
    def __init__(
        self,
        input_ids: torch.LongTensor,
        generation_config: "GenerationConfig",
        logits_processor: "LogitsProcessorList",
        stopping_criteria: Optional[StoppingCriteriaList],
        draft_model: "PreTrainedModel",
        verifier_list: List["PreTrainedModel"],
        pad_token_id: Optional[int],
        eos_token_id: Optional[Union[int, List[int]]],
        model_kwargs: Dict,
        ):
        device = input_ids.device
        self.input_ids = input_ids # self.input_ids is verified by the last level model
        self.staged_input_ids = input_ids # self.staged_input_ids is verified by intermediate models but not the last level model
        self.new_candidate_length = draft_model.generation_config.num_assistant_tokens
        self.logits_processor = logits_processor
        self.stopping_criteria = stopping_criteria
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        self.candidate_generator = AssistedCandidateGenerator(
            input_ids=input_ids,
            assistant_model=draft_model,
            generation_config=generation_config,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
        )
        self.generation_config = generation_config
        self.verifier_list = []
        self.verifier_level = 0
        
        
        
        
        for verifier in verifier_list:
            cascade_verifier = CascadeCandidateVerifier(
                input_ids=input_ids,
                verifier_model=verifier,
                generation_config=generation_config,
                logits_processor=logits_processor,
                model_kwargs=model_kwargs,
            )
            self.verifier_list.append(cascade_verifier)
        
        self.verifier_length = len(self.verifier_list)
        
        # set verifier strategy using the default method
        def generate_verifier_strategy(verifier_strategy=None):
            if verifier_strategy is not None:
                return verifier_strategy
            else:
                verifier_strategy = []
                for i in range(len(self.verifier_list)):
                    verifier_strategy.append(((i) ** 2) * 10)
            return verifier_strategy
        
        self.verifier_strategy = generate_verifier_strategy()
        
    def run(self):
        # init values
        batch_size, cur_len = self.input_ids.shape
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.input_ids.device)
        this_peer_finished = False
        is_raising_level = False # init to be true to compare against self.input_ids
        while not this_peer_finished:
            logger.info(f"start new candidate generation, current level: {self.verifier_level}, new candidate length: {self.new_candidate_length}, staged length: {self.staged_input_ids.shape[1]}, input length: {self.input_ids.shape[1]}")
            candidate_input_ids, candidate_logits = self._generate_candidates()
            is_done_candidate = self.stopping_criteria(candidate_input_ids, None)
            num_generated_candidates = candidate_input_ids.shape[1] - self.staged_input_ids.shape[1]
            logger.info(f"generated {num_generated_candidates} candidates")
            new_logits, n_matches, num_valid_tokens, new_cache_size = self._verify_candidates(candidate_input_ids, is_done_candidate, is_raising_level)
            logger.info(f"verified {n_matches} candidates")
            self._crop_all_past_key_values(new_cache_size)            
            self._update_new_candidate_length(n_matches)
            is_raising_level = self._update_verifier_level()
            logger.info(f"raising level: {is_raising_level}")
            unfinished_sequences = unfinished_sequences & ~self.stopping_criteria(self.input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0
        
        return self.input_ids
            

    def _update_new_candidate_length(self, num_matches: int):
        # AIMD algorithm
        if num_matches == int(self.new_candidate_length):
            self.new_candidate_length += 2.0
        else:
            self.new_candidate_length = max(1.0, int(self.new_candidate_length / 2.0))            
        self.candidate_generator.set_num_assistant_tokens(self.new_candidate_length)
    
    def _update_verifier_level(self):
        # return if the level is raising
        # if true, then the next verifier should compare against self.input_ids
        # else it's in the same level or decreasing level, compare against staged results
        
        # determine the verifier level based on:
        # input_ids, staged_input_ids, self.new_candidate_length
        cur_level = self.verifier_level
        increment = self.new_candidate_length + self.staged_input_ids.shape[1] - self.input_ids.shape[1]
        # determine level by verifier_strategy[level] <= increment < verifier_strategy[level+1]
        level = 0
        for i in range(len(self.verifier_strategy)):
            if increment >= self.verifier_strategy[i]:
                level = i
        self.verifier_level = level
        
        # if staged length reaches max length, then raise to the last level verifier
        if self.staged_input_ids.shape[1] >= self.generation_config.max_length:
            self.verifier_level = self.verifier_length - 1
        
        if cur_level < self.verifier_level:
            return True
        else:
            return False
        
    
    def _verify_candidates(self, candidate_input_ids: torch.LongTensor, is_done_candidate: bool, is_raising_level: bool):
        if is_raising_level:
            verified_input_ids, new_logits, n_matches, num_valid_tokens, new_cache_size = self.verifier_list[self.verifier_level].get_continuation(
                candidate_input_ids, self.input_ids, is_done_candidate)
        else:
            verified_input_ids, new_logits, n_matches, num_valid_tokens, new_cache_size = self.verifier_list[self.verifier_level].get_continuation(
                candidate_input_ids, self.staged_input_ids, is_done_candidate)
        
        # stage the temporary results
        self.staged_input_ids = verified_input_ids
        
        # if this is the last level verifier, we can update the input_ids
        if self.verifier_level == self.verifier_length - 1:
            self.input_ids = verified_input_ids
        
        return new_logits, n_matches, num_valid_tokens, new_cache_size
    
    def _generate_candidates(self):
        candidate_input_ids, candidate_logits = self.candidate_generator.get_candidates(self.staged_input_ids) # always generate based on staged results
        return candidate_input_ids, candidate_logits
    
    def _crop_all_past_key_values(self, new_cache_size):
        # crop past_key_values for self.candidate_generator and all verifiers
        
        # crop candidate generator past_key_values
        if self.candidate_generator.assistant_kwargs.get("past_key_values", None) is not None:
            self.candidate_generator.assistant_kwargs["past_key_values"] = _crop_past_key_values(
                self.candidate_generator.assistant_model,
                self.candidate_generator.assistant_kwargs["past_key_values"],
                new_cache_size,
            )
        
        # crop verifier past_key_values
        for verifier in self.verifier_list:
            if verifier.verifier_kwargs.get("past_key_values", None) is not None:
                verifier.verifier_kwargs["past_key_values"] = _crop_past_key_values(
                    verifier.verifier_model,
                    verifier.verifier_kwargs["past_key_values"],
                    new_cache_size,
                )
        
    