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
        self.input_ids = input_ids
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
        
    def run(self):
        # init values
        batch_size, cur_len = self.input_ids.shape
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.input_ids.device)
        this_peer_finished = False
        while not this_peer_finished:
            candidate_input_ids, candidate_logits = self._generate_candidates()
            is_done_candidate = self.stopping_criteria(candidate_input_ids, None)
            num_generated_candidates = candidate_input_ids.shape[1] - self.input_ids.shape[1]
            new_logits, n_matches, num_valid_tokens = self._verify_candidates(candidate_input_ids, is_done_candidate)
            self._update_new_candidate_length(n_matches)
            self._update_verifier_level()
            logger.info(f"candidates: {num_generated_candidates}, valid tokens: {num_valid_tokens}")
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
        # Naive version for 1 verifier
        self.verifier_level = 0
    
    def _verify_candidates(self, candidate_input_ids: torch.LongTensor, is_done_candidate: bool):
        verified_input_ids, new_logits, n_matches, num_valid_tokens = self.verifier_list[self.verifier_level].get_continuation(
            candidate_input_ids, self.input_ids, is_done_candidate)
        # copy verified_input_ids to self.input_ids
        self.input_ids = verified_input_ids
        return new_logits, n_matches, num_valid_tokens
    
    def _generate_candidates(self):
        candidate_input_ids, candidate_logits = self.candidate_generator.get_candidates(self.input_ids)
        return candidate_input_ids, candidate_logits
    