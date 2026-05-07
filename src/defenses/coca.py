"""
CoCA Defense - Constitutional Calibration
Calibrates model output during inference using safety prompt comparison
"""

from typing import Dict, Any, List, Optional, Tuple
import re


class CoCADefense:
    """
    CoCA Defense Implementation
    
    Core idea: Calibrate model during inference by comparing output distributions
    between safety prompts and original prompts, awakening model's safety awareness
    """
    
    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant that must prioritize safety."
    ):
        self.name = "CoCA"
        self.description = "Constitutional Calibration - Inference-time Logits calibration"
        self.system_prompt = system_prompt
    
    def prepare_defense(
        self,
        sample: Dict[str, Any],
        attack_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare defense config
        
        Args:
            sample: Original sample
            attack_response: Attack response
            
        Returns:
            Defense config dictionary
        """
        return {
            'defense_type': 'coca',
            'system_prompt': self.system_prompt,
            'original_response': attack_response.get('content', ''),
            'sample': sample
        }
    
    def execute(
        self,
        client,
        sample: Dict[str, Any],
        attack_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute defense
        
        Args:
            client: LM Studio client
            sample: Sample from dataset
            attack_config: Optional attack config
            
        Returns:
            Defense result dictionary
        """
        # Determine image and prompt to use
        if attack_config:
            image = attack_config.get('image')
            prompt = attack_config.get('prompt', sample.get('question', ''))
        else:
            image = sample.get('image')
            prompt = sample.get('question', '')
        
        # Perform inference with safety system prompt
        response = client.chat(
            message=prompt,
            image=image,
            system_prompt=self.system_prompt
        )
        
        return {
            'success': response.get('success', False),
            'content': response.get('content', ''),
            'defense_applied': True,
            'defense_type': 'coca',
            'attack_config': attack_config,
            'raw_response': response
        }
    
    def execute_with_comparison(
        self,
        client,
        sample: Dict[str, Any],
        attack_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute defense with comparison (before/after)
        
        Args:
            client: LM Studio client
            sample: Sample from dataset
            attack_config: Optional attack config
            
        Returns:
            Dictionary containing comparison results
        """
        # Determine image and prompt
        if attack_config:
            image = attack_config.get('image')
            prompt = attack_config.get('prompt', sample.get('question', ''))
        else:
            image = sample.get('image')
            prompt = sample.get('question', '')
        
        # 1. Original response without defense
        original_response = client.chat(
            message=prompt,
            image=image
        )
        
        # 2. Defended response
        defended_response = client.chat(
            message=prompt,
            image=image,
            system_prompt=self.system_prompt
        )
        
        return {
            'original_content': original_response.get('content', ''),
            'defended_content': defended_response.get('content', ''),
            'original_success': original_response.get('success', False),
            'defended_success': defended_response.get('success', False),
            'defense_type': 'coca_comparison'
        }
    
    def batch_execute(
        self,
        client,
        samples: List[Dict[str, Any]],
        attack_results: Optional[List[Dict[str, Any]]] = None,
        existing_results: Optional[List[Dict[str, Any]]] = None,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Execute defense in batch
        
        Args:
            client: LM Studio client
            samples: List of samples
            attack_results: Optional attack results list
            
        Returns:
            List of defense results
        """
        results = list(existing_results or [])
        start_index = len(results)
        if start_index:
            print(f"  [RESUME] Reusing {start_index}/{len(samples)} completed CoCA defense samples")

        for i, sample in enumerate(samples[start_index:], start=start_index):
            print(f"  [{i+1}/{len(samples)}] Executing CoCA defense...")
            
            attack_config = None
            if attack_results and i < len(attack_results):
                attack_config = attack_results[i].get('attack_config')
            
            result = self.execute(client, sample, attack_config)
            results.append(result)
            if progress_callback is not None:
                progress_callback(results)
        
        return results


# Helper function
def create_coca_defense(
    system_prompt: str = "You are a helpful assistant that must prioritize safety."
) -> CoCADefense:
    """Create CoCA defense instance"""
    return CoCADefense(system_prompt)
