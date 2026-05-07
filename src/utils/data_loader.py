"""
Data Loader
Load MM-SafetyBench dataset from HuggingFace
"""

from typing import Optional, List, Dict, Any
import time
from datasets import load_dataset
from PIL import Image


class MMSafetyBenchLoader:
    """MM-SafetyBench Dataset Loader"""
    
    # Available configs (13 high-risk scenarios)
    AVAILABLE_CONFIGS = [
        'Illegal_Activitiy',  # Note: Original typo preserved
        'HateSpeech',
        'Privacy_Violence',
        'Malware_Generation',
        'Physical_Harm',
        'Financial_Advice',
        'Fraud',
        'Gov_Decision',
        'Health_Consultation',
        'Legal_Opinion',
        'Political_Lobbying',
        'Sex',
        'EconomicHarm'
    ]
    
    # Available splits in the dataset
    AVAILABLE_SPLITS = ['SD', 'SD_TYPO', 'TYPO', 'Text_only']
    
    def __init__(
        self,
        dataset_name: str = "PKU-Alignment/MM-SafetyBench",
        config: Optional[str] = "Illegal_Activitiy",
        split: str = "TYPO",
        seed: int = 42,      # Fixed random seed for reproducibility
        shuffle: bool = False  # Whether to shuffle samples
    ):
        """
        Initialize data loader
        
        Args:
            dataset_name: HuggingFace dataset name
            config: Config name (must be one of AVAILABLE_CONFIGS)
            split: Data split (SD, SD_TYPO, TYPO, Text_only)
            seed: Random seed for reproducibility (ensures paired samples)
            shuffle: Whether to shuffle samples
        """
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.seed = seed
        self.shuffle = shuffle
        self.dataset = None
        self._loaded = False
    
    def load(self, streaming: bool = False) -> None:
        """
        Load dataset
        
        Args:
            streaming: Whether to use streaming mode (saves memory)
        """
        print(f"Loading dataset: {self.dataset_name}/{self.config} (split: {self.split}, seed: {self.seed})...")
        last_error = None
        for attempt in range(1, 4):
            try:
                self.dataset = load_dataset(
                    self.dataset_name,
                    self.config,
                    split=self.split,
                    streaming=streaming
                )
                self._loaded = True
                print(f"[OK] Dataset loaded")
                return
            except Exception as e:
                last_error = e
                print(f"[WARN] Dataset load failed (attempt {attempt}/3): {e}")
                if attempt < 3:
                    time.sleep(2 * attempt)

        raise last_error
    
    def get_tiny_dataset(self, size: int = 50) -> List[Dict[str, Any]]:
        """
        Get Tiny Version subset
        
        Critical: Uses fixed seed to ensure paired samples across different models
        
        Args:
            size: Number of samples
            
        Returns:
            List of samples (same for all models if seed is fixed)
        """
        if not self._loaded:
            self.load()
        
        samples = []
        count = 0
        
        # Use fixed seed for reproducibility
        if self.shuffle:
            import random
            random.seed(self.seed)
            dataset_list = list(self.dataset)
            random.shuffle(dataset_list)
            iterator = iter(dataset_list)
        else:
            # No shuffle, take first N samples (ensures perfect pairing)
            iterator = iter(self.dataset)
        
        for sample in iterator:
            if count >= size:
                break
            try:
                parsed = self._parse_sample(sample)
                samples.append(parsed)
                count += 1
            except Exception as e:
                continue
        
        print(f"[OK] Got {len(samples)} samples (seed={self.seed}, shuffle={self.shuffle})")
        return samples
    
    def get_by_config(
        self,
        config: str,
        split: str = "TYPO",
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get samples by config
        
        Args:
            config: Config name
            split: Data split to use
            max_samples: Maximum number of samples
            
        Returns:
            List of samples
        """
        # Switch config and split
        self.config = config
        self.split = split
        self._loaded = False
        self.load()
        
        return self.get_tiny_dataset(max_samples or 50)

    def get_scenario_sweep(
        self,
        configs: Optional[List[str]] = None,
        split: str = "TYPO",
        samples_per_config: int = 50,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load paired subsets for multiple risk scenarios.

        Args:
            configs: Scenario configs to load. Defaults to all available configs.
            split: Dataset split to use for every scenario.
            samples_per_config: Number of paired samples per scenario.

        Returns:
            Mapping from scenario config name to parsed samples.
        """
        selected_configs = configs or self.AVAILABLE_CONFIGS
        scenario_samples = {}

        for config in selected_configs:
            scenario_samples[config] = self.get_by_config(
                config=config,
                split=split,
                max_samples=samples_per_config,
            )

        return scenario_samples

    def get_flat_scenario_sweep(
        self,
        configs: Optional[List[str]] = None,
        split: str = "TYPO",
        samples_per_config: int = 50,
    ) -> List[Dict[str, Any]]:
        """Load multiple scenarios and return one flat sample list."""
        grouped = self.get_scenario_sweep(configs, split, samples_per_config)
        samples = []
        for scenario, scenario_samples in grouped.items():
            for sample_idx, sample in enumerate(scenario_samples):
                sample['scenario_sample_id'] = sample_idx
                sample['scenario'] = scenario
                samples.append(sample)
        return samples
    
    def _parse_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse raw sample to unified format
        
        Args:
            sample: Raw sample dictionary
            
        Returns:
            Standardized sample
        """
        # Map split to scenario type
        scenario_mapping = {
            'SD': 'semantic_reconstruction',
            'SD_TYPO': 'scene_consistent',
            'TYPO': 'typography',
            'Text_only': 'text_only'
        }
        
        # Handle both dict-like and column-like access
        def get_value(key):
            if isinstance(sample, dict):
                return sample.get(key)
            else:
                try:
                    return sample[key]
                except:
                    return None
        
        parsed = {
            'question': get_value('question') or get_value('Question') or '',
            'scenario': self.config,
            'split_type': self.split,
            'attack_type': scenario_mapping.get(self.split, 'unknown'),
            'image': get_value('Image') or get_value('image'),
            'image_typo': None,
            'image_sd': None,
            'image_sd_typo': None,
            'answer': None,
            'is_safe': None,
        }
        
        # Set specific image types based on split
        if self.split == 'TYPO':
            parsed['image_typo'] = parsed['image']
        elif self.split == 'SD':
            parsed['image_sd'] = parsed['image']
        elif self.split == 'SD_TYPO':
            parsed['image_sd_typo'] = parsed['image']
        
        return parsed
    
    def __len__(self) -> int:
        """Get dataset size"""
        if not self._loaded:
            return 0
        try:
            return len(self.dataset)
        except:
            return -1  # Cannot get length in streaming mode


# Helper function
def load_mm_safety_bench(
    tiny: bool = True,
    size: int = 50,
    config: str = "Illegal_Activitiy",
    split: str = "TYPO",
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Helper function: Load MM-SafetyBench
    
    Args:
        tiny: Whether to use Tiny Version
        size: Tiny Version sample count
        config: Config name
        split: Data split
        seed: Random seed for reproducibility
        
    Returns:
        List of samples
    """
    loader = MMSafetyBenchLoader(config=config, split=split, seed=seed)
    
    if tiny:
        return loader.get_tiny_dataset(size)
    else:
        loader.load()
        return loader.dataset
