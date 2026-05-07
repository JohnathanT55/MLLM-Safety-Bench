"""
LM Studio API Client
Supports loading, switching, and chatting with multimodal models
"""

import base64
import io
import requests
from typing import Optional, List, Dict, Any
from PIL import Image
import yaml


class LMStudioClient:
    """LM Studio REST API Client"""
    
    def __init__(self, config_path: str = "configs/models.yaml"):
        """
        Initialize client
        
        Args:
            config_path: Path to model configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.base_url = self.config['lmstudio']['base_url']
        self.api_key = self.config['lmstudio']['api_key']
        self.timeout = self.config['lmstudio']['timeout']
        self.current_model = None
        
        # Default inference parameters
        self.inference_config = self.config.get('inference', {})
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def list_models(self) -> List[str]:
        """
        List available models in LM Studio
        
        Returns:
            List of model IDs
        """
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return [model['id'] for model in data.get('data', [])]
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to get model list: {e}")
            return []

    def get_model_state(self, model_id: str) -> Optional[str]:
        """Get model load state from LM Studio management API."""
        management_url = self.base_url.replace('/v1', '/api/v0')
        try:
            response = requests.get(
                f"{management_url}/models/{model_id}",
                headers=self._get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get('state')
        except requests.exceptions.RequestException:
            return None
    
    def load_model(self, model_key: str) -> bool:
        """
        Load/switch to specified model
        
        Args:
            model_key: Model key from config file (e.g., 'qwen3_5_9b')
            
        Returns:
            Whether loading was successful
        """
        if model_key not in self.config['models']:
            print(f"[ERROR] Unknown model key: {model_key}")
            return False

        if self.current_model == model_key:
            return True
        
        model_config = self.config['models'][model_key]
        model_id = model_config['model_id']
        
        state = self.get_model_state(model_id)
        if state == 'loaded':
            self.current_model = model_key
            print(f"[OK] Model loaded: {model_config['name']}")
            return True

        load_url = self.base_url.replace('/v1', '/api/v1') + '/models/load'
        payload = {"model": model_id}
        if model_config.get('context_length'):
            payload['context_length'] = model_config['context_length']

        try:
            response = requests.post(
                load_url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            if data.get('status') == 'loaded':
                self.current_model = model_key
                print(f"[OK] Model loaded: {model_config['name']}")
                return True

            if data.get('error'):
                print(f"[ERROR] LM Studio load failed: {data['error']}")
            else:
                print(f"[ERROR] Unexpected LM Studio load response: {data}")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to load model via LM Studio API: {e}")

        fallback_state = self.get_model_state(model_id)
        if fallback_state is None:
            print(f"[ERROR] Could not query LM Studio model state for {model_id}")
        else:
            print(f"[ERROR] Model not loaded in LM Studio: {model_id} (state={fallback_state})")

        print("   Load the model manually in LM Studio, then rerun or resume the benchmark.")
        return False
    
    def image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to Base64 encoding
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded string (without prefix)
        """
        buffer = io.BytesIO()
        # Convert to RGB mode (handle RGBA, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def chat(
        self,
        message: str,
        image: Optional[Image.Image] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send multimodal chat request
        
        Args:
            message: Text message
            image: Optional PIL Image object
            system_prompt: Optional system prompt
            **kwargs: Override default inference parameters
            
        Returns:
            Dictionary containing response text and metadata
        """
        if not self.current_model:
            raise ValueError("Model not loaded, please call load_model() first")
        
        model_config = self.config['models'][self.current_model]
        model_id = model_config['model_id']
        
        # Build message content
        content = []
        
        # If image exists, add to content
        if image is not None:
            if not model_config.get('vision_enabled', False):
                print("[WARN] Current model does not support visual input")
            else:
                base64_image = self.image_to_base64(image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
        
        # Add text message
        content.append({
            "type": "text",
            "text": message
        })
        
        # Build request body
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        # Merge inference parameters
        inference_params = {**self.inference_config, **kwargs}
        
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": inference_params.get('max_tokens', 512),
            "temperature": inference_params.get('temperature', 0.7),
            "top_p": inference_params.get('top_p', 0.9),
        }
        
        # Add optional parameters
        if inference_params.get('stop'):
            payload['stop'] = inference_params['stop']
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            choice = result['choices'][0]
            message_response = choice['message']
            
            return {
                "success": True,
                "content": message_response.get('content', ''),
                "finish_reason": choice.get('finish_reason', 'unknown'),
                "usage": result.get('usage', {}),
                "raw_response": result
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "content": None
            }
    
    def chat_text_only(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Text-only chat (no vision)
        
        Args:
            message: Text message
            system_prompt: Optional system prompt
            **kwargs: Inference parameters
            
        Returns:
            Response dictionary
        """
        return self.chat(message, image=None, system_prompt=system_prompt, **kwargs)
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current model info
        
        Returns:
            Model config dictionary
        """
        if not self.current_model:
            return None
        return self.config['models'][self.current_model]
    
    def test_connection(self) -> bool:
        """
        Test connection to LM Studio
        
        Returns:
            Whether connection was successful
        """
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._get_headers(),
                timeout=10
            )
            return response.status_code == 200
        except:
            return False


# Helper function
def create_client(config_path: str = "configs/models.yaml") -> LMStudioClient:
    """Create LM Studio client instance"""
    return LMStudioClient(config_path)
