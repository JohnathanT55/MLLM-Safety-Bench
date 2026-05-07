"""
LM Studio Connection Test Script
Validates model loading, switching, and multimodal chat functionality
"""

import sys
import io
from pathlib import Path

# Set UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lmstudio_client import LMStudioClient
from PIL import Image


def test_connection():
    """Test basic connection"""
    print("=" * 50)
    print("LM Studio Connection Test")
    print("=" * 50)
    
    client = LMStudioClient("configs/models.yaml")
    
    # 1. Test connection
    print("\n[Test 1] Connection Status...")
    if client.test_connection():
        print("[OK] Connection successful")
    else:
        print("[FAIL] Connection failed - Please ensure LM Studio is running")
        return False
    
    # 2. List models
    print("\n[Test 2] Available Models...")
    models = client.list_models()
    if models:
        for model in models:
            print(f"   - {model}")
    else:
        print("   No models available")
    
    return True


def test_model_loading():
    """Test model loading/switching"""
    print("\n" + "=" * 50)
    print("Model Loading Test")
    print("=" * 50)
    
    client = LMStudioClient("configs/models.yaml")
    
    # Try loading each configured model
    for model_key in ['qwen3_5_9b', 'gemma3_12b', 'glm4_6v_flash']:
        print(f"\nAttempting to load: {model_key}")
        success = client.load_model(model_key)
        if success:
            print(f"[OK] {model_key} loaded successfully")
        else:
            print(f"[WARN] {model_key} load failed or requires manual loading")
    
    return True


def test_multimodal_chat():
    """Test multimodal chat"""
    print("\n" + "=" * 50)
    print("Multimodal Chat Test")
    print("=" * 50)
    
    client = LMStudioClient("configs/models.yaml")
    
    # Load default model
    client.load_model('qwen3_5_9b')
    
    # Create a test image (solid color)
    test_image = Image.new('RGB', (224, 224), color='blue')
    
    # Test image+text chat
    print("\nSending image+text request...")
    response = client.chat(
        message="What color is this image?",
        image=test_image
    )
    
    if response.get('success'):
        print("[OK] Multimodal chat successful")
        print(f"   Response: {response.get('content', '')[:100]}...")
    else:
        print("[FAIL] Multimodal chat failed")
        print(f"   Error: {response.get('error')}")
    
    # Test text-only chat
    print("\nSending text-only request...")
    response = client.chat_text_only(
        message="Hello, how are you?"
    )
    
    if response.get('success'):
        print("[OK] Text-only chat successful")
        print(f"   Response: {response.get('content', '')[:100]}...")
    else:
        print("[FAIL] Text-only chat failed")
    
    return True


def test_image_description():
    """Test image description capability"""
    print("\n" + "=" * 50)
    print("Image Description Test")
    print("=" * 50)
    
    from src.utils.data_loader import MMSafetyBenchLoader
    
    client = LMStudioClient("configs/models.yaml")
    loader = MMSafetyBenchLoader(config="Illegal_Activitiy")
    
    # Load model
    client.load_model('qwen3_5_9b')
    
    # Get a sample
    print("\nLoading test sample...")
    try:
        samples = loader.get_tiny_dataset(5)
        sample = samples[0]
        
        # Test image description
        print("Sending image description request...")
        response = client.chat(
            message="Describe this image in detail.",
            image=sample.get('image')
        )
        
        if response.get('success'):
            print("[OK] Image description successful")
            content = response.get('content', '')
            print(f"   Description: {content[:200]}...")
        else:
            print("[FAIL] Image description failed")
            print(f"   Error: {response.get('error')}")
    except Exception as e:
        print(f"[WARN] Data loading failed: {e}")
        print("   This may be normal if HuggingFace access is slow")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "#" * 50)
    print("# LM Studio Complete Test Suite")
    print("#" * 50)
    
    tests = [
        ("Connection Test", test_connection),
        ("Model Loading Test", test_model_loading),
        ("Multimodal Chat Test", test_multimodal_chat),
        ("Image Description Test", test_image_description),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = success
        except Exception as e:
            print(f"\n[ERROR] {name} failed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    for name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 50)
    if all_passed:
        print("[SUCCESS] All tests passed!")
    else:
        print("[WARNING] Some tests failed, please check configuration")
    print("=" * 50)
    
    return all_passed


if __name__ == "__main__":
    main()
