#!/usr/bin/env python3
"""
Test script for model mapping functionality in /v1/messages endpoint
Tests the models.json based configuration system
"""

import json
import requests
from typing import Dict, Any

# Test configurations
BASE_URL = "http://localhost:8000"
API_KEY = "sk-your-custom-key-here"

def load_model_mappings():
    """Load model mappings from models.json"""
    try:
        with open("models.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        if isinstance(config, dict) and "anthropic_model_mappings" in config:
            return config["anthropic_model_mappings"]
        else:
            print("No model mappings found in models.json")
            return {}
    except Exception as e:
        print(f"Error loading models.json: {e}")
        return {}

def test_model_mapping():
    """Test various model names to verify mapping works"""
    
    # Load actual mappings from models.json
    mappings = load_model_mappings()
    print(f"Loaded {len(mappings)} model mappings from models.json:")
    for key, value in mappings.items():
        print(f"  {key} -> {value}")
    
    if not mappings:
        print("No mappings to test. Please configure models.json first.")
        return
    
    # Test some of the configured mappings
    test_cases = list(mappings.items())[:10]  # Test first 10 mappings
    
    # Add some unmapped models for testing
    test_cases.extend([
        ("anthropic-claude-3.5-sonnet", "anthropic-claude-3.5-sonnet"),  # Should remain unchanged
        ("unknown-model", "unknown-model"),  # Should remain unchanged
    ])
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json",
        "x-anthropic-version": "2023-06-01"
    }
    
    for input_model, expected_model in test_cases:
        print(f"\nTesting: {input_model} -> {expected_model}")
        
        payload = {
            "model": input_model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/v1/messages",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"✓ Success: {input_model} accepted")
            elif response.status_code == 404:
                print(f"✗ Model not found: {input_model} -> {expected_model}")
                print(f"  Response: {response.text}")
            else:
                print(f"✗ Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"✗ Connection error: Server not running at {BASE_URL}")
            break
        except Exception as e:
            print(f"✗ Unexpected error: {e}")

def test_models_endpoint():
    """Test the /v1/models endpoint to see available models"""
    print("\nTesting /v1/models endpoint:")
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(f"{BASE_URL}/v1/models", headers=headers)
        if response.status_code == 200:
            models = response.json()
            print("Available models:")
            for model in models.get("data", []):
                print(f"  - {model.get('id', 'Unknown')}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"Connection error: Server not running at {BASE_URL}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def test_config_format():
    """Test models.json configuration format"""
    print("\nTesting models.json configuration:")
    
    try:
        with open("models.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        print("✓ models.json is valid JSON")
        
        if isinstance(config, dict):
            if "models" in config:
                print(f"✓ Found 'models' section with {len(config['models'])} models")
            else:
                print("✗ Missing 'models' section")
            
            if "anthropic_model_mappings" in config:
                print(f"✓ Found 'anthropic_model_mappings' section with {len(config['anthropic_model_mappings'])} mappings")
            else:
                print("✗ Missing 'anthropic_model_mappings' section")
                
        elif isinstance(config, list):
            print("⚠ Using legacy format (array of model names)")
            print("Consider upgrading to new format with mapping support")
        else:
            print("✗ Invalid config format")
            
    except FileNotFoundError:
        print("✗ models.json not found")
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in models.json: {e}")
    except Exception as e:
        print(f"✗ Error reading models.json: {e}")

if __name__ == "__main__":
    print("Model Mapping Test Script (models.json based)")
    print("=" * 60)
    
    print("\n1. Testing models.json configuration format:")
    test_config_format()
    
    print("\n2. Testing available models:")
    test_models_endpoint()
    
    print("\n3. Testing model mapping functionality:")
    test_model_mapping()
    
    print("\n" + "=" * 60)
    print("Test completed!")