#!/usr/bin/env python3
"""
Example script demonstrating concurrent async VLLM client usage.

This script shows how to:
1. Connect to an async VLLM server
2. Send multiple concurrent generation requests
3. Measure performance differences between concurrent and sequential requests

Usage:
    # First, start the async VLLM server:
    trl vllm-serve-async --model microsoft/DialoGPT-medium --port 8001

    # Then run this example:
    python examples/async_vllm_client_example.py
"""

import asyncio
import time
from typing import List

from trl.extras.async_vllm_client import AsyncVLLMClient


async def single_request_example(client: AsyncVLLMClient):
    """Example of a single async generation request."""
    print("🔍 Single request example:")
    
    start_time = time.time()
    result = await client.generate_async(
        prompts=["Hello, how are you today?"],
        max_tokens=50,
        temperature=0.7
    )
    end_time = time.time()
    
    print(f"   - Generated {len(result['completion_ids'])} completion(s)")
    print(f"   - Time taken: {end_time - start_time:.2f} seconds")
    print(f"   - First completion tokens: {result['completion_ids'][0][:10]}...")
    print()


async def concurrent_requests_example(client: AsyncVLLMClient):
    """Example of concurrent async generation requests."""
    print("🚀 Concurrent requests example:")
    
    prompts_list = [
        ["Tell me a joke about programming"],
        ["What is artificial intelligence?"],
        ["Explain quantum computing in simple terms"],
        ["What are the benefits of renewable energy?"],
        ["How does machine learning work?"]
    ]
    
    start_time = time.time()
    
    # Create multiple concurrent tasks
    tasks = [
        client.generate_async(
            prompts=prompts,
            max_tokens=80,
            temperature=0.8
        )
        for prompts in prompts_list
    ]
    
    # Wait for all tasks to complete concurrently
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    print(f"   - Generated {len(results)} concurrent requests")
    print(f"   - Total time taken: {end_time - start_time:.2f} seconds")
    print(f"   - Average time per request: {(end_time - start_time) / len(results):.2f} seconds")
    
    # Show sample results
    for i, result in enumerate(results):
        print(f"   - Request {i+1} tokens: {len(result['completion_ids'][0])} tokens")
    print()


async def sequential_requests_example(client: AsyncVLLMClient):
    """Example of sequential async generation requests for comparison."""
    print("🐌 Sequential requests example (for comparison):")
    
    prompts_list = [
        ["Tell me a joke about programming"],
        ["What is artificial intelligence?"],
        ["Explain quantum computing in simple terms"],
        ["What are the benefits of renewable energy?"],
        ["How does machine learning work?"]
    ]
    
    start_time = time.time()
    
    results = []
    for prompts in prompts_list:
        result = await client.generate_async(
            prompts=prompts,
            max_tokens=80,
            temperature=0.8
        )
        results.append(result)
    
    end_time = time.time()
    
    print(f"   - Generated {len(results)} sequential requests")
    print(f"   - Total time taken: {end_time - start_time:.2f} seconds")
    print(f"   - Average time per request: {(end_time - start_time) / len(results):.2f} seconds")
    print()


async def batch_requests_example(client: AsyncVLLMClient):
    """Example of batch processing multiple prompts in a single request."""
    print("📦 Batch requests example:")
    
    prompts = [
        "Tell me a joke about programming",
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms",
        "What are the benefits of renewable energy?",
        "How does machine learning work?"
    ]
    
    start_time = time.time()
    
    # Send all prompts in a single batch request
    result = await client.generate_async(
        prompts=prompts,
        max_tokens=80,
        temperature=0.8
    )
    
    end_time = time.time()
    
    print(f"   - Generated {len(result['completion_ids'])} completions in single batch")
    print(f"   - Total time taken: {end_time - start_time:.2f} seconds")
    print(f"   - Average time per completion: {(end_time - start_time) / len(result['completion_ids']):.2f} seconds")
    print()


async def stress_test_example(client: AsyncVLLMClient, num_concurrent_requests: int = 10):
    """Example of stress testing with many concurrent requests."""
    print(f"⚡ Stress test example ({num_concurrent_requests} concurrent requests):")
    
    start_time = time.time()
    
    # Create many concurrent tasks
    tasks = [
        client.generate_async(
            prompts=[f"Generate a creative story about request {i}"],
            max_tokens=100,
            temperature=0.9
        )
        for i in range(num_concurrent_requests)
    ]
    
    # Wait for all tasks to complete concurrently
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    print(f"   - Completed {len(results)} concurrent requests")
    print(f"   - Total time taken: {end_time - start_time:.2f} seconds")
    print(f"   - Requests per second: {len(results) / (end_time - start_time):.2f}")
    print(f"   - Average time per request: {(end_time - start_time) / len(results):.2f} seconds")
    print()


async def main():
    """Main example function demonstrating various async client capabilities."""
    print("🤖 Async VLLM Client Examples")
    print("=" * 50)
    
    # Initialize the async client
    client = AsyncVLLMClient(base_url="http://localhost:8001")
    
    try:
        # Check if server is available
        print("🔗 Connecting to async VLLM server...")
        await client.check_server_async(total_timeout=5.0)
        print("✅ Connected to server successfully!\n")
        
        # Run various examples
        await single_request_example(client)
        await concurrent_requests_example(client)
        await sequential_requests_example(client)
        await batch_requests_example(client)
        await stress_test_example(client, num_concurrent_requests=8)
        
        print("🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure the async VLLM server is running:")
        print("   trl vllm-serve-async --model microsoft/DialoGPT-medium --port 8001")
        
    finally:
        # Clean up
        await client.close()


if __name__ == "__main__":
    # Run the async example
    asyncio.run(main())