# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import atexit
import base64
import logging
import socket
import time
from io import BytesIO
from typing import Optional, Union
from urllib.parse import urlparse

import torch
from torch import nn
from transformers import is_torch_xpu_available

from ..import_utils import is_aiohttp_available, is_vllm_ascend_available, is_vllm_available


if is_aiohttp_available():
    import aiohttp
    from aiohttp import ClientError
else:
    aiohttp = None
    ClientError = Exception


if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator


logger = logging.get_logger(__name__)


class AsyncVLLMClient:
    """
    An async client class to interact with an async vLLM server.

    This class provides async methods to generate completions, initialize and manage weight update groups, and update
    model weights in a distributed setting. Before using it, start the async vLLM server with `trl vllm-serve-async`.

    Args:
        base_url (`str`, *optional*):
            Base URL for the async vLLM server (e.g., `"http://localhost:8001"`). If provided, `host` and `server_port`
            are ignored.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the async vLLM server. Ignored if `base_url` is provided.
        server_port (`int`, *optional*, defaults to `8001`):
            Port number of the async vLLM server. Ignored if `base_url` is provided.
        group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group.
        connection_timeout (`float`, *optional*, defaults to `0.0`):
            Total timeout duration in seconds to wait for the server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

    Examples:
        Run the async vLLM server with the model `Qwen/Qwen2.5-7B`:

        ```
        $ trl vllm-serve-async --model Qwen/Qwen2.5-7B --port 8001
        ...
        INFO:     Application startup complete.
        INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
        ```

        Use the async client to generate completions:

        ```python
        >>> import asyncio
        >>> from trl.extras.async_vllm_client import AsyncVLLMClient

        >>> async def main():
        ...     client = AsyncVLLMClient()
        ...     result = await client.generate_async(["Hello, AI!", "Tell me a joke"])
        ...     print(result)
        ...     await client.close()

        >>> asyncio.run(main())
        ```

        Concurrent async generations:

        ```python
        >>> async def concurrent_example():
        ...     client = AsyncVLLMClient()
        ...     
        ...     tasks = [
        ...         client.generate_async(["Tell me a joke"], max_tokens=50),
        ...         client.generate_async(["What is AI?"], max_tokens=50),
        ...         client.generate_async(["Explain quantum physics"], max_tokens=50)
        ...     ]
        ...     
        ...     results = await asyncio.gather(*tasks)
        ...     await client.close()
        ...     return results
        ```
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        host: str = "0.0.0.0",
        server_port: int = 8001,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):
        if not is_aiohttp_available():
            raise ImportError("aiohttp is not installed. Please install it with `pip install aiohttp`.")
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install it with `pip install trl[vllm]`.")

        self.session = None

        if base_url is not None:
            # Parse the base_url to extract host and port
            parsed_url = urlparse(base_url)
            self.host = socket.gethostbyname(parsed_url.hostname)
            scheme = parsed_url.scheme or "http"
            self.base_url = f"{scheme}://{parsed_url.netloc}{parsed_url.path}"
        else:
            self.host = host
            self.server_port = server_port
            self.base_url = f"http://{self.host}:{self.server_port}"
        self.group_port = group_port
        
        # We'll check server availability when first method is called

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def check_server_async(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check server availability asynchronously with retries on failure, within a total timeout duration.
        If the server is not up after the total timeout duration, raise a `ConnectionError`.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        session = await self._get_session()
        url = f"{self.base_url}/health/"
        start_time = time.time()

        while True:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return
            except Exception as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if total_timeout > 0 and elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The async vLLM server can't be reached at {self.base_url} after {total_timeout} seconds. "
                        "Make sure the server is running by running `trl vllm-serve-async`."
                    ) from exc

            # Retry logic: wait before trying again
            if total_timeout > 0:
                logger.info(f"Server is not up yet. Retrying in {retry_interval} seconds...")
            await asyncio.sleep(retry_interval)

    async def generate_async(
        self,
        prompts: list[str],
        images: Optional[list] = None,
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
        generation_kwargs: Optional[dict] = None,
    ) -> dict:
        """
        Generates model completions for the provided prompts asynchronously.

        Args:
            prompts (`list[str]`):
                List of text prompts for which the model will generate completions.
            images (`list[PIL.Image]`, *optional*):
                List of PIL Images to send along with the prompts.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. `-1` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            guided_decoding_regex (`str`, *optional*):
                Regular expression to guide the decoding process.
            generation_kwargs (`dict`, *optional*):
                Additional generation parameters to pass to the vLLM `SamplingParams`. This can include parameters like
                `seed`, `frequency_penalty`, etc. If it contains keys that conflict with the other parameters, they
                will override them.

        Returns:
            `dict` with keys:
                - `completion_ids` (`list[list[int]]`):
                    List of lists of token IDs representing the model-generated completions for each prompt.
                - `logprobs` (`list[list[float]]`):
                    List of lists of log probabilities for each generated token.
        """
        session = await self._get_session()
        url = f"{self.base_url}/generate_async/"

        def pil_to_base64(image):
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            return base64.b64encode(img_bytes).decode("utf-8")

        # Convert PIL images to base64 strings
        images_b64 = [pil_to_base64(img) for img in images] if images else None

        payload = {
            "prompts": prompts,
            "images": images_b64,
            "n": n,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "guided_decoding_regex": guided_decoding_regex,
            "generation_kwargs": generation_kwargs or {},
        }

        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "completion_ids": result["completion_ids"],
                    "logprobs": result["logprobs"]
                }
            else:
                error_text = await response.text()
                raise Exception(f"Request failed: {response.status}, {error_text}")

    async def init_communicator_async(self, device: Union[torch.device, str, int] = 0):
        """
        Initializes the weight update group in a distributed setup for model synchronization asynchronously.

        Args:
            device (`torch.device`, `str`, or `int`, *optional*, defaults to `0`):
                Device of trainer main process. It's the device that will be used for the weights synchronization. Can
                be a `torch.device` object, a string like `'cuda:0'`, or an integer device index.
        """
        session = await self._get_session()
        
        # Get the world size from the server
        url = f"{self.base_url}/get_world_size/"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                vllm_world_size = data["world_size"]
            else:
                error_text = await response.text()
                raise Exception(f"Request failed: {response.status}, {error_text}")

        world_size = vllm_world_size + 1  # add the client to the world
        self.rank = vllm_world_size  # the client's rank is the last process

        # Initialize weight update group
        url = f"{self.base_url}/init_communicator/"
        # Will simplify it after torch xpu 2.9 support get uuid.
        if is_torch_xpu_available():
            if hasattr(torch.xpu.get_device_properties(device), "uuid"):
                client_device_uuid = str(torch.xpu.get_device_properties(device).uuid)
            else:
                client_device_uuid = "42"
        else:
            client_device_uuid = str(torch.cuda.get_device_properties(device).uuid)

        payload = {
            "host": "0.0.0.0",
            "port": self.group_port,
            "world_size": world_size,
            "client_device_uuid": client_device_uuid,
        }

        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Request failed: {response.status}, {error_text}")

        # Brief delay to allow server initialization
        await asyncio.sleep(0.1)

        # Set up the communication group for weight broadcasting
        if is_torch_xpu_available():
            store = torch.distributed.TCPStore(
                host_name=self.host, port=self.group_port, world_size=world_size, is_master=(self.rank == 0)
            )
            prefixed_store = torch.distributed.PrefixStore("client2server", store)
            pg = torch.distributed.ProcessGroupXCCL(
                store=prefixed_store,
                rank=self.rank,
                size=world_size,
            )
            self.communicator = pg
        else:
            pg = StatelessProcessGroup.create(
                host=self.host, port=self.group_port, rank=self.rank, world_size=world_size
            )
            self.communicator = PyNcclCommunicator(pg, device=device)

        # When the client object is deleted, close the weight update group
        atexit.register(self.close_communicator_sync)

    async def update_named_param_async(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes asynchronously.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        session = await self._get_session()
        
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"{self.base_url}/update_named_param/"
        payload = {"name": name, "dtype": dtype, "shape": list(shape)}
        
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Request failed: {response.status}, {error_text}")

        if is_torch_xpu_available():
            # Use XCCL to broadcast the updated weights from the client (src) to all workers.
            self.communicator.broadcast(weights, root=self.rank)
            self.communicator.barrier()
        else:
            # Use NCCL to broadcast the updated weights from the client (src) to all workers.
            self.communicator.broadcast(weights, src=self.rank)
            self.communicator.group.barrier()

    async def update_model_params_async(self, model: nn.Module):
        """
        Updates all parameters of the given model by calling `update_named_param_async` for each parameter in the model.

        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
        """
        for name, param in model.named_parameters():
            # Update each parameter individually
            await self.update_named_param_async(name, param.data)

    async def reset_prefix_cache_async(self):
        """
        Resets the prefix cache for the model asynchronously.
        """
        session = await self._get_session()
        url = f"{self.base_url}/reset_prefix_cache/"
        
        async with session.post(url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Request failed: {response.status}, {error_text}")

    async def close_communicator_async(self):
        """
        Closes the weight update group and cleans up the communication group asynchronously.
        """
        if self.session is None:
            return
            
        url = f"{self.base_url}/close_communicator/"

        try:
            async with self.session.post(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Request failed: {response.status}, {error_text}")
        except Exception:
            # The server might be already down, so we don't need to close the communicator
            pass

    def close_communicator_sync(self):
        """
        Synchronous version of close_communicator for atexit registration.
        """
        if hasattr(self, 'communicator'):
            try:
                # Try to run the async version if we're in an async context
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, we can't use run_until_complete
                    # Just cleanup the communicator directly
                    if hasattr(self, 'communicator'):
                        del self.communicator
                else:
                    loop.run_until_complete(self.close_communicator_async())
            except Exception:
                # Fallback: just cleanup the communicator
                if hasattr(self, 'communicator'):
                    del self.communicator

    async def close(self):
        """
        Close the async client and clean up resources.
        """
        await self.close_communicator_async()
        
        if self.session:
            await self.session.close()
            self.session = None


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        client = AsyncVLLMClient()
        
        # Check server availability
        await client.check_server_async(total_timeout=10)
        
        # Single async generation
        result = await client.generate_async(["Hello, world!"], max_tokens=32)
        print("Single generation:", result)
        
        # Concurrent async generations
        tasks = [
            client.generate_async(["Tell me a joke"], max_tokens=50),
            client.generate_async(["What is AI?"], max_tokens=50),
            client.generate_async(["Explain quantum physics"], max_tokens=50)
        ]
        
        concurrent_results = await asyncio.gather(*tasks)
        print("Concurrent generations:", len(concurrent_results), "results")
        
        await client.close()

    asyncio.run(main())