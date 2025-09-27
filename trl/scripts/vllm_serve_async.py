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

import argparse
import asyncio
import base64
import logging
import math
import os
import time
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Optional

import torch
from transformers import is_torch_xpu_available, is_vision_available

from trl import TrlParser
from trl.import_utils import (
    is_fastapi_available,
    is_pydantic_available,
    is_uvicorn_available,
    is_vllm_ascend_available,
    is_vllm_available,
)


if is_fastapi_available():
    from fastapi import FastAPI


if is_pydantic_available():
    from pydantic import BaseModel


if is_uvicorn_available():
    import uvicorn


if is_vision_available():
    from PIL import Image


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.utils import get_open_port

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator

    # Try to import AsyncLLMEngine
    try:
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        ASYNC_ENGINE_AVAILABLE = True
    except ImportError:
        ASYNC_ENGINE_AVAILABLE = False


logger = logging.getLogger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class WeightSyncWorkerExtension:
    """
    A vLLM worker extension that enables weight synchronization between a client and multiple server workers.

    This worker uses a `StatelessProcessGroup` to establish communication and a `PyNcclCommunicator` or
    `ProcessGroupXCCL` to handle efficient GPU-based communication using NCCL. The primary purpose of this class is to
    receive updated model weights from a client process and distribute them to all worker processes participating in
    model inference.
    """

    # The following attributes are initialized when `init_communicator` method is called.
    communicator = None  # Communicator for weight updates
    client_rank = None  # Source rank for broadcasting updated weights

    def init_communicator(self, host: str, port: int, world_size: int, client_device_uuid: str) -> None:
        """
        Initializes the weight update communicator using a stateless process group.

        This method creates a `StatelessProcessGroup` that allows external training processes to communicate with vLLM
        workers without interfering with the global torch distributed group.

        Args:
            host (`str`):
                Hostname or IP address of the master node.
            port (`int`):
                Port number to be used for communication.
            world_size (`int`):
                Total number of participating processes in the update group.
            client_device_uuid (`str`):
                UUID of the device of client main process. Used to assert that devices are different from vllm workers
                devices.
        """
        if self.communicator is not None:
            return

        # TODO: will remove after torch xpu 2.9 support uuid in get_device_properties
        if torch.cuda.is_available() or (
            is_torch_xpu_available() and hasattr(torch.xpu.get_device_properties(self.device), "uuid")
        ):
            if is_torch_xpu_available():
                device_uuid = str(torch.xpu.get_device_properties(self.device).uuid)
            else:
                device_uuid = str(torch.cuda.get_device_properties(self.device).uuid)
            assert device_uuid != client_device_uuid, "vLLM workers and clients must use different devices."

        # Get the rank of the current worker in the global world group.
        rank = get_world_group().rank

        if is_torch_xpu_available():
            store = torch.distributed.TCPStore(
                host_name=host, port=port, world_size=world_size, is_master=(rank == 0)
            )
            prefixed_store = torch.distributed.PrefixStore("server2client", store)
            pg = torch.distributed.ProcessGroupXCCL(
                store=prefixed_store, rank=rank, size=world_size
            )
            self.communicator = pg
        else:
            pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
            self.communicator = PyNcclCommunicator(pg, device=self.device)

        # The client process that sends updated weights has the highest rank (world_size - 1).
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: str, shape: Sequence[int]) -> None:
        """
        Receives updated weights from the client process and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`str`):
                Data type of the weight tensor as a string (e.g., `"torch.float32"`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """
        if self.communicator is None:
            raise RuntimeError("Communicator not initialized. Call init_communicator first.")

        dtype = getattr(torch, dtype.split(".")[-1])
        # Allocate memory for the incoming weight tensor on the correct device.
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        if is_torch_xpu_available():
            self.communicator.broadcast(weight, root=self.client_rank)
            self.communicator.barrier()
        else:
            self.communicator.broadcast(weight, src=self.client_rank)
            self.communicator.group.barrier()

        # Load the received weights into the model.
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:
        """
        Closes the communicator when weight synchronization is no longer needed.

        This method deletes the NCCL communicator to release associated resources.
        """

        if self.communicator is not None:
            del self.communicator
            self.communicator = None


@dataclass
class ScriptArguments:
    r"""
    Arguments for the async vLLM server script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str`, *optional*):
            Revision to use for the model. If not specified, the default branch will be used.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        data_parallel_size (`int`, *optional*, defaults to `1`):
            Number of data parallel workers to use.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8001`):
            Port to run the server on.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        max_model_len (`int`, *optional*):
            If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        enable_prefix_caching (`bool`, *optional*):
            Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the hardware support
            this feature.
        enforce_eager (`bool`, *optional*, defaults to `False`):
            Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always execute the
            model in eager mode. If `False` (default behavior), we will use CUDA graph and eager execution in hybrid.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation to use for vLLM. Must be one of `"transformers"` or `"vllm"`. `"transformers"`: Use
            the `transformers` backend for model implementation. `"vllm"`: Use the `vllm` library for model
            implementation.
        kv_cache_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for KV cache. If set to `"auto"`, the dtype will default to the model data type.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to trust remote code when loading models. Set to `True` to allow executing code from model
            repositories. This is required for some custom models but introduces security risks.
        use_async_engine (`bool`, *optional*, defaults to `True`):
            Whether to use async engine if available. If `False`, will use sync engine wrapped in executor.
        log_level (`str`, *optional*, defaults to `"info"`):
            Log level for uvicorn. Possible choices: `"critical"`, `"error"`, `"warning"`, `"info"`, `"debug"`,
            `"trace"`.
    """

    model: str = field(
        metadata={"help": "Model name or path to load the model from."},
    )
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of data parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8001,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )
    enforce_eager: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always "
            "execute the model in eager mode. If `False` (default behavior), we will use CUDA graph and eager "
            "execution in hybrid."
        },
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={
            "help": "Model implementation to use for vLLM. Must be one of 'transformers' or 'vllm'. 'transformers': Use "
            "the transformers backend for model implementation. 'vllm': Use the vllm library for model implementation."
        },
    )
    kv_cache_dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for KV cache. If set to 'auto', the dtype will default to the model data type."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to trust remote code when loading models. Set to True to allow executing code from model "
            "repositories. This is required for some custom models but introduces security risks."
        },
    )
    use_async_engine: bool = field(
        default=True,
        metadata={
            "help": "Whether to use async engine if available. If False, will use sync engine wrapped in executor."
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Log level for uvicorn. Possible choices: 'critical', 'error', 'warning', 'info', 'debug', 'trace'."
        },
    )


def async_llm_worker(
    script_args: ScriptArguments, data_parallel_rank: int, master_port: int, connection: Connection
) -> None:
    # Set required environment variables for DP to work with vLLM
    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(script_args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    if script_args.use_async_engine and ASYNC_ENGINE_AVAILABLE:
        # Use async engine
        engine_args = AsyncEngineArgs(
            model=script_args.model,
            revision=script_args.revision,
            tensor_parallel_size=script_args.tensor_parallel_size,
            gpu_memory_utilization=script_args.gpu_memory_utilization,
            enforce_eager=script_args.enforce_eager,
            dtype=script_args.dtype,
            quantization=None,
            seed=0,
            enable_prefix_caching=script_args.enable_prefix_caching,
            kv_cache_dtype=script_args.kv_cache_dtype,
            max_model_len=script_args.max_model_len,
            worker_extension_cls="trl.scripts.vllm_serve_async.WeightSyncWorkerExtension",
            trust_remote_code=script_args.trust_remote_code,
        )
        llm = AsyncLLMEngine.from_engine_args(engine_args)
    else:
        # Use sync engine as fallback
        llm = LLM(
            model=script_args.model,
            revision=script_args.revision,
            tensor_parallel_size=script_args.tensor_parallel_size,
            gpu_memory_utilization=script_args.gpu_memory_utilization,
            enforce_eager=script_args.enforce_eager,
            dtype=script_args.dtype,
            quantization=None,
            seed=0,
            enable_prefix_caching=script_args.enable_prefix_caching,
            kv_cache_dtype=script_args.kv_cache_dtype,
            max_model_len=script_args.max_model_len,
            worker_extension_cls="trl.scripts.vllm_serve_async.WeightSyncWorkerExtension",
            trust_remote_code=script_args.trust_remote_code,
        )

    # Send ready signal to parent process
    connection.send({"status": "ready", "engine_type": "async" if hasattr(llm, "generate") else "sync"})

    while True:
        message = connection.recv()
        connection.send({"llm": llm})


def chunk_list(lst: list, n: int) -> list[list]:
    """
    Split list `lst` into `n` evenly distributed sublists.

    Example:
    ```python
    >>> chunk_list([1, 2, 3, 4, 5, 6], 2)
    [[1, 2, 3], [4, 5, 6]]

    >>> chunk_list([1, 2, 3, 4, 5, 6], 4)
    [[1, 2], [3, 4], [5], [6]]

    >>> chunk_list([1, 2, 3, 4, 5, 6], 8)
    [[1], [2], [3], [4], [5], [6], [], []]
    ```
    """
    k, r = divmod(len(lst), n)
    return [lst[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]


def sanitize_logprob(logprob):
    """Sanitize logprob values to handle NaN or infinite values."""

    value = logprob.logprob
    if math.isnan(value):
        return -float("inf")

    return value


async def main(script_args: ScriptArguments):
    if not is_fastapi_available():
        raise ImportError(
            "FastAPI is not installed. Please install it with `pip install fastapi` or `pip install trl[vllm]`."
        )

    if not is_pydantic_available():
        raise ImportError(
            "Pydantic is not installed. Please install it with `pip install pydantic` or `pip install trl[vllm]`."
        )

    if not is_uvicorn_available():
        raise ImportError(
            "Uvicorn is not installed. Please install it with `pip install uvicorn` or `pip install trl[vllm]`."
        )

    if not is_vllm_available():
        raise ImportError("vLLM is not installed. Please install it with `pip install trl[vllm]`.")

    # Spawn dp workers, and setup pipes for communication
    master_port = get_open_port()
    connections = []
    processes = []
    for data_parallel_rank in range(script_args.data_parallel_size):
        parent_conn, child_conn = Pipe()
        process = Process(target=async_llm_worker, args=(script_args, data_parallel_rank, master_port, child_conn))
        process.start()
        connections.append(parent_conn)
        processes.append(process)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Wait for all workers to be ready
        for connection in connections:
            response = connection.recv()
            print(f"Worker ready: {response}")

        # Get the LLM instances from workers
        for connection in connections:
            connection.send("get_llm")
        
        llms = []
        for connection in connections:
            response = connection.recv()
            llms.append(response["llm"])
        
        app.state.llms = llms
        app.state.is_async_engine = hasattr(llms[0], "generate") if llms else False
        
        yield
        
        # Cleanup
        for process in processes:
            process.terminate()
            process.join()

    app = FastAPI(lifespan=lifespan)

    # Define the endpoints for the model server
    @app.get("/health/")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/get_world_size/")
    async def get_world_size():
        """Get the world size of the vLLM workers."""
        total_world_size = 0
        for llm in app.state.llms:
            if hasattr(llm, 'engine'):
                # Async engine
                total_world_size += llm.engine.parallel_config.world_size
            else:
                # Sync engine
                total_world_size += llm.llm_engine.parallel_config.world_size
        return {"world_size": total_world_size}

    class AsyncGenerateRequest(BaseModel):
        prompts: list[str]
        images: Optional[list[str]] = None
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None
        generation_kwargs: dict = {}

    class AsyncGenerateResponse(BaseModel):
        completion_ids: list[list[int]]
        logprobs: list[list[float]]

    @app.post("/generate_async/", response_model=AsyncGenerateResponse)
    async def generate_async(request: AsyncGenerateRequest):
        """Generate completions asynchronously."""
        
        # Convert base64 images back to PIL if needed
        images = None
        if request.images:
            images = []
            for img_str in request.images:
                img_bytes = base64.b64decode(img_str)
                img = Image.open(BytesIO(img_bytes))
                images.append(img)

        # Setup sampling parameters
        guided_decoding = None
        if request.guided_decoding_regex:
            guided_decoding = GuidedDecodingParams(regex=request.guided_decoding_regex)

        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k > 0 else None,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            guided_decoding=guided_decoding,
            logprobs=1,  # Return logprobs for each token
            **request.generation_kwargs
        )

        # Split prompts across workers
        prompt_chunks = chunk_list(request.prompts, len(app.state.llms))
        
        # Generate using all workers concurrently
        tasks = []
        for i, (llm, prompts_chunk) in enumerate(zip(app.state.llms, prompt_chunks)):
            if not prompts_chunk:  # Skip empty chunks
                continue
                
            if app.state.is_async_engine:
                # Use async generation
                task = llm.generate(prompts_chunk, sampling_params)
            else:
                # Wrap sync generation in executor
                loop = asyncio.get_event_loop()
                task = loop.run_in_executor(None, llm.generate, prompts_chunk, sampling_params)
            
            tasks.append(task)

        # Wait for all generations to complete
        results = await asyncio.gather(*tasks)

        # Process results
        completion_ids = []
        logprobs = []
        
        for result in results:
            for request_output in result:
                for output in request_output.outputs:
                    completion_ids.append(output.token_ids)
                    # Extract logprobs for each token
                    token_logprobs = []
                    if output.logprobs:
                        for token_logprob in output.logprobs:
                            if token_logprob:
                                # Get the logprob for the selected token
                                token_logprobs.append(sanitize_logprob(next(iter(token_logprob.values()))))
                            else:
                                token_logprobs.append(0.0)
                    logprobs.append(token_logprobs)

        return AsyncGenerateResponse(
            completion_ids=completion_ids,
            logprobs=logprobs
        )

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int
        client_device_uuid: str

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        """Initialize weight update communicator."""
        try:
            for llm in app.state.llms:
                if hasattr(llm, 'engine'):
                    # Async engine - access workers through engine
                    workers = llm.engine.workers
                else:
                    # Sync engine - access workers through llm_engine
                    workers = llm.llm_engine.workers
                
                for worker in workers:
                    if hasattr(worker, 'init_communicator'):
                        worker.init_communicator(
                            request.host, 
                            request.port, 
                            request.world_size, 
                            request.client_device_uuid
                        )
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        """Update named parameter in the model."""
        try:
            for llm in app.state.llms:
                if hasattr(llm, 'engine'):
                    # Async engine
                    workers = llm.engine.workers
                else:
                    # Sync engine
                    workers = llm.llm_engine.workers
                
                for worker in workers:
                    if hasattr(worker, 'update_named_param'):
                        worker.update_named_param(request.name, request.dtype, request.shape)
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        """Reset prefix cache for the model."""
        try:
            for llm in app.state.llms:
                if hasattr(llm, 'engine') and hasattr(llm.engine, 'reset_prefix_cache'):
                    # Async engine
                    llm.engine.reset_prefix_cache()
                elif hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'reset_prefix_cache'):
                    # Sync engine
                    llm.llm_engine.reset_prefix_cache()
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.post("/close_communicator/")
    async def close_communicator():
        """Close weight update communicator."""
        try:
            for llm in app.state.llms:
                if hasattr(llm, 'engine'):
                    # Async engine
                    workers = llm.engine.workers
                else:
                    # Sync engine
                    workers = llm.llm_engine.workers
                
                for worker in workers:
                    if hasattr(worker, 'close_communicator'):
                        worker.close_communicator()
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Start the server
    config = uvicorn.Config(app, host=script_args.host, port=script_args.port, log_level=script_args.log_level)
    server = uvicorn.Server(config)
    await server.serve()


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    if subparsers is not None:
        parser = subparsers.add_parser("vllm-serve-async", help="Start an async vLLM server", dataclass_types=ScriptArguments)
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    asyncio.run(main(script_args))