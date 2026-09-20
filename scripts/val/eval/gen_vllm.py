import os
import json
import random
import re
import argparse
import concurrent.futures
import multiprocessing
import signal
import socket
import time
import gc
import torch
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from eval_integrity import validate_jsonl_output
# Try to import distributed cleanup helpers for releasing GPU memory.
try:
    from vllm.distributed.parallel_state import destroy_model_parallel
except ImportError:
    destroy_model_parallel = None
try:
    from vllm.distributed.parallel_state import destroy_distributed_environment
except ImportError:
    destroy_distributed_environment = None

# --------------------------------------------------------------------------- #
#                   Global constants / variables                              #
# --------------------------------------------------------------------------- #
DATA_DIR = "../data"
MODEL_FOLDER = "../../model"
# 输出基目录，可由统一 eval.sh 入口覆盖为每次运行的 eval_results/。
OUTPUT_BASE = "justrl_eval_outputs"
OUTPUT_NAME = None
STUDENT_ENABLE_THINKING = False

# --------------------------------------------------------------------------- #
#              Pre-allocate unique ports for parallel vLLM workers            #
# --------------------------------------------------------------------------- #
def _ephemeral_port_range() -> tuple[int, int]:
    """Return the OS ephemeral (auto-assigned) source-port range."""
    try:
        low, high = Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()[:2]
        return int(low), int(high)
    except Exception:
        return 32768, 60999


def _port_is_free(port: int) -> bool:
    """True only when nothing listens on *port* and we can still bind it."""
    # 1) A leftover EngineCore/vLLM process must not be accepting connections.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            return False
    # 2) The port must be bindable on every interface vLLM may use.
    for host in ("127.0.0.1", "0.0.0.0"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                probe.bind((host, port))
        except OSError:
            return False
    return True


def allocate_unique_ports(num_ports: int) -> list[int]:
    """
    Reserve *num_ports* distinct ports from the *non-ephemeral* range.

    Why not bind(0)?  After the probing socket is closed the port returns to
    the OS ephemeral pool, so one of the many bind(0) calls issued internally
    by the 8 concurrently starting vLLM EngineCore instances (ZMQ sockets,
    torch TCPStore, ...) can steal the reserved MASTER_PORT before vLLM binds
    it -- the exact EADDRINUSE failure seen in retries.  Picking ports
    strictly *below* ip_local_port_range removes that race entirely: the OS
    never auto-assigns from this window, so only another long-lived listener
    could collide, and those are detected by _port_is_free (which also skips
    ports still held by leftover processes from a crashed attempt).
    """
    ephemeral_low, _ = _ephemeral_port_range()
    cand_high = ephemeral_low - 1
    cand_low = max(11024, ephemeral_low - 20000)
    if cand_low >= cand_high:  # extremely small ephemeral range: fall back
        cand_low, cand_high = 11024, 32767

    start = random.randint(cand_low, cand_high)
    ports: list[int] = []
    candidate = start
    scanned = 0
    while len(ports) < num_ports and scanned <= (cand_high - cand_low):
        if candidate > cand_high:
            candidate = cand_low
        port = candidate
        candidate += 1
        scanned += 1
        if port in ports or not _port_is_free(port):
            continue
        ports.append(port)

    if len(ports) != num_ports:
        raise RuntimeError(
            f"Could not find {num_ports} free ports in [{cand_low}, {cand_high}]; "
            "leftover vLLM processes may still be holding ports"
        )
    print(f"Reserved worker ports (non-ephemeral window): {ports}", flush=True)
    return ports


# --------------------------------------------------------------------------- #
#                 Process-tree cleanup (EngineCore leftovers)                  #
# --------------------------------------------------------------------------- #
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _descendant_pids(root_pid: int) -> list[int]:
    """Collect every descendant PID of *root_pid* by walking /proc."""
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                stat = fh.read()
            # comm may contain spaces/parens; ppid follows the last ')'.
            rparen = stat.rfind(b")")
            if rparen == -1:
                continue
            fields = stat[rparen + 2:].split()
            children.setdefault(int(fields[1]), []).append(int(entry))
        except Exception:
            continue
    descendants: list[int] = []
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        for child in children.get(pid, []):
            descendants.append(child)
            stack.append(child)
    return descendants


def _kill_process_tree(root_pid: int, label: str = "") -> None:
    """SIGTERM-then-SIGKILL every process under *root_pid* (but not itself).

    A crashed worker can orphan vLLM EngineCore grandchildren which keep
    holding GPU memory and listening sockets; reaping them here makes the
    next retry start from a clean slate instead of colliding on EADDRINUSE.
    """
    pids = [pid for pid in _descendant_pids(root_pid) if pid != root_pid]
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.time() + 10
    remaining = [pid for pid in pids if _pid_alive(pid)]
    while remaining and time.time() < deadline:
        time.sleep(0.2)
        remaining = [pid for pid in remaining if _pid_alive(pid)]
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if remaining:
        print(f"[{label or root_pid}] force-killed {len(remaining)} leftover processes", flush=True)


def _abort_executor(executor: "concurrent.futures.ProcessPoolExecutor") -> None:
    """Fail fast: cancel queued futures and kill in-flight worker process trees."""
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:  # Python < 3.9 lacks cancel_futures
        executor.shutdown(wait=False)
    for pid in list((getattr(executor, "_processes", None) or {})):
        _kill_process_tree(pid, label=f"worker pid={pid}")
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def extract_max_number(path):
    """Extract all numbers from a path and return the largest one for sorting."""
    numbers = re.findall(r'\d+', path)
    if numbers:
        return max(int(n) for n in numbers)
    return -1  # If there is no number, keep this entry at the end.

# Collect model paths and sort them by descending numeric suffix.
try:
    model_files = os.listdir(MODEL_FOLDER)
    MODEL_NAMES_CANDIDATES = [os.path.join(MODEL_FOLDER, f) for f in model_files]
    MODEL_NAMES_CANDIDATES.sort(key=extract_max_number, reverse=True)
except (FileNotFoundError, NotADirectoryError, OSError):
    MODEL_NAMES_CANDIDATES = []

# Active model list.
MODEL_NAMES = MODEL_NAMES_CANDIDATES
MODEL_NAMES = ["../../model/Qwen3-4B"]

TASKS = [
    {"name": "AIME24", "path": f"{DATA_DIR}/AIME24/test.parquet", "N": 16},
    {"name": "AIME25", "path": f"{DATA_DIR}/AIME25/test.parquet", "N": 16},
    {"name": "AMC23", "path": f"{DATA_DIR}/AMC23/test.parquet", "N": 16},
    {"name": "HMMT24", "path": f"{DATA_DIR}/HMMT24/test.parquet", "N": 16},
    {"name": "HMMT25", "path": f"{DATA_DIR}/HMMT25/test.parquet", "N": 16},
    {"name": "MATH-500", "path": f"{DATA_DIR}/MATH-500/test.parquet", "N": 16},
]

PROMPT_TEMPLATE = """{problem} Please reason step by step, and put your final answer within \\boxed{{}}."""
MAX_TOKENS  = 31744
TEMPERATURE = 0.7
TOP_P       = 0.95
REPLACE     = False

# Avg@16/Pass@16 protocol: every problem is sampled N_ROLLOUTS times in one
# evaluation run.  Rollout j of a problem uses sampling seed
# ``base_seed + j`` (with the default base seed 42 that is seeds 42..57), so
# the 16 generations are independent draws without requiring 16 separate
# evaluation runs.  All rollouts of one benchmark are written into a single
# JSONL file (16 lines per example, rollout_id=0..N_ROLLOUTS-1).
N_ROLLOUTS  = 16

# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #
def load_samples(filepath: str):
    """Read parquet file and return a list of prompts (no duplication).

    Returns a tuple (samples, needs_suffix) where *needs_suffix* indicates
    whether the prompt suffix from PROMPT_TEMPLATE should be appended in
    main().  Datasets whose parquet 'prompt' column already embeds the
    instruction (e.g. "Please reason step by step …") must NOT receive a
    second copy of that suffix.
    """
    df = pd.read_parquet(filepath)
    # BRUMO25 / CMIMC25 store raw problem text without any instruction suffix.
    raw_problem_datasets = ("BRUMO25", "CMIMC25")
    is_raw = any(tag in filepath for tag in raw_problem_datasets)

    if is_raw:
        samples = [
            {
                "example_id": i,
                "prompt": df.at[i, "problem"].strip(),
                "answer": df.at[i, "answer"].strip(),
            }
            for i in range(len(df))
        ]
        needs_suffix = True   # prompt is bare problem text → append PROMPT_TEMPLATE
    else:
        samples = [
            {
                "example_id": i,
                "prompt": df.at[i, "prompt"][0]["content"].strip(),
                "answer": df.at[i, "reward_model"]["ground_truth"].strip(),
            }
            for i in range(len(df))
        ]
        needs_suffix = False  # prompt already contains the instruction suffix

    print(f"Total unique samples: {len(samples)} (needs_suffix={needs_suffix})")
    return samples, needs_suffix


def split_rollout_ids(rollout_ids: list[int], num_workers: int):
    """Round-robin split of rollout IDs into num_workers chunks."""
    chunks = [[] for _ in range(num_workers)]
    for idx, rollout_id in enumerate(rollout_ids):
        chunks[idx % num_workers].append(rollout_id)
    return chunks


# --------------------------------------------------------------------------- #
#              Worker process (one persistent model per GPU)                    #
# --------------------------------------------------------------------------- #
def worker_process(args_tuple):
    """
    Each worker loads the model once, then generates all assigned task/rollout
    requests in one vLLM call.

    args_tuple = (model_name, task_payloads, gpu_id, worker_port,
                  enable_thinking, base_seed, n_rollouts)

    gpu_id      : physical GPU index string, e.g. "3" – used for CUDA_VISIBLE_DEVICES.
    worker_port : unique TCP port pre-allocated by the parent process so that
                  concurrent vLLM EngineCore instances never collide on MASTER_PORT.
    base_seed   : evaluation base seed; rollout j of every sample uses sampling
                  seed ``base_seed + j`` (Avg@16/Pass@16 protocol).
    n_rollouts  : number of independent samples drawn per problem.
    """
    (model_name, task_payloads, gpu_id, worker_port,
     enable_thinking, base_seed, n_rollouts) = args_tuple
    if enable_thinking:
        raise ValueError("Student thinking mode is disabled for all OPD evaluations")
    base_seed = int(base_seed)
    n_rollouts = int(n_rollouts)

    # ------------------------------------------------------------------ #
    # 1. Isolate this worker to its dedicated GPU.
    # ------------------------------------------------------------------ #
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # ------------------------------------------------------------------ #
    # 2. Set a unique MASTER_PORT that was reserved by the parent process.
    #    Clear any inherited distributed-training env-vars so that vLLM's
    #    own init_process_group starts clean and does not inherit a stale
    #    RANK / WORLD_SIZE from a torchrun launcher.
    # ------------------------------------------------------------------ #
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(worker_port)
    os.environ["VLLM_HOST_IP"] = "127.0.0.1"

    for _var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "GROUP_RANK",
                 "LOCAL_WORLD_SIZE", "TORCHELASTIC_RESTART_COUNT"):
        os.environ.pop(_var, None)

    print(
        f"[GPU {gpu_id}] worker_port={worker_port} | "
        f"MASTER_PORT set to {worker_port}",
        flush=True,
    )
    
    results = {payload["name"]: [] for payload in task_payloads}
    llm = None
    stop_token_ids = []
    
    try:
        print(
            f"[GPU {gpu_id}] | Model: {model_name} | tasks={len(task_payloads)} "
            f"| loading model (TP=1, enable_thinking={enable_thinking})...",
            flush=True,
        )
        
        # Initialize a single-GPU, single-instance LLM.
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
            seed=base_seed,
        )
        
        # Get the tokenizer.
        try:
            tokenizer = llm.get_tokenizer()
            
            # Encode stop tokens.
            for stop_token in ["<|im_end|>", "<|endoftext|>"]:
                try:
                    if hasattr(tokenizer, "encode"):
                        encoded = tokenizer.encode(stop_token, add_special_tokens=False)
                        if encoded:
                            stop_token_ids.append(encoded[0])
                except Exception:
                    pass
        except Exception as e:
            tokenizer = None
            print(f"[GPU {gpu_id}] Warning: Could not get tokenizer for stop tokens: {e}", flush=True)
        
        if tokenizer is None:
            raise RuntimeError("Tokenizer is required for apply_chat_template, but it could not be loaded.")

        request_prompts = []
        request_sampling_params = []
        request_metadata = []

        for payload in task_payloads:
            task_name = payload["name"]
            samples = payload["samples"]
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": sample["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=STUDENT_ENABLE_THINKING,
                )
                for sample in samples
            ]

            for rollout_id in range(n_rollouts):
                rollout_seed = base_seed + rollout_id
                for sample, formatted_prompt in zip(samples, formatted_prompts):
                    request_prompts.append(formatted_prompt)
                    request_sampling_params.append(
                        SamplingParams(
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            max_tokens=MAX_TOKENS,
                            stop_token_ids=stop_token_ids if stop_token_ids else None,
                            seed=rollout_seed,
                        )
                    )
                    request_metadata.append((task_name, sample, rollout_id, rollout_seed))

        if not request_prompts:
            raise RuntimeError(f"GPU {gpu_id} received no evaluation requests")

        print(
            f"[GPU {gpu_id}] Generating {len(request_prompts)} requests in one vLLM call...",
            flush=True,
        )
        outputs = llm.generate(
            request_prompts,
            request_sampling_params,
            use_tqdm=False,
        )
        if len(outputs) != len(request_metadata):
            raise RuntimeError(
                f"GPU {gpu_id} returned {len(outputs)} outputs for "
                f"{len(request_metadata)} requests"
            )

        for metadata, out in zip(request_metadata, outputs):
            task_name, sample, rollout_id, rollout_seed = metadata
            results[task_name].append(
                {
                    "task_name": task_name,
                    "example_id": sample["example_id"],
                    "prompt": sample["prompt"],
                    "answer": sample["answer"],
                    "eval_seed": base_seed,
                    "rollout_id": rollout_id,
                    "rollout_seed": rollout_seed,
                    "student_enable_thinking": STUDENT_ENABLE_THINKING,
                    "response": out.outputs[0].text,
                }
            )
    
    except Exception as e:
        print(f"[GPU {gpu_id}] Critical Error: {e}", flush=True)
        raise
    
    finally:
        # Explicitly release vLLM resources.
        # This helps prevent CUDA context deadlocks and zombie processes.
        print(f"[GPU {gpu_id}] Cleaning up resources...", flush=True)
        if llm is not None:
            del llm

        # Reap any EngineCore/ZMQ grandchildren this worker may have orphaned
        # (happens when engine init fails partway, e.g. EADDRINUSE); otherwise
        # they keep their listening sockets and break the next retry attempt.
        try:
            _kill_process_tree(os.getpid(), label=f"GPU {gpu_id}")
        except Exception as cleanup_error:
            print(f"[GPU {gpu_id}] Warning: process-tree cleanup failed: {cleanup_error}", flush=True)

        if destroy_model_parallel is not None:
            try:
                destroy_model_parallel()
            except Exception:
                pass

        if destroy_distributed_environment is not None:
            try:
                destroy_distributed_environment()
            except Exception:
                pass
        
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[GPU {gpu_id}] Cleanup done.", flush=True)

    return results


def resolve_gpu_workers():
    """Resolve the exact GPU IDs assigned by the evaluation orchestrator."""
    requested = os.environ.get("EVAL_GPU_IDS", "").replace(",", " ").split()
    if requested:
        if len(set(requested)) != len(requested):
            raise ValueError(f"EVAL_GPU_IDS contains duplicates: {requested}")
        return requested

    visible_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_count == 0:
        raise RuntimeError("No CUDA GPU is visible and EVAL_GPU_IDS is not set")
    return [str(gpu_id) for gpu_id in range(visible_count)]


def validate_task_results(task_name, samples, results, eval_seed, n_rollouts):
    """Reject missing, duplicated, or mis-seeded generations before writing."""
    expected_keys = {
        (int(sample["example_id"]), rollout_id)
        for sample in samples
        for rollout_id in range(int(n_rollouts))
    }
    observed_keys = []
    for result in results:
        rollout_id = int(result["rollout_id"])
        if int(result["eval_seed"]) != int(eval_seed):
            raise RuntimeError(f"{task_name}: inconsistent eval_seed in worker output")
        if not 0 <= rollout_id < int(n_rollouts):
            raise RuntimeError(f"{task_name}: rollout_id {rollout_id} out of range")
        if int(result["rollout_seed"]) != int(eval_seed) + rollout_id:
            raise RuntimeError(f"{task_name}: inconsistent rollout_seed in worker output")
        observed_keys.append((int(result["example_id"]), rollout_id))

    observed_set = set(observed_keys)
    if len(observed_keys) != len(observed_set):
        raise RuntimeError(f"{task_name}: duplicate example/rollout generations detected")
    if observed_set != expected_keys:
        missing = sorted(expected_keys - observed_set)[:10]
        unexpected = sorted(observed_set - expected_keys)[:10]
        raise RuntimeError(
            f"{task_name}: incomplete generation; expected={len(expected_keys)}, "
            f"observed={len(observed_set)}, missing={missing}, unexpected={unexpected}"
        )


def write_jsonl_atomic(out_path, results):
    """Publish only complete task outputs so interrupted runs remain resumable."""
    temporary_path = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as stream:
        for item in results:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    temporary_path.replace(out_path)


# --------------------------------------------------------------------------- #
#                                   main                                      #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Generate evaluation rollouts with vLLM.")
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=int(os.environ.get("EVAL_SEED", "0")),
        help="Base request-level seed; rollout i uses eval_seed + i.",
    )
    parser.add_argument(
        "--eval-seeds",
        default=os.environ.get("EVAL_SEEDS", ""),
        help="Single base seed (Avg@16/Pass@16 protocol); rollout i uses base_seed + i.",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Deprecated compatibility flag; Student thinking cannot be enabled.",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable thinking when applying the chat template.",
    )
    parser.set_defaults(enable_thinking=STUDENT_ENABLE_THINKING)
    args = parser.parse_args()
    if args.enable_thinking:
        parser.error("Student thinking mode is fixed to false for fair OPD comparison")
    args.enable_thinking = STUDENT_ENABLE_THINKING
    eval_seeds = [
        int(value)
        for value in args.eval_seeds.replace(",", " ").split()
    ] if args.eval_seeds.strip() else [args.eval_seed]
    if len(eval_seeds) != len(set(eval_seeds)):
        raise ValueError(f"Duplicate evaluation seeds: {eval_seeds}")
    if len(eval_seeds) != 1:
        raise ValueError(
            "Avg@16/Pass@16 protocol requires exactly one base eval seed "
            f"(rollout i uses base_seed + i), got {eval_seeds}"
        )
    base_seed = eval_seeds[0]
    n_rollouts = int(N_ROLLOUTS)
    if n_rollouts < 1:
        raise ValueError(f"N_ROLLOUTS must be a positive integer, got {n_rollouts}")

    # One persistent model process per explicitly assigned GPU.
    gpu_workers = resolve_gpu_workers()
    num_workers = len(gpu_workers)

    print(f"GPU workers (one model per GPU): {gpu_workers}")
    print(f"apply_chat_template enable_thinking={args.enable_thinking}")

    for model_name in MODEL_NAMES:
        print(f"\n{'='*50}\nStarting evaluation for model: {model_name}\n{'='*50}")
        
        output_name = OUTPUT_NAME or model_name.split('/')[-1]
        OUT_DIR = Path(OUTPUT_BASE) / output_name
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        pending_tasks = []
        for task in TASKS:
            task_name = task["name"]
            task_path = task["path"]

            print(
                f"Starting evaluation for task: {task_name} "
                f"(N={n_rollouts} rollouts per problem, base seed={base_seed}, "
                f"rollout seeds={base_seed}..{base_seed + n_rollouts - 1})"
            )

            # 1. Load original prompts.
            # needs_suffix=True  → dataset stores raw problem text (BRUMO25/CMIMC25)
            # needs_suffix=False → dataset already embeds the instruction suffix
            samples, needs_suffix = load_samples(task_path)

            # Avg@16/Pass@16 protocol: one output file per benchmark holding all
            # n_rollouts generations of every example (rollout_id=0..n-1).
            # Only complete, metadata-consistent outputs are resumable. A
            # corrupt or interrupted file is regenerated and atomically replaced.
            out_path = OUT_DIR / (
                f"{task_name.lower()}_t{TEMPERATURE}_p{TOP_P}_n{n_rollouts}"
                f"-MNT{MAX_TOKENS}_seed{base_seed}.jsonl"
            )
            if not REPLACE and out_path.exists():
                valid, reason = validate_jsonl_output(
                    out_path, task_name, samples, range(n_rollouts), base_seed
                )
                if valid:
                    print(f"Validated complete result at '{out_path}'. Skipping.")
                    continue
                print(f"Existing result is not reusable ({reason}); regenerating: {out_path}")

            # Only append the instruction suffix when the dataset does not
            # already include it, to prevent double-suffixing (e.g. AMC23
            # parquet 'prompt' already ends with "Please reason step by step…").
            if needs_suffix:
                for sample in samples:
                    sample["prompt"] = PROMPT_TEMPLATE.format(problem=sample["prompt"])

            if len(samples) > 0:
                print("Example prompt after formatting:")
                print(samples[0]["prompt"])

            pending_tasks.append(
                {
                    "name": task_name,
                    "samples": samples,
                    "out_path": out_path,
                }
            )

        if not pending_tasks:
            print(f"All task outputs already exist for model: {model_name}")
            continue

        worker_payloads = [[] for _ in range(num_workers)]
        for task in pending_tasks:
            sample_chunks = [task["samples"][index::num_workers] for index in range(num_workers)]
            for worker_index, samples in enumerate(sample_chunks):
                if samples:
                    worker_payloads[worker_index].append(
                        {
                            "name": task["name"],
                            "samples": samples,
                        }
                    )

        # Pre-allocate a unique port for every live worker in the parent
        # process so that all child workers are guaranteed distinct ports
        # before any of them attempts to bind.
        live_worker_indices = [
            index for index in range(num_workers) if worker_payloads[index]
        ]
        worker_ports = allocate_unique_ports(len(live_worker_indices))
        port_map = dict(zip(live_worker_indices, worker_ports))

        args_list = [
            (
                model_name,
                worker_payloads[index],
                gpu_workers[index],
                port_map[index],
                args.enable_thinking,
                base_seed,
                n_rollouts,
            )
            for index in live_worker_indices
        ]
        if not args_list:
            raise RuntimeError("No GPU worker received evaluation requests")

        all_results = {task["name"]: [] for task in pending_tasks}
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(args_list), mp_context=ctx
        ) as executor:
            futures = [executor.submit(worker_process, worker_args) for worker_args in args_list]
            try:
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Persistent GPU workers",
                ):
                    worker_results = future.result()
                    for task_name, task_results in worker_results.items():
                        all_results[task_name].extend(task_results)
            except BaseException:
                # One worker failed (or the run was interrupted): kill every
                # sibling worker immediately instead of waiting hours for the
                # remaining generations, so the retry starts from a clean node.
                print("[MAIN] A worker failed; aborting all remaining workers...", flush=True)
                _abort_executor(executor)
                raise

        for task in pending_tasks:
            task_name = task["name"]
            task_results = all_results[task_name]
            validate_task_results(
                task_name, task["samples"], task_results, base_seed, n_rollouts
            )
            task_results.sort(
                key=lambda item: (int(item["example_id"]), int(item["rollout_id"]))
            )
            out_path = task["out_path"]
            write_jsonl_atomic(out_path, task_results)
            print(
                f"Saved {len(task_results)} results ({n_rollouts} rollouts x "
                f"{len(task['samples'])} problems) for {task_name}, "
                f"base_seed={base_seed} to {out_path}"
            )


if __name__ == "__main__":
    # Set the start method to avoid multiprocessing issues in some environments.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
