#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
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
"""Launch CPU or distributed GPU checkpoint conversion with NeMo Run."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

import nemo_run as run
from arguments import build_parser, conversion_worker_args
from nemo_run.config import get_nemorun_home
from torchx.specs.api import AppState


logger = logging.getLogger(__name__)

CONTAINER_REPO_ROOT = Path("/opt/Megatron-Bridge")

# This launcher is a patched copy that lives OUTSIDE the Megatron-Bridge repo
# (in Megatron-bridge-das/scripts/conversion). Point LOCAL_REPO_ROOT at the
# actual repo so bridge Python code stays on the worker's PYTHONPATH, and
# expose the local scripts dir so the patched worker + arguments are picked
# up ahead of the pristine 3rdparty copies.
_LOCAL_SCRIPTS_DIR = Path(__file__).resolve().parent
LOCAL_REPO_ROOT = _LOCAL_SCRIPTS_DIR.parents[1] / "3rdparty" / "Megatron-Bridge"
_ORIGINAL_CONVERSION_DIR = LOCAL_REPO_ROOT / "scripts" / "conversion"


def _parse_env(values: list[str]) -> list[str]:
    """Validate inherited environment variable names without materializing values."""
    env_names: list[str] = []
    for name in values:
        if "=" in name:
            raise ValueError("--env accepts NAME only; export the value before launching to keep it out of arguments.")
        if name not in os.environ:
            raise ValueError(f"Environment variable '{name}' is not set in the launcher environment.")
        if name not in env_names:
            env_names.append(name)
    return env_names


def _parse_mounts(values: list[str]) -> list[str]:
    """Normalize explicit same-path and host-to-container mounts."""
    mounts: list[str] = []
    for value in values:
        if ":" in value:
            host_path, container_path = value.split(":", 1)
        else:
            host_path = value
            container_path = value
        if not host_path or not container_path:
            raise ValueError(f"Invalid --mount value '{value}'; expected HOST or HOST:CONTAINER.")
        mount = f"{host_path}:{container_path}"
        if mount not in mounts:
            mounts.append(mount)
    return mounts


def _validate_args(args: argparse.Namespace) -> None:
    """Validate execution resources and conversion parallelism before launch."""
    if args.nodes < 1:
        raise ValueError("--nodes must be at least 1.")
    distributed_timeout_minutes = getattr(args, "distributed_timeout_minutes", None)
    if distributed_timeout_minutes is not None and distributed_timeout_minutes < 1:
        raise ValueError("--distributed-timeout-minutes must be at least 1.")
    if any(not value.strip() for value in args.srun_args):
        raise ValueError("--srun-arg values must not be empty.")
    for name in ("tp", "pp", "ep", "etp"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be at least 1.")

    if args.executor == "local":
        if args.nodes != 1:
            raise ValueError("Local execution supports exactly one node.")
        if args.detach:
            raise ValueError("--detach is only supported by the Slurm executor.")
        if args.srun_args:
            raise ValueError("--srun-arg is only supported by the Slurm executor.")
        if args.mount:
            raise ValueError("--mount is only supported by the Slurm executor; mount paths before local execution.")
    elif args.executor == "manual":
        if args.device != "gpu":
            raise ValueError("Manual executor is for multi-node GPU launches; use --executor local --device cpu instead.")
        if args.detach:
            raise ValueError("--detach is only supported by the Slurm executor.")
        if args.srun_args:
            raise ValueError("--srun-arg is only supported by the Slurm executor.")
        if args.mount:
            raise ValueError("--mount is only supported by the Slurm executor.")
        if args.submission_dry_run and args.node_rank != 0:
            # Every node prints the same command; rank 0 is enough.
            pass
        if args.node_rank < 0 or args.node_rank >= args.nodes:
            raise ValueError(f"--node-rank must satisfy 0 <= rank < --nodes ({args.nodes}).")
        if args.nodes > 1 and (not args.master_addr or args.master_addr == "127.0.0.1"):
            raise ValueError(
                "Multi-node manual execution requires --master-addr set to the master node's routable address."
            )
        if not (1 <= args.master_port <= 65535):
            raise ValueError("--master-port must be between 1 and 65535.")
    elif not args.account or not args.partition:
        raise ValueError("Slurm execution requires --account and --partition.")
    elif not args.container_image:
        raise ValueError("Slurm execution requires --container-image or CONTAINER_IMAGE.")

    if args.command == "roundtrip" and args.device != "gpu":
        raise ValueError("Round-trip validation requires the GPU backend.")

    if args.device == "cpu":
        if args.nodes != 1:
            raise ValueError("CPU conversion supports exactly one node and one process.")
        if args.gpus_per_node not in (None, 0):
            raise ValueError("CPU conversion does not accept --gpus-per-node.")
        if args.gres:
            raise ValueError("CPU conversion does not accept --gres.")
        if any(getattr(args, name) != 1 for name in ("tp", "pp", "ep", "etp")):
            raise ValueError("CPU conversion requires TP=PP=EP=ETP=1.")
    else:
        if args.gpus_per_node is None or args.gpus_per_node < 1:
            raise ValueError("GPU conversion requires --gpus-per-node of at least 1.")
        if args.executor == "local":
            worker_values = [args.hf_model]
            if args.command != "roundtrip":
                worker_values.append(args.megatron_path)
            if args.command == "export":
                worker_values.append(args.hf_path)
            if any(shlex.quote(value) != value for value in worker_values):
                raise ValueError(
                    "Local GPU execution cannot pass model IDs or paths containing whitespace or shell "
                    "metacharacters through NeMo Run 0.10; use shell-safe names and paths."
                )
        world_size = args.nodes * args.gpus_per_node
        model_parallel_size = args.tp * args.pp
        if world_size % model_parallel_size != 0:
            raise ValueError("nodes*gpus-per-node must be divisible by TP*PP.")
        expert_model_parallel_size = args.etp * args.ep * args.pp
        if world_size % expert_model_parallel_size != 0:
            raise ValueError("nodes*gpus-per-node must be divisible by ETP*EP*PP.")

    if args.command == "export":
        if args.save_every_n_ranks < 1:
            raise ValueError("--save-every-n-ranks must be at least 1.")
        distributed_save = args.distributed_save if args.distributed_save is not None else args.device == "gpu"
        if args.device == "cpu" and distributed_save:
            raise ValueError("--distributed-save is only supported by the GPU backend.")
        if not distributed_save and args.save_every_n_ranks != 1:
            raise ValueError("--save-every-n-ranks requires --distributed-save.")
        if args.device == "cpu" and args.export_weight_dtype is not None:
            raise ValueError("--export-weight-dtype is only supported by the GPU backend.")


def _build_executor(
    args: argparse.Namespace,
    env_names: list[str],
    mounts: list[str],
) -> object:
    """Build a Local or Slurm NeMo Run executor."""
    task_count = args.gpus_per_node if args.device == "gpu" else 1
    launcher = run.Torchrun() if args.executor == "local" and args.device == "gpu" else None
    if args.executor == "local":
        return run.LocalExecutor(
            nodes=1,
            ntasks_per_node=task_count,
            launcher=launcher,
            packager=run.Packager(),
        )

    gpu_kwargs = {}
    if args.device == "gpu" and not args.no_gpu_resource_request:
        gpu_kwargs["gpus_per_node"] = args.gpus_per_node
    container_env = [*env_names]
    if "PYTHONPATH" not in container_env:
        container_env.append("PYTHONPATH")
    executor = run.SlurmExecutor(
        account=args.account,
        partition=args.partition,
        job_name_prefix=args.experiment_name,
        nodes=args.nodes,
        ntasks_per_node=task_count,
        mem=args.mem,
        exclusive=True,
        time=args.time,
        gres=args.gres,
        launcher=launcher,
        tunnel=run.LocalTunnel(job_dir=os.path.join(get_nemorun_home(), "experiments")),
        packager=run.Packager(),
        container_image=args.container_image,
        container_mounts=mounts,
        container_env=container_env,
        additional_parameters={"export": ",".join(container_env)},
        srun_args=args.srun_args,
        **gpu_kwargs,
    )
    # Values are inherited by Slurm and selected by name for the container;
    # keeping env_vars empty prevents NeMo Run from serializing secrets.
    executor.env_vars = {}
    return executor


def _build_task(args: argparse.Namespace) -> tuple[run.Script, list[str]]:
    """Build the in-job conversion or round-trip task and its display arguments."""
    display_args = conversion_worker_args(args)
    if args.executor == "local":
        # Use the patched local worker; unchanged siblings (cpu_backend, utils)
        # remain in the 3rdparty conversion dir and are picked up via PYTHONPATH.
        task_path = _LOCAL_SCRIPTS_DIR / "run_conversion.py"
        repo_root = LOCAL_REPO_ROOT
    else:
        task_path = CONTAINER_REPO_ROOT / "scripts" / "conversion" / "run_conversion.py"
        repo_root = CONTAINER_REPO_ROOT
    task_args = display_args if args.executor == "local" else [shlex.quote(argument) for argument in display_args]
    if args.executor == "local":
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        # Include the pristine 3rdparty conversion dir so unchanged worker
        # siblings (cpu_backend, utils) resolve even though only a subset of
        # the modules live next to the patched worker.
        pythonpath = f"{repo_root}/src:{repo_root}/3rdparty/Megatron-LM:{_ORIGINAL_CONVERSION_DIR}"
        if existing_pythonpath:
            pythonpath = f"{pythonpath}:{existing_pythonpath}"
    else:
        pythonpath = f"{repo_root}/src:{repo_root}/3rdparty/Megatron-LM:$PYTHONPATH"
    task_env = {"PYTHONPATH": pythonpath}
    if args.executor == "local" and args.device == "gpu":
        # The torchrun console script can belong to a different Python than the
        # uv environment running this launcher. PyTorch honors PYTHON_EXEC for
        # workers, so keep conversion dependencies from this environment.
        task_env["PYTHON_EXEC"] = sys.executable
    task = run.Script(
        path=str(task_path),
        # NeMo Run recognizes the literal "python" entrypoint when building a
        # torchrun command. An absolute interpreter path is treated as a
        # non-Python executable and makes torchrun execute this file directly.
        entrypoint="python",
        env=task_env,
        args=task_args,
    )
    return task, display_args


def _raise_on_failed_tasks(experiment: run.Experiment) -> None:
    """Raise when a synchronous NeMo Run task did not complete successfully."""
    unsuccessful = [f"{job.id}={job.state}" for job in experiment.jobs if job.state != AppState.SUCCEEDED]
    if unsuccessful:
        raise RuntimeError(f"Conversion failed: {', '.join(unsuccessful)}")


def _run_manual(args: argparse.Namespace) -> None:
    """Run the worker directly under torchrun on this node (multi-node friendly).

    Every participating node runs the same convert.sh invocation with the same
    --nodes / --gpus-per-node / --master-addr / --master-port and varies only
    --node-rank. NeMo Run is bypassed entirely, so this works in environments
    without Slurm and without the NeMo Run + torchx worker orchestration.
    """
    worker_args = conversion_worker_args(args)
    worker_script = str(_LOCAL_SCRIPTS_DIR / "run_conversion.py")

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [
        f"{LOCAL_REPO_ROOT}/src",
        f"{LOCAL_REPO_ROOT}/3rdparty/Megatron-LM",
        str(_ORIGINAL_CONVERSION_DIR),
    ]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    # Consistent with the local executor path so torchrun spawns workers using
    # the same interpreter the launcher is running under.
    env.setdefault("PYTHON_EXEC", sys.executable)

    torchrun_cmd = [
        "torchrun",
        f"--nnodes={args.nodes}",
        f"--nproc_per_node={args.gpus_per_node}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        worker_script,
        *worker_args,
    ]

    logger.info(
        "Manual multi-node launch (node_rank=%d/%d) command: %s",
        args.node_rank,
        args.nodes,
        shlex.join(torchrun_cmd),
    )
    if args.nodes > 1:
        peer_hint = torchrun_cmd.copy()
        rank_index = next(i for i, arg in enumerate(peer_hint) if arg.startswith("--node_rank="))
        peer_hint[rank_index] = "--node_rank=<PEER_RANK>"
        logger.info("Run this same command on every peer, varying only --node-rank: %s", shlex.join(peer_hint))

    if args.submission_dry_run:
        return

    completed = subprocess.run(torchrun_cmd, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"torchrun exited with status {completed.returncode}")


def main(argv: list[str] | None = None) -> None:
    """Parse conversion arguments and launch the selected NeMo Run experiment."""
    args = build_parser(include_execution=True).parse_args(argv)
    _validate_args(args)

    if args.executor == "manual":
        # Manual mode bypasses NeMo Run entirely and drives torchrun on this
        # node. --env / --mount aren't meaningful here; the caller controls
        # the shell environment on each node directly.
        logger.info("Executor: manual; device: %s; nodes=%d node_rank=%d", args.device, args.nodes, args.node_rank)
        _run_manual(args)
        return

    env_names = _parse_env(args.env)
    mounts = _parse_mounts(args.mount)
    executor = _build_executor(args, env_names, mounts)
    task, worker_args = _build_task(args)

    experiment_name = args.experiment_name or f"conversion-{args.command}-{args.device}"
    logger.info(
        "Conversion command: %s",
        shlex.join(["python", task.path, *worker_args]),
    )
    logger.info("Executor: %s; device: %s", args.executor, args.device)
    logger.info("Forwarded environment variables: %s", ", ".join(env_names) or "none")
    if args.executor == "slurm":
        logger.info("Container mounts: %s", ", ".join(mounts) or "none")

    with run.Experiment(experiment_name) as experiment:
        experiment.add(task, executor=executor, name=f"{args.command}-{args.device}")
        if args.submission_dry_run:
            experiment.dryrun()
            return
        experiment.run(detach=args.detach, tail_logs=not args.detach)
    if not args.detach:
        _raise_on_failed_tasks(experiment)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
