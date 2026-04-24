import json
import os
import socket
import subprocess
import sys
import traceback
from datetime import datetime


def heading(title):
    print(f"\n=== {title} ===", flush=True)


def safe_run(cmd):
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "output": proc.stdout.strip(),
        }
    except Exception as exc:
        return {
            "cmd": cmd,
            "error": f"{type(exc).__name__}: {exc}",
        }


def print_json(label, data):
    print(f"{label}: {json.dumps(data, indent=2, sort_keys=True)}", flush=True)


def check_torch():
    heading("Torch Check")
    try:
        import torch

        data = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cuda_device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            data["device_0_name"] = torch.cuda.get_device_name(0)
        print_json("torch", data)
        return True
    except Exception:
        traceback.print_exc()
        return False


def check_system():
    heading("System Check")
    data = {
        "timestamp": datetime.now().isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        "SLURM_JOB_NODELIST": os.environ.get("SLURM_JOB_NODELIST"),
        "SLURM_GPUS": os.environ.get("SLURM_GPUS"),
        "SLURM_GPUS_ON_NODE": os.environ.get("SLURM_GPUS_ON_NODE"),
        "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
    }
    print_json("system", data)
    print_json("nvidia_smi_L", safe_run(["nvidia-smi", "-L"]))
    print_json("nvidia_smi", safe_run(["nvidia-smi"]))


def check_ray():
    heading("Ray Driver And Worker Check")
    import ray

    results = {}
    ray.init(
        num_gpus=1,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=True,
    )
    try:
        results["cluster_resources"] = ray.cluster_resources()
        results["available_resources"] = ray.available_resources()
        results["driver_gpu_ids"] = ray.get_gpu_ids()
        print_json("ray_driver", results)

        @ray.remote
        def cpu_probe():
            import os
            import ray
            import torch

            return {
                "kind": "cpu_probe",
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "ray_get_gpu_ids": ray.get_gpu_ids(),
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_cuda_device_count": torch.cuda.device_count(),
            }

        @ray.remote(num_gpus=1)
        def gpu_probe():
            import os
            import ray
            import torch

            data = {
                "kind": "gpu_probe",
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "ray_get_gpu_ids": ray.get_gpu_ids(),
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_cuda_device_count": torch.cuda.device_count(),
            }
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                data["torch_device_0_name"] = torch.cuda.get_device_name(0)
            return data

        cpu_result = ray.get(cpu_probe.remote())
        print_json("ray_cpu_probe", cpu_result)
        gpu_result = ray.get(gpu_probe.remote())
        print_json("ray_gpu_probe", gpu_result)
        return True
    finally:
        ray.shutdown()


def check_rllib():
    heading("RLlib Smoke Check")
    import ray
    from ray.rllib.agents.ppo import PPOTrainer

    tests = [
        {"name": "ppo_num_workers_0", "num_workers": 0, "num_gpus": 1},
        {"name": "ppo_num_workers_1", "num_workers": 1, "num_gpus": 1},
    ]

    for test in tests:
        print(f"running {test['name']}", flush=True)
        ray.init(
            num_gpus=1,
            include_dashboard=False,
            ignore_reinit_error=True,
            log_to_driver=True,
        )
        try:
            config = {
                "env": "CartPole-v1",
                "framework": "torch",
                "num_workers": test["num_workers"],
                "num_gpus": test["num_gpus"],
                "num_envs_per_worker": 1,
                "log_level": "WARN",
                "rollout_fragment_length": 100,
                "train_batch_size": 200,
                "sgd_minibatch_size": 50,
            }
            trainer = PPOTrainer(env="CartPole-v1", config=config)
            try:
                result = trainer.train()
                summary = {
                    "training_iteration": result.get("training_iteration"),
                    "timesteps_total": result.get("timesteps_total"),
                    "episode_reward_mean": result.get("episode_reward_mean"),
                }
                print_json(test["name"], summary)
            finally:
                trainer.stop()
        except Exception:
            print(f"{test['name']} failed", flush=True)
            traceback.print_exc()
        finally:
            ray.shutdown()


def main():
    check_system()
    check_torch()
    try:
        check_ray()
    except Exception:
        traceback.print_exc()
    try:
        check_rllib()
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
