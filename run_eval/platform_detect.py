#!/usr/bin/env python3
"""
Platform Detection Module - Detects GPU (AMD/NVIDIA) and Network (EFA/IB) platforms.
"""

import subprocess
import shutil
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class GPUInfo:
    vendor: str
    compiler: Optional[str] = None
    compiler_version: Optional[str] = None
    architectures: List[str] = field(default_factory=list)
    device_count: int = 0
    device_names: List[str] = field(default_factory=list)
    driver_version: Optional[str] = None

    @property
    def platform(self) -> str:
        if self.vendor == "AMD":
            return "hip"
        elif self.vendor == "NVIDIA":
            return "cuda"
        return "none"

    @property
    def available(self) -> bool:
        return self.vendor != "None" and self.compiler is not None


@dataclass
class NetworkInfo:
    type: str
    devices: List[str] = field(default_factory=list)
    device_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    has_rdma: bool = False
    has_ibverbs: bool = False

    @property
    def available(self) -> bool:
        return self.type in ("EFA", "IB") and self.has_rdma


@dataclass
class PlatformInfo:
    gpu: GPUInfo
    network: NetworkInfo
    hostname: str = ""
    kernel_version: str = ""

    def summary(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("PLATFORM DETECTION RESULTS")
        lines.append("=" * 60)
        lines.append(f"Hostname: {self.hostname}")
        lines.append(f"Kernel: {self.kernel_version}")
        lines.append("")

        lines.append("-" * 40)
        lines.append("GPU Platform:")
        lines.append("-" * 40)
        if self.gpu.available:
            lines.append(f"  Vendor: {self.gpu.vendor}")
            lines.append(f"  Compiler: {self.gpu.compiler}")
            if self.gpu.compiler_version:
                lines.append(f"  Compiler Version: {self.gpu.compiler_version}")
            lines.append(f"  Device Count: {self.gpu.device_count}")
            for i, name in enumerate(self.gpu.device_names):
                lines.append(f"    [{i}] {name}")
            if self.gpu.architectures:
                lines.append(f"  Architectures: {', '.join(self.gpu.architectures)}")
            if self.gpu.driver_version:
                lines.append(f"  Driver Version: {self.gpu.driver_version}")
        else:
            lines.append("  No GPU detected")
        lines.append("")

        lines.append("-" * 40)
        lines.append("Network Platform:")
        lines.append("-" * 40)
        if self.network.available:
            lines.append(f"  Type: {self.network.type}")
            lines.append(f"  RDMA Support: {'Yes' if self.network.has_rdma else 'No'}")
            lines.append(f"  libibverbs: {'Available' if self.network.has_ibverbs else 'Not found'}")
            if self.network.devices:
                lines.append("  Devices:")
                for dev in self.network.devices:
                    details = self.network.device_details.get(dev, {})
                    dev_type = details.get("type", "unknown")
                    lines.append(f"    - {dev} ({dev_type})")
        else:
            lines.append("  No RDMA network detected")
            lines.append(f"  libibverbs: {'Available' if self.network.has_ibverbs else 'Not found'}")

        lines.append("=" * 60)
        return "\n".join(lines)


def _run_command(cmd: List[str], timeout: int = 10) -> tuple:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False, "", ""


def detect_amd_gpu() -> Optional[GPUInfo]:
    hipcc = shutil.which("hipcc")
    if not hipcc:
        return None

    info = GPUInfo(vendor="AMD", compiler=hipcc)

    success, stdout, _ = _run_command([hipcc, "--version"])
    if success and stdout:
        for line in stdout.split("\n"):
            if "HIP version" in line or "hip version" in line.lower():
                info.compiler_version = line.strip()
                break
            elif "clang version" in line.lower():
                info.compiler_version = line.strip()
                break

    success, stdout, _ = _run_command(["rocminfo"])
    if success and stdout:
        for line in stdout.split("\n"):
            line = line.strip()
            if "Name:" in line and "CPU" not in line:
                name = line.split("Name:")[-1].strip()
                if name and name not in info.device_names:
                    info.device_names.append(name)
            if "gfx" in line.lower():
                for part in line.split():
                    if part.startswith("gfx") and part not in info.architectures:
                        info.architectures.append(part)
        info.device_count = len(info.device_names)

    if info.device_count == 0:
        success, stdout, _ = _run_command(["rocm-smi", "--showid"])
        if success and stdout:
            for line in stdout.split("\n"):
                if "GPU" in line and "ID" in line:
                    info.device_count += 1

    if not info.architectures:
        info.architectures = ["gfx906", "gfx908", "gfx90a", "gfx942"]

    return info


def detect_nvidia_gpu() -> Optional[GPUInfo]:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None

    info = GPUInfo(vendor="NVIDIA", compiler=nvcc)

    success, stdout, _ = _run_command([nvcc, "--version"])
    if success and stdout:
        for line in stdout.split("\n"):
            if "release" in line.lower():
                info.compiler_version = line.strip()
                break

    success, stdout, _ = _run_command([
        "nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"
    ])
    if success and stdout:
        for line in stdout.strip().split("\n"):
            if line.strip():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 1:
                    info.device_names.append(parts[0])
                if len(parts) >= 2 and not info.driver_version:
                    info.driver_version = parts[1]
                if len(parts) >= 3:
                    arch = f"sm_{parts[2].replace('.', '')}"
                    if arch not in info.architectures:
                        info.architectures.append(arch)
        info.device_count = len(info.device_names)

    if info.device_count == 0:
        success, stdout, _ = _run_command(["nvidia-smi", "-L"])
        if success and stdout:
            for line in stdout.split("\n"):
                if line.strip().startswith("GPU"):
                    info.device_count += 1
                    match = re.search(r"GPU \d+: (.+?) \(", line)
                    if match:
                        info.device_names.append(match.group(1))

    if not info.architectures:
        info.architectures = ["sm_70"]

    return info


def detect_gpu() -> GPUInfo:
    amd_info = detect_amd_gpu()
    if amd_info and amd_info.available:
        return amd_info

    nvidia_info = detect_nvidia_gpu()
    if nvidia_info and nvidia_info.available:
        return nvidia_info

    return GPUInfo(vendor="None")


def check_ibverbs() -> bool:
    header_paths = ["/usr/include/infiniband/verbs.h", "/usr/local/include/infiniband/verbs.h"]
    if not any(os.path.exists(p) for p in header_paths):
        return False

    lib_paths = [
        "/usr/lib/x86_64-linux-gnu/libibverbs.so",
        "/usr/lib64/libibverbs.so",
        "/usr/lib/libibverbs.so"
    ]
    if any(os.path.exists(p) for p in lib_paths):
        return True

    success, _, _ = _run_command(["pkg-config", "--exists", "libibverbs"])
    return success


def detect_efa() -> Optional[List[Dict[str, Any]]]:
    success, stdout, _ = _run_command(["fi_info", "-p", "efa"])
    if not success:
        return None

    devices = []
    current_device = {}

    for line in stdout.split("\n"):
        line = line.strip()
        if line.startswith("provider:") and "efa" in line.lower():
            if current_device:
                devices.append(current_device)
            current_device = {"type": "EFA", "provider": line.split(":")[-1].strip()}
        elif line.startswith("fabric:"):
            current_device["fabric"] = line.split(":")[-1].strip()
        elif line.startswith("domain:"):
            current_device["domain"] = line.split(":")[-1].strip()

    if current_device:
        devices.append(current_device)

    return devices if devices else None


def detect_infiniband() -> Optional[List[Dict[str, Any]]]:
    success, stdout, _ = _run_command(["ibv_devices"])
    if not success:
        return None

    devices = []

    for line in stdout.strip().split("\n")[1:]:
        parts = line.split()
        if not parts:
            continue
        device_name = parts[0]
        if device_name.startswith("-") or device_name == "device":
            continue

        device_info = {"name": device_name, "type": "IB"}

        stat_success, stat_stdout, _ = _run_command(["ibstat", device_name])
        if stat_success and stat_stdout:
            for stat_line in stat_stdout.split("\n"):
                stat_line = stat_line.strip()
                if "Port GUID:" in stat_line:
                    device_info["port_guid"] = stat_line.split(":")[-1].strip()
                elif "Link layer:" in stat_line:
                    link_layer = stat_line.split(":")[-1].strip()
                    device_info["link_layer"] = link_layer
                    if "Ethernet" in link_layer:
                        device_info["type"] = "RoCE"
                elif "Rate:" in stat_line:
                    device_info["rate"] = stat_line.split(":")[-1].strip()

        if "efa" in device_name.lower():
            device_info["type"] = "EFA"
        elif "mlx" in device_name.lower():
            device_info["type"] = "IB"

        devices.append(device_info)

    return devices if devices else None


def detect_network() -> NetworkInfo:
    info = NetworkInfo(type="None")
    info.has_ibverbs = check_ibverbs()

    efa_devices = detect_efa()
    if efa_devices:
        info.type = "EFA"
        info.has_rdma = True
        for dev in efa_devices:
            domain = dev.get("domain", dev.get("name", "efa"))
            info.devices.append(domain)
            info.device_details[domain] = dev
        return info

    ib_devices = detect_infiniband()
    if ib_devices:
        has_ib = any(d.get("type") == "IB" for d in ib_devices)
        has_efa = any(d.get("type") == "EFA" for d in ib_devices)
        has_roce = any(d.get("type") == "RoCE" for d in ib_devices)

        if has_efa:
            info.type = "EFA"
        elif has_ib:
            info.type = "IB"
        elif has_roce:
            info.type = "RoCE"
        else:
            info.type = "IB"

        info.has_rdma = True
        for dev in ib_devices:
            name = dev.get("name", "unknown")
            info.devices.append(name)
            info.device_details[name] = dev
        return info

    info.type = "Ethernet"
    info.has_rdma = False
    return info


def detect_platform(verbose: bool = False) -> PlatformInfo:
    if verbose:
        print("Detecting platform...")

    hostname = ""
    kernel = ""

    success, stdout, _ = _run_command(["hostname"])
    if success:
        hostname = stdout.strip()

    success, stdout, _ = _run_command(["uname", "-r"])
    if success:
        kernel = stdout.strip()

    if verbose:
        print("  Checking for GPU...")
    gpu_info = detect_gpu()

    if verbose:
        print("  Checking for RDMA network...")
    network_info = detect_network()

    return PlatformInfo(
        gpu=gpu_info,
        network=network_info,
        hostname=hostname,
        kernel_version=kernel
    )


def get_platform_string(platform_info: PlatformInfo) -> str:
    """Returns platform string like 'AMD_IB', 'NV_EFA', 'NV', 'CPU_IB', etc."""
    parts = []

    if platform_info.gpu.vendor == "AMD":
        parts.append("AMD")
    elif platform_info.gpu.vendor == "NVIDIA":
        parts.append("NV")
    else:
        parts.append("CPU")

    if platform_info.network.type == "EFA":
        parts.append("EFA")
    elif platform_info.network.type == "IB":
        parts.append("IB")
    elif platform_info.network.type == "RoCE":
        parts.append("RoCE")

    return "_".join(parts)


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Detect hardware platform (GPU and network)")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output platform string")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detection progress")

    args = parser.parse_args()
    platform_info = detect_platform(verbose=args.verbose)

    if args.quiet:
        print(get_platform_string(platform_info))
    elif args.json:
        output = {
            "platform_string": get_platform_string(platform_info),
            "hostname": platform_info.hostname,
            "kernel": platform_info.kernel_version,
            "gpu": {
                "vendor": platform_info.gpu.vendor,
                "platform": platform_info.gpu.platform,
                "available": platform_info.gpu.available,
                "compiler": platform_info.gpu.compiler,
                "compiler_version": platform_info.gpu.compiler_version,
                "device_count": platform_info.gpu.device_count,
                "device_names": platform_info.gpu.device_names,
                "architectures": platform_info.gpu.architectures,
                "driver_version": platform_info.gpu.driver_version,
            },
            "network": {
                "type": platform_info.network.type,
                "available": platform_info.network.available,
                "has_rdma": platform_info.network.has_rdma,
                "has_ibverbs": platform_info.network.has_ibverbs,
                "devices": platform_info.network.devices,
                "device_details": platform_info.network.device_details,
            }
        }
        print(json.dumps(output, indent=2))
    else:
        print(platform_info.summary())


if __name__ == "__main__":
    main()
