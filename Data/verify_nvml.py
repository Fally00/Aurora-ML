"""
Aurora-ML NVML Verification Script

Tests nvidia-ml-py (NVML) on target GPUs:
- RTX 5050 (laptop)
- RTX 4060 (desktop)

Run: python -m Data.verify_nvml
"""

from __future__ import annotations

import sys
from typing import Optional


def verify_nvml() -> bool:
    """Verify NVML is working and print all readable GPU counters."""
    try:
        import pynvml as nvml
    except ImportError as e:
        print(f"FAIL: nvidia-ml-py not installed: {e}")
        return False

    try:
        nvml.nvmlInit()
    except Exception as e:
        print(f"FAIL: NVML init failed: {e}")
        return False

    try:
        driver_version = nvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode("utf-8")
        print(f"Driver Version: {driver_version}")
    except Exception as e:
        print(f"WARN: Could not get driver version: {e}")

    try:
        device_count = nvml.nvmlDeviceGetCount()
        print(f"GPU Count: {device_count}")
    except Exception as e:
        print(f"FAIL: Could not get device count: {e}")
        return False

    if device_count == 0:
        print("FAIL: No GPUs detected")
        return False

    for i in range(device_count):
        print(f"\n{'='*60}")
        print(f"GPU {i}")
        print(f"{'='*60}")

        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
        except Exception as e:
            print(f"FAIL: Could not get handle for GPU {i}: {e}")
            continue

        # Name
        try:
            name = nvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            print(f"Name: {name}")
        except Exception as e:
            print(f"Name: ERROR - {e}")

        # UUID
        try:
            uuid = nvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8")
            print(f"UUID: {uuid}")
        except Exception as e:
            print(f"UUID: ERROR - {e}")

        # Architecture
        try:
            arch = nvml.nvmlDeviceGetArchitecture(handle)
            arch_names = {
                nvml.NVML_DEVICE_ARCH_KEPLER: "Kepler",
                nvml.NVML_DEVICE_ARCH_MAXWELL: "Maxwell",
                nvml.NVML_DEVICE_ARCH_PASCAL: "Pascal",
                nvml.NVML_DEVICE_ARCH_VOLTA: "Volta",
                nvml.NVML_DEVICE_ARCH_TURING: "Turing",
                nvml.NVML_DEVICE_ARCH_AMPERE: "Ampere",
                nvml.NVML_DEVICE_ARCH_HOPPER: "Hopper",
                nvml.NVML_DEVICE_ARCH_ADA: "Ada Lovelace",
                nvml.NVML_DEVICE_ARCH_BLACKWELL: "Blackwell",
            }
            print(f"Architecture: {arch_names.get(arch, f'Unknown ({arch})')}")
        except Exception as e:
            print(f"Architecture: ERROR - {e}")

        # Temperature
        try:
            temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            print(f"Temperature: {temp}°C")
        except Exception as e:
            print(f"Temperature: ERROR - {e}")

        # Utilization
        try:
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU Utilization: {util.gpu}%")
            print(f"Memory Utilization: {util.memory}%")
        except Exception as e:
            print(f"Utilization: ERROR - {e}")

        # Memory Info
        try:
            mem = nvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"Memory Total: {mem.total / (1024**3):.2f} GB")
            print(f"Memory Used: {mem.used / (1024**3):.2f} GB")
            print(f"Memory Free: {mem.free / (1024**3):.2f} GB")
            print(f"Memory Usage: {(mem.used / mem.total * 100):.1f}%")
        except Exception as e:
            print(f"Memory Info: ERROR - {e}")

        # Power
        try:
            power = nvml.nvmlDeviceGetPowerUsage(handle)
            print(f"Power Usage: {power / 1000:.2f} W")
        except Exception as e:
            print(f"Power Usage: ERROR - {e}")

        try:
            power_limit = nvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
            print(f"Power Limit Range: {power_limit[0] / 1000:.2f} W - {power_limit[1] / 1000:.2f} W")
            current_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle)
            print(f"Current Power Limit: {current_limit / 1000:.2f} W")
        except Exception as e:
            print(f"Power Limit: ERROR - {e}")

        # Clocks
        try:
            core_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)
            mem_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
            sm_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
            video_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_VIDEO)
            print(f"Graphics Clock: {core_clock} MHz")
            print(f"Memory Clock: {mem_clock} MHz")
            print(f"SM Clock: {sm_clock} MHz")
            print(f"Video Clock: {video_clock} MHz")
        except Exception as e:
            print(f"Clocks: ERROR - {e}")

        # Max Clocks
        try:
            max_core = nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)
            max_mem = nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_MEM)
            max_sm = nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_SM)
            print(f"Max Graphics Clock: {max_core} MHz")
            print(f"Max Memory Clock: {max_mem} MHz")
            print(f"Max SM Clock: {max_sm} MHz")
        except Exception as e:
            print(f"Max Clocks: ERROR - {e}")

        # Fan Speed
        try:
            fan = nvml.nvmlDeviceGetFanSpeed(handle)
            print(f"Fan Speed: {fan}%")
        except Exception as e:
            print(f"Fan Speed: ERROR - {e}")

        # Throttle Reasons
        try:
            throttle = nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            reasons = []
            reason_map = {
                nvml.NVML_CLOCKS_THROTTLE_REASON_GPU_IDLE: "GPU Idle",
                nvml.NVML_CLOCKS_THROTTLE_REASON_APPLICATIONS_CLOCKS_SETTING: "Applications Clocks Setting",
                nvml.NVML_CLOCKS_THROTTLE_REASON_SW_POWER_CAP: "SW Power Cap",
                nvml.NVML_CLOCKS_THROTTLE_REASON_HW_POWER_CAP: "HW Power Cap",
                nvml.NVML_CLOCKS_THROTTLE_REASON_SW_THERMAL: "SW Thermal",
                nvml.NVML_CLOCKS_THROTTLE_REASON_HW_THERMAL: "HW Thermal",
                nvml.NVML_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE: "HW Power Brake",
                nvml.NVML_CLOCKS_THROTTLE_REASON_SYNC_BOOST: "Sync Boost",
            }
            for bit, name in reason_map.items():
                if throttle & bit:
                    reasons.append(name)
            if reasons:
                print(f"Throttle Reasons: {', '.join(reasons)}")
            else:
                print("Throttle Reasons: None")
        except Exception as e:
            print(f"Throttle Reasons: ERROR - {e}")

        # ECC Mode
        try:
            ecc_mode = nvml.nvmlDeviceGetEccMode(handle)
            ecc_enabled = nvml.nvmlDeviceGetEccMode(handle)
            print(f"ECC Mode: {'Enabled' if ecc_enabled else 'Disabled'}")
        except Exception as e:
            print(f"ECC Mode: ERROR - {e}")

        # Compute Mode
        try:
            compute_mode = nvml.nvmlDeviceGetComputeMode(handle)
            compute_modes = {
                nvml.NVML_COMPUTEMODE_DEFAULT: "Default",
                nvml.NVML_COMPUTEMODE_EXCLUSIVE_THREAD: "Exclusive Thread",
                nvml.NVML_COMPUTEMODE_PROHIBITED: "Prohibited",
                nvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS: "Exclusive Process",
            }
            print(f"Compute Mode: {compute_modes.get(compute_mode, f'Unknown ({compute_mode})')}")
        except Exception as e:
            print(f"Compute Mode: ERROR - {e}")

        # Running Processes
        try:
            compute_procs = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
            graphics_procs = nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            all_procs = compute_procs + graphics_procs
            if all_procs:
                print(f"Running Processes ({len(all_procs)}):")
                for p in all_procs:
                    used_mem = p.usedGpuMemory / (1024**2) if p.usedGpuMemory else 0
                    print(f"  PID {p.pid}: {used_mem:.1f} MB")
            else:
                print("Running Processes: None")
        except Exception as e:
            print(f"Running Processes: ERROR - {e}")

        # Performance State
        try:
            pstate = nvml.nvmlDeviceGetPerformanceState(handle)
            print(f"Performance State: P{pstate}")
        except Exception as e:
            print(f"Performance State: ERROR - {e}")

        # PCI Info
        try:
            pci = nvml.nvmlDeviceGetPciInfo(handle)
            print(f"PCI Bus ID: {pci.busId}")
            print(f"PCI Domain: {pci.domain}")
            print(f"PCI Bus: {pci.bus}")
            print(f"PCI Device: {pci.device}")
            print(f"PCI Sub System ID: {pci.subSystemId:04x}")
        except Exception as e:
            print(f"PCI Info: ERROR - {e}")

        # Compute Capability
        try:
            # NVML doesn't directly expose compute capability, but we can infer from architecture
            pass
        except Exception as e:
            print(f"Compute Capability: ERROR - {e}")

    print(f"\n{'='*60}")
    print("NVML Verification Complete")
    print(f"{'='*60}")
    return True


def test_collector_metrics() -> bool:
    """Test the exact metrics the collector will use."""
    try:
        import pynvml as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print(f"Cannot test collector metrics: {e}")
        return False

    print("\n--- Collector Metrics Test ---")

    # These are the exact calls collector_v2.py makes
    tests = [
        ("nvmlDeviceGetUtilizationRates", lambda: nvml.nvmlDeviceGetUtilizationRates(handle)),
        ("nvmlDeviceGetMemoryInfo", lambda: nvml.nvmlDeviceGetMemoryInfo(handle)),
        ("nvmlDeviceGetTemperature", lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)),
        ("nvmlDeviceGetPowerUsage", lambda: nvml.nvmlDeviceGetPowerUsage(handle)),
        ("nvmlDeviceGetPowerManagementLimitConstraints", lambda: nvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)),
        ("nvmlDeviceGetClockInfo (GRAPHICS)", lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)),
        ("nvmlDeviceGetClockInfo (MEM)", lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)),
        ("nvmlDeviceGetFanSpeed", lambda: nvml.nvmlDeviceGetFanSpeed(handle)),
        ("nvmlDeviceGetCurrentClocksThrottleReasons", lambda: nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)),
    ]

    all_passed = True
    for name, fn in tests:
        try:
            result = fn()
            print(f"  OK: {name}")
        except Exception as e:
            print(f"  FAIL: {name} - {e}")
            all_passed = False

    return all_passed


def main():
    print("Aurora-ML NVML Verification")
    print("=" * 60)

    success = verify_nvml()
    if success:
        success = test_collector_metrics()

    if success:
        print("\n✓ All NVML checks passed")
        sys.exit(0)
    else:
        print("\n✗ Some NVML checks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
