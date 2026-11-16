"""
Tests for Data.collector_v2 — mocked hardware sensors, degradation paths.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Data.collector_v2 import (
    _init_nvml,
    _get_gpu_metrics,
    _empty_gpu_metrics,
    _init_lhm,
    _get_cpu_package_temp_and_clocks,
    _get_top_cpu_processes,
    _get_top_gpu_processes,
    get_workload_tag,
    set_workload_tag,
    take_snapshot,
    _apply_rolling_features,
)
from Core.schema_v2 import SystemSnapshotV2, ProcessInfo
from config import VALID_WORKLOAD_TAGS, WORKLOAD_TAG_FILE


class TestNVML:
    """Tests for NVML GPU metrics collection with mocking."""

    def test_empty_gpu_metrics_structure(self):
        metrics = _empty_gpu_metrics()
        expected_keys = {
            "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_mem_pct",
            "gpu_temp_c", "gpu_power_w", "gpu_power_limit_w", "gpu_core_clock_mhz",
            "gpu_mem_clock_mhz", "gpu_fan_pct", "gpu_throttle_reasons",
        }
        assert set(metrics.keys()) == expected_keys
        for k, v in metrics.items():
            if k == "gpu_throttle_reasons":
                assert v == []
            else:
                assert v is None

    @patch("Data.collector_v2._nvml_handle", None)
    @patch("Data.collector_v2._nvml_initialized", False)
    @patch("Data.collector_v2._nvml_warning_logged", False)
    def test_get_gpu_metrics_nvml_unavailable(self):
        """When NVML is not available, returns empty metrics with None values.

        Force the pynvml import to fail so the test is hermetic: it passes
        whether or not a real NVIDIA GPU/driver is present on the host.
        """
        with patch.dict(sys.modules, {"pynvml": None}):
            metrics = _get_gpu_metrics()
        empty = _empty_gpu_metrics()
        assert metrics == empty

    @patch("Data.collector_v2._nvml_handle", None)
    @patch("Data.collector_v2._nvml_initialized", False)
    @patch("Data.collector_v2._nvml_warning_logged", False)
    def test_init_nvml_import_error(self, capfd):
        """Test NVML init handles import error gracefully."""
        with patch.dict(sys.modules, {"pynvml": None}):
            result = _init_nvml()
            assert result is False
        out = capfd.readouterr()
        assert "NVML unavailable" in out.err

    @patch("Data.collector_v2._nvml_handle", None)
    @patch("Data.collector_v2._nvml_initialized", False)
    @patch("Data.collector_v2._nvml_warning_logged", False)
    def test_init_nvml_init_error(self, capfd):
        """Test NVML init handles nvmlInit() error gracefully."""
        mock_nvml = MagicMock()
        mock_nvml.nvmlInit.side_effect = Exception("Driver not found")
        with patch.dict(sys.modules, {"pynvml": mock_nvml}):
            result = _init_nvml()
            assert result is False
        out = capfd.readouterr()
        assert "NVML unavailable" in out.err

    def test_get_gpu_metrics_full_mock(self):
        """Test full GPU metrics collection with mocked NVML."""
        mock_nvml = MagicMock()

        # Mock device handle
        mock_handle = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        # Mock utilization
        mock_util = MagicMock()
        mock_util.gpu = 75
        mock_util.memory = 50
        mock_nvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        # Mock memory
        mock_mem = MagicMock()
        mock_mem.total = 8 * 1024**3  # 8 GB
        mock_mem.used = 4 * 1024**3   # 4 GB
        mock_mem.free = 4 * 1024**3
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        # Mock temperature
        mock_nvml.nvmlDeviceGetTemperature.return_value = 68

        # Mock power
        mock_nvml.nvmlDeviceGetPowerUsage.return_value = 150000  # mW
        mock_nvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = (100000, 200000)

        # Mock clocks
        mock_nvml.nvmlDeviceGetClockInfo.side_effect = lambda h, t: {
            mock_nvml.NVML_CLOCK_GRAPHICS: 1800,
            mock_nvml.NVML_CLOCK_MEM: 7000,
        }[t]

        # Mock fan
        mock_nvml.nvmlDeviceGetFanSpeed.return_value = 55

        # Mock throttle reasons - need to set all constants to proper int values
        mock_nvml.NVML_CLOCKS_THROTTLE_REASON_GPU_IDLE = 1
        mock_nvml.NVML_CLOCKS_THROTTLE_REASON_APPLICATIONS_CLOCKS_SETTING = 2
        mock_nvml.NVML_CLOCKS_THROTTLE_REASON_SW_POWER_CAP = 4
        mock_nvml.NVML_CLOCKS_THROTTLE_REASON_HW_POWER_CAP = 8
        mock_nvml.NVML_CLOCKS_THROTTLE_REASON_SW_THERMAL = 16
        mock_nvml.NVML_CLOCKS_THROTTLE_REASON_HW_THERMAL = 32
        mock_nvml.NVML_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE = 64
        mock_nvml.NVML_CLOCKS_THROTTLE_REASON_SYNC_BOOST = 128
        # Return value with only HW_THERMAL bit set (32)
        mock_nvml.nvmlDeviceGetCurrentClocksThrottleReasons.return_value = 32

        with patch("Data.collector_v2._nvml_handle", mock_nvml):
            with patch("Data.collector_v2._nvml_initialized", True):
                with patch("Data.collector_v2._nvml_warning_logged", True):
                    mock_nvml.nvmlDeviceGetCount.return_value = 1
                    metrics = _get_gpu_metrics()

        assert metrics["gpu_util_pct"] == 75.0
        assert metrics["gpu_mem_used_mb"] == 4096.0
        assert metrics["gpu_mem_total_mb"] == 8192.0
        assert metrics["gpu_mem_pct"] == 50.0
        assert metrics["gpu_temp_c"] == 68.0
        assert metrics["gpu_power_w"] == 150.0
        assert metrics["gpu_power_limit_w"] == 200.0
        assert metrics["gpu_core_clock_mhz"] == 1800.0
        assert metrics["gpu_mem_clock_mhz"] == 7000.0
        assert metrics["gpu_fan_pct"] == 55.0
        assert metrics["gpu_throttle_reasons"] == ["hw_thermal"]

    def test_get_gpu_metrics_partial_failure(self):
        """Test GPU metrics handles partial failures (some calls fail)."""
        mock_nvml = MagicMock()
        mock_handle = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_nvml.nvmlDeviceGetCount.return_value = 1

        # Utilization works
        mock_util = MagicMock()
        mock_util.gpu = 50
        mock_nvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        # Memory works
        mock_mem = MagicMock()
        mock_mem.total = 8 * 1024**3
        mock_mem.used = 2 * 1024**3
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        # Temperature fails
        mock_nvml.nvmlDeviceGetTemperature.side_effect = Exception("Not supported")

        # Power works
        mock_nvml.nvmlDeviceGetPowerUsage.return_value = 100000

        # Clocks fail
        mock_nvml.nvmlDeviceGetClockInfo.side_effect = Exception("Not supported")

        # Fan works
        mock_nvml.nvmlDeviceGetFanSpeed.return_value = 40

        # Throttle reasons fail
        mock_nvml.nvmlDeviceGetCurrentClocksThrottleReasons.side_effect = Exception("Not supported")

        with patch("Data.collector_v2._nvml_handle", mock_nvml):
            with patch("Data.collector_v2._nvml_initialized", True):
                with patch("Data.collector_v2._nvml_warning_logged", True):
                    metrics = _get_gpu_metrics()

        assert metrics["gpu_util_pct"] == 50.0
        assert metrics["gpu_mem_used_mb"] == 2048.0
        assert metrics["gpu_temp_c"] is None
        assert metrics["gpu_power_w"] == 100.0
        assert metrics["gpu_core_clock_mhz"] is None
        assert metrics["gpu_mem_clock_mhz"] is None
        assert metrics["gpu_fan_pct"] == 40.0
        assert metrics["gpu_throttle_reasons"] == []


class TestLHM:
    """Tests for LibreHardwareMonitor CPU package temp with mocking."""

    @patch("Data.collector_v2.platform.system", return_value="Linux")
    def test_init_lhm_non_windows(self, mock_platform):
        """LHM should return False on non-Windows without trying to import wmi."""
        result = _init_lhm()
        assert result is False

    @patch("Data.collector_v2.platform.system", return_value="Windows")
    @patch("Data.collector_v2._lhm_handle", None)
    @patch("Data.collector_v2._lhm_initialized", False)
    @patch("Data.collector_v2._lhm_warning_logged", False)
    def test_init_lhm_import_error(self, mock_platform, capfd):
        """Test LHM init handles import error gracefully."""
        with patch.dict(sys.modules, {"wmi": None}):
            result = _init_lhm()
            assert result is False
        out = capfd.readouterr()
        assert "LibreHardwareMonitor unavailable" in out.err

    @patch("Data.collector_v2.platform.system", return_value="Windows")
    @patch("Data.collector_v2._lhm_handle", None)
    @patch("Data.collector_v2._lhm_initialized", False)
    @patch("Data.collector_v2._lhm_warning_logged", False)
    def test_init_lhm_wmi_error(self, mock_platform, capfd):
        """Test LHM init handles WMI connection error gracefully."""
        mock_wmi = MagicMock()
        mock_wmi.WMI.side_effect = Exception("Access denied")
        with patch.dict(sys.modules, {"wmi": mock_wmi}):
            result = _init_lhm()
            assert result is False
        out = capfd.readouterr()
        assert "LibreHardwareMonitor unavailable" in out.err

    def test_get_cpu_package_temp_and_clocks_mock(self):
        """Test CPU package temp and clocks collection with mocked LHM."""
        mock_wmi = MagicMock()
        mock_lhm = MagicMock()

        # Create mock sensors
        sensor1 = MagicMock()
        sensor1.SensorType = "Temperature"
        sensor1.Name = "CPU Package"
        sensor1.Value = 65.0

        sensor2 = MagicMock()
        sensor2.SensorType = "Clock"
        sensor2.Name = "Core 0"
        sensor2.Value = 3200.0

        sensor3 = MagicMock()
        sensor3.SensorType = "Clock"
        sensor3.Name = "Core 1"
        sensor3.Value = 3300.0

        sensor4 = MagicMock()
        sensor4.SensorType = "Temperature"
        sensor4.Name = "CPU Core 0"  # Not package
        sensor4.Value = 60.0

        mock_lhm.Sensor.return_value = [sensor1, sensor2, sensor3, sensor4]
        mock_wmi.WMI.return_value = mock_lhm

        with patch("Data.collector_v2._lhm_handle", mock_lhm):
            with patch("Data.collector_v2._lhm_initialized", True):
                with patch("Data.collector_v2._lhm_warning_logged", True):
                    with patch("Data.collector_v2.platform.system", return_value="Windows"):
                        temp, clocks = _get_cpu_package_temp_and_clocks()

        assert temp == 65.0
        assert clocks == [3200.0, 3300.0]

    def test_get_cpu_package_temp_and_clocks_no_sensors(self):
        """Test LHM when no relevant sensors found."""
        mock_lhm = MagicMock()
        mock_lhm.Sensor.return_value = []

        with patch("Data.collector_v2._lhm_handle", mock_lhm):
            with patch("Data.collector_v2._lhm_initialized", True):
                with patch("Data.collector_v2._lhm_warning_logged", True):
                    with patch("Data.collector_v2.platform.system", return_value="Windows"):
                        temp, clocks = _get_cpu_package_temp_and_clocks()

        assert temp is None
        assert clocks == []

    def test_get_cpu_package_temp_and_clocks_exception(self):
        """Test LHM handles exceptions gracefully."""
        mock_lhm = MagicMock()
        mock_lhm.Sensor.side_effect = Exception("WMI error")

        with patch("Data.collector_v2._lhm_handle", mock_lhm):
            with patch("Data.collector_v2._lhm_initialized", True):
                with patch("Data.collector_v2._lhm_warning_logged", True):
                    with patch("Data.collector_v2.platform.system", return_value="Windows"):
                        temp, clocks = _get_cpu_package_temp_and_clocks()

        assert temp is None
        assert clocks == []


class TestPsutilHelpers:
    """Tests for psutil-based process collection."""

    @patch("Data.collector_v2.psutil.process_iter")
    def test_get_top_cpu_processes(self, mock_process_iter):
        """Test top CPU processes collection."""
        mock_proc1 = MagicMock()
        mock_proc1.info = {"pid": 100, "name": "chrome.exe", "cpu_percent": 25.0}
        mock_proc2 = MagicMock()
        mock_proc2.info = {"pid": 200, "name": "code.exe", "cpu_percent": 15.0}
        mock_proc3 = MagicMock()
        mock_proc3.info = {"pid": 300, "name": "idle.exe", "cpu_percent": 0.0}
        mock_proc4 = MagicMock()
        mock_proc4.info = {"pid": 400, "name": "system.exe", "cpu_percent": 5.0}

        mock_process_iter.return_value = [mock_proc1, mock_proc2, mock_proc3, mock_proc4]

        procs = _get_top_cpu_processes(3)
        assert len(procs) == 3
        assert procs[0].pid == 100
        assert procs[0].cpu_percent == 25.0
        assert procs[1].pid == 200
        assert procs[2].pid == 400

    @patch("Data.collector_v2.psutil.process_iter")
    def test_get_top_cpu_processes_access_denied(self, mock_process_iter):
        """Test top CPU processes handles AccessDenied."""
        mock_proc1 = MagicMock()
        mock_proc1.info = {"pid": 100, "name": "chrome.exe", "cpu_percent": 25.0}
        mock_proc2 = MagicMock()
        mock_proc2.info = {"pid": 200, "name": "system.exe", "cpu_percent": 10.0}
        mock_proc2.info = {"pid": 200, "name": "system.exe", "cpu_percent": 10.0}
        mock_proc2 = MagicMock()
        mock_proc2.info = {"pid": 200, "name": "system.exe", "cpu_percent": 10.0}
        # This will raise AccessDenied when accessed
        mock_proc3 = MagicMock()
        mock_proc3.info = {"pid": 300, "name": "protected.exe", "cpu_percent": 50.0}
        type(mock_proc3).info = property(lambda self: (_ for _ in ()).throw(psutil.AccessDenied()))

        import psutil
        mock_process_iter.return_value = [mock_proc1, mock_proc2, mock_proc3]

        procs = _get_top_cpu_processes(5)
        # Should get the two accessible ones
        assert len(procs) == 2

    @patch("Data.collector_v2._init_nvml", return_value=True)
    @patch("Data.collector_v2._nvml_handle")
    def test_get_top_gpu_processes(self, mock_nvml_handle, mock_init):
        """Test top GPU processes collection."""
        mock_nvml = mock_nvml_handle
        mock_handle = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        # Mock compute processes
        mock_compute = MagicMock()
        mock_compute.pid = 100
        mock_compute.usedGpuMemory = 2 * 1024**2  # 2 MB

        mock_graphics = MagicMock()
        mock_graphics.pid = 200
        mock_graphics.usedGpuMemory = 4 * 1024**2  # 4 MB

        mock_nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [mock_compute]
        mock_nvml.nvmlDeviceGetGraphicsRunningProcesses.return_value = [mock_graphics]

        # Mock psutil.Process for each
        with patch("Data.collector_v2.psutil.Process") as mock_psutil_process:
            mock_proc1 = MagicMock()
            mock_proc1.name.return_value = "python.exe"
            mock_proc1.cpu_percent.return_value = 10.0
            mock_proc2 = MagicMock()
            mock_proc2.name.return_value = "game.exe"
            mock_proc2.cpu_percent.return_value = 30.0
            mock_psutil_process.side_effect = [mock_proc1, mock_proc2]

            procs = _get_top_gpu_processes(5)

        assert len(procs) == 2
        # Should be sorted by GPU memory (4 MB first)
        assert procs[0].pid == 200
        assert procs[0].gpu_memory_mb == 4.0
        assert procs[1].pid == 100
        assert procs[1].gpu_memory_mb == 2.0

    @patch("Data.collector_v2._init_nvml", return_value=False)
    def test_get_top_gpu_processes_nvml_unavailable(self, mock_init):
        """Test top GPU processes returns empty when NVML unavailable."""
        procs = _get_top_gpu_processes(5)
        assert procs == []


class TestWorkloadTag:
    """Tests for workload tag file reading/writing."""

    def setup_method(self):
        # Clean up tag file before each test
        tag_file = Path(WORKLOAD_TAG_FILE)
        if tag_file.exists():
            tag_file.unlink()
        # Reset global state
        import Data.collector_v2 as collector_module
        collector_module._workload_tag = "idle"

    def teardown_method(self):
        tag_file = Path(WORKLOAD_TAG_FILE)
        if tag_file.exists():
            tag_file.unlink()
        # Reset global state
        import Data.collector_v2 as collector_module
        collector_module._workload_tag = "idle"

    def test_get_workload_tag_default(self):
        """Default tag should be 'idle' when no file exists."""
        tag = get_workload_tag()
        assert tag == "idle"

    def test_get_workload_tag_from_file(self):
        """Should read tag from file."""
        Path(WORKLOAD_TAG_FILE).write_text("gaming")
        tag = get_workload_tag()
        assert tag == "gaming"

    def test_get_workload_tag_invalid_in_file(self):
        """Invalid tag in file should fall back to current/default."""
        Path(WORKLOAD_TAG_FILE).write_text("invalid_tag")
        tag = get_workload_tag()
        assert tag == "idle"  # default

    def test_get_workload_tag_whitespace(self):
        """Should strip whitespace from file."""
        Path(WORKLOAD_TAG_FILE).write_text("  compile  \n")
        tag = get_workload_tag()
        assert tag == "compile"

    def test_set_workload_tag_valid(self):
        """Setting valid tag should write to file and update global."""
        result = set_workload_tag("browsing")
        assert result is True
        assert get_workload_tag() == "browsing"
        assert Path(WORKLOAD_TAG_FILE).read_text().strip() == "browsing"

    def test_set_workload_tag_invalid(self):
        """Setting invalid tag should return False and not update."""
        # First set a valid tag
        set_workload_tag("gaming")
        # Then try invalid
        result = set_workload_tag("not_a_valid_tag")
        assert result is False
        assert get_workload_tag() == "gaming"  # unchanged from previous valid

    def test_set_workload_tag_case_insensitive(self):
        """Tag setting should be case insensitive."""
        result = set_workload_tag("GAMING")
        assert result is True
        assert get_workload_tag() == "gaming"

    def test_valid_workload_tags_constant(self):
        """VALID_WORKLOAD_TAGS should contain expected values."""
        expected = {"idle", "gaming", "compile", "browsing", "stress"}
        assert VALID_WORKLOAD_TAGS == expected


def test_rolling_buffers_are_bounded_and_populate_features(monkeypatch):
    import Data.collector_v2 as collector

    monkeypatch.setattr(collector, "_cpu_history", deque(maxlen=2))
    monkeypatch.setattr(collector, "_ram_history", deque(maxlen=2))
    monkeypatch.setattr(collector, "_bw_history", deque(maxlen=2))

    snapshots = [
        SystemSnapshotV2(cpu_pct=10, ram_pct=30, bandwidth_kbps=100),
        SystemSnapshotV2(cpu_pct=20, ram_pct=40, bandwidth_kbps=200),
        SystemSnapshotV2(cpu_pct=30, ram_pct=50, bandwidth_kbps=300),
    ]
    for snap in snapshots:
        _apply_rolling_features(snap)

    assert len(collector._cpu_history) == 2
    assert len(collector._ram_history) == 2
    assert len(collector._bw_history) == 2
    assert snapshots[-1].cpu_delta == 10
    assert snapshots[-1].cpu_rolling_mean == 25
    assert snapshots[-1].ram_rolling_mean == 45
    assert snapshots[-1].bw_rolling_mean == 250
    assert snapshots[-1].bw_spike == 1.2


class TestTakeSnapshot:
    """Tests for the main take_snapshot function."""

    @patch("Data.collector_v2.psutil.cpu_percent")
    @patch("Data.collector_v2.psutil.virtual_memory")
    @patch("Data.collector_v2.psutil.disk_usage")
    @patch("Data.collector_v2.psutil.net_io_counters")
    @patch("Data.collector_v2._get_gpu_metrics")
    @patch("Data.collector_v2._get_cpu_package_temp_and_clocks")
    @patch("Data.collector_v2._get_top_cpu_processes")
    @patch("Data.collector_v2._get_top_gpu_processes")
    @patch("Data.collector_v2.get_workload_tag")
    @patch("Data.collector_v2.psutil.Process")
    def test_take_snapshot_basic(
        self,
        mock_psutil_process,
        mock_get_workload_tag,
        mock_get_top_gpu,
        mock_get_top_cpu,
        mock_get_cpu_temp,
        mock_get_gpu_metrics,
        mock_net_io,
        mock_disk_usage,
        mock_virtual_memory,
        mock_cpu_percent,
    ):
        """Test basic snapshot collection with all mocks."""
        # CPU
        mock_cpu_percent.side_effect = [45.0, [40.0, 50.0, 42.0, 48.0]]

        # RAM
        mock_vmem = MagicMock()
        mock_vmem.percent = 60.0
        mock_vmem.used = 8 * 1024**3
        mock_vmem.available = 5 * 1024**3
        mock_vmem.total = 16 * 1024**3
        mock_virtual_memory.return_value = mock_vmem

        # Disk
        mock_disk = MagicMock()
        mock_disk.percent = 50.0
        mock_disk.free = 500 * 1024**3
        mock_disk.total = 1000 * 1024**3
        mock_disk_usage.return_value = mock_disk

        # Network
        mock_net = MagicMock()
        mock_net.bytes_sent = 1000000
        mock_net.bytes_recv = 2000000
        mock_net.packets_sent = 5000
        mock_net.packets_recv = 6000
        mock_net_io.return_value = mock_net

        # GPU
        mock_get_gpu_metrics.return_value = _empty_gpu_metrics()

        # CPU temp (LHM)
        mock_get_cpu_temp.return_value = (None, [])

        # Top processes
        mock_get_top_cpu.return_value = [ProcessInfo(pid=100, name="test.exe", cpu_percent=20.0)]
        mock_get_top_gpu.return_value = []

        # Workload tag
        mock_get_workload_tag.return_value = "idle"

        # Collector self
        mock_self_proc = MagicMock()
        mock_self_proc.cpu_percent.return_value = 0.5
        mock_self_proc.memory_info.return_value = MagicMock(rss=50 * 1024**2)
        mock_psutil_process.return_value = mock_self_proc

        snap = take_snapshot()

        assert isinstance(snap, SystemSnapshotV2)
        assert snap.cpu_pct == 45.0
        assert snap.cpu_per_core_pct == [40.0, 50.0, 42.0, 48.0]
        assert snap.ram_pct == 60.0
        assert snap.ram_used_mb == 8192.0
        assert snap.disk_pct == 50.0
        assert snap.workload_tag == "idle"
        assert snap.collector_cpu_pct == 0.5
        assert snap.collector_rss_mb == 50.0

    @patch("Data.collector_v2.psutil.cpu_percent")
    @patch("Data.collector_v2.psutil.virtual_memory")
    @patch("Data.collector_v2.psutil.disk_usage")
    @patch("Data.collector_v2.psutil.net_io_counters")
    @patch("Data.collector_v2._get_gpu_metrics")
    @patch("Data.collector_v2._get_cpu_package_temp_and_clocks")
    @patch("Data.collector_v2._get_top_cpu_processes")
    @patch("Data.collector_v2._get_top_gpu_processes")
    @patch("Data.collector_v2.get_workload_tag")
    @patch("Data.collector_v2.psutil.Process")
    def test_take_snapshot_with_gpu_and_cpu_temp(
        self,
        mock_psutil_process,
        mock_get_workload_tag,
        mock_get_top_gpu,
        mock_get_top_cpu,
        mock_get_cpu_temp,
        mock_get_gpu_metrics,
        mock_net_io,
        mock_disk_usage,
        mock_virtual_memory,
        mock_cpu_percent,
    ):
        """Test snapshot with real GPU metrics and CPU package temp."""
        mock_cpu_percent.side_effect = [80.0, [75.0, 85.0, 82.0, 78.0]]

        mock_vmem = MagicMock()
        mock_vmem.percent = 70.0
        mock_vmem.used = 12 * 1024**3
        mock_vmem.available = 4 * 1024**3
        mock_vmem.total = 16 * 1024**3
        mock_virtual_memory.return_value = mock_vmem

        mock_disk = MagicMock()
        mock_disk.percent = 60.0
        mock_disk.free = 400 * 1024**3
        mock_disk.total = 1000 * 1024**3
        mock_disk_usage.return_value = mock_disk

        mock_net = MagicMock()
        mock_net.bytes_sent = 5000000
        mock_net.bytes_recv = 10000000
        mock_net.packets_sent = 10000
        mock_net.packets_recv = 15000
        mock_net_io.return_value = mock_net

        # GPU with real data
        mock_get_gpu_metrics.return_value = {
            "gpu_util_pct": 90.0,
            "gpu_mem_used_mb": 6144.0,
            "gpu_mem_total_mb": 8192.0,
            "gpu_mem_pct": 75.0,
            "gpu_temp_c": 75.0,
            "gpu_power_w": 200.0,
            "gpu_power_limit_w": 220.0,
            "gpu_core_clock_mhz": 1900.0,
            "gpu_mem_clock_mhz": 7500.0,
            "gpu_fan_pct": 70.0,
            "gpu_throttle_reasons": ["sw_power_cap"],
        }

        # CPU package temp
        mock_get_cpu_temp.return_value = (72.0, [3500.0, 3600.0, 3400.0, 3700.0])

        mock_get_top_cpu.return_value = [ProcessInfo(pid=1, name="gcc", cpu_percent=60.0)]
        mock_get_top_gpu.return_value = [ProcessInfo(pid=2, name="nvcc", cpu_percent=20.0, gpu_memory_mb=4096.0)]

        mock_get_workload_tag.return_value = "compile"

        mock_self_proc = MagicMock()
        mock_self_proc.cpu_percent.return_value = 1.2
        mock_self_proc.memory_info.return_value = MagicMock(rss=60 * 1024**2)
        mock_psutil_process.return_value = mock_self_proc

        snap = take_snapshot()

        assert snap.cpu_pct == 80.0
        assert snap.cpu_package_temp_c == 72.0
        assert snap.cpu_core_clocks_mhz == [3500.0, 3600.0, 3400.0, 3700.0]
        assert snap.gpu_util_pct == 90.0
        assert snap.gpu_temp_c == 75.0
        assert snap.gpu_throttle_reasons == ["sw_power_cap"]
        assert snap.workload_tag == "compile"
        assert len(snap.top_cpu_processes) == 1
        assert len(snap.top_gpu_processes) == 1

    @patch("Data.collector_v2.time.time", side_effect=[100.0, 101.0])
    @patch("Data.collector_v2.psutil.cpu_percent")
    @patch("Data.collector_v2.psutil.virtual_memory")
    @patch("Data.collector_v2.psutil.disk_usage")
    @patch("Data.collector_v2.psutil.net_io_counters")
    @patch("Data.collector_v2._get_gpu_metrics")
    @patch("Data.collector_v2._get_cpu_package_temp_and_clocks")
    @patch("Data.collector_v2._get_top_cpu_processes")
    @patch("Data.collector_v2._get_top_gpu_processes")
    @patch("Data.collector_v2.get_workload_tag")
    @patch("Data.collector_v2.psutil.Process")
    def test_take_snapshot_network_delta(
        self,
        mock_psutil_process,
        mock_get_workload_tag,
        mock_get_top_gpu,
        mock_get_top_cpu,
        mock_get_cpu_temp,
        mock_get_gpu_metrics,
        mock_net_io,
        mock_disk_usage,
        mock_virtual_memory,
        mock_cpu_percent,
        mock_time,
    ):
        """Test network bandwidth calculation uses delta between calls."""
        # Need to reset module globals between tests
        import Data.collector_v2 as collector_module
        collector_module._prev_net = None
        collector_module._prev_net_time = None

        mock_cpu_percent.side_effect = [50.0, [45.0, 55.0], 50.0, [45.0, 55.0]]
        mock_vmem = MagicMock(percent=50.0, used=8*1024**3, available=8*1024**3, total=16*1024**3)
        mock_virtual_memory.return_value = mock_vmem
        mock_disk = MagicMock(percent=50.0, free=500*1024**3, total=1000*1024**3)
        mock_disk_usage.return_value = mock_disk

        mock_get_gpu_metrics.return_value = _empty_gpu_metrics()
        mock_get_cpu_temp.return_value = (None, [])
        mock_get_top_cpu.return_value = []
        mock_get_top_gpu.return_value = []
        mock_get_workload_tag.return_value = "idle"
        mock_self_proc = MagicMock()
        mock_self_proc.cpu_percent.return_value = 0.5
        mock_self_proc.memory_info.return_value = MagicMock(rss=50*1024**2)
        mock_psutil_process.return_value = mock_self_proc

        # First call - no previous, bandwidth should be 0
        mock_net1 = MagicMock()
        mock_net1.bytes_sent = 1000
        mock_net1.bytes_recv = 2000
        mock_net1.packets_sent = 10
        mock_net1.packets_recv = 20
        mock_net_io.return_value = mock_net1

        snap1 = take_snapshot()
        assert snap1.bandwidth_kbps == 0.0  # First call, no delta
        assert snap1.packet_rate_pps == 0.0

        # Second call - with previous, should calculate delta
        # Use simple objects with numeric attributes instead of MagicMock for arithmetic
        class NetCounters:
            def __init__(self, sent, recv, pkts_sent, pkts_recv):
                self.bytes_sent = sent
                self.bytes_recv = recv
                self.packets_sent = pkts_sent
                self.packets_recv = pkts_recv

        mock_net_io.return_value = NetCounters(1000 + 125000, 2000 + 125000, 10 + 500, 20 + 500)

        snap2 = take_snapshot()
        # 250 KB * 8 bits / 1 second = 2000 kbps
        expected_kbps = (250000 * 8) / (1 * 1000)  # 2000 kbps
        assert abs(snap2.bandwidth_kbps - expected_kbps) < 1.0
        assert abs(snap2.packet_rate_pps - 1000.0) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
