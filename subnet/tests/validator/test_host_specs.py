"""Tests for the validator startup min-spec check."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from subnet.validator import host_specs
from subnet.validator.host_specs import (
    MIN_CPUS,
    MIN_RAM_GIB,
    HostSpecs,
    check_host_min_specs,
    read_host_specs,
)


class TestHostSpecsDataclass:
    def test_no_deficits_when_at_minimum(self):
        specs = HostSpecs(cpus=MIN_CPUS, ram_gib=MIN_RAM_GIB)
        assert specs.deficits() == []

    def test_no_deficits_when_above_minimum(self):
        specs = HostSpecs(cpus=MIN_CPUS + 2, ram_gib=MIN_RAM_GIB + 4)
        assert specs.deficits() == []

    def test_cpu_deficit_only(self):
        specs = HostSpecs(cpus=2, ram_gib=MIN_RAM_GIB + 4)
        d = specs.deficits()
        assert len(d) == 1
        assert d[0].startswith("cpus=2 (min ")

    def test_ram_deficit_only(self):
        specs = HostSpecs(cpus=MIN_CPUS + 2, ram_gib=3.8)
        d = specs.deficits()
        assert len(d) == 1
        assert d[0].startswith("ram_gib=3.8 (min ")

    def test_both_deficits(self):
        # The WildSage box: 2 vCPU, 3.824 GiB — both should trip.
        specs = HostSpecs(cpus=2, ram_gib=3.824)
        d = specs.deficits()
        assert len(d) == 2
        assert any("cpus=2" in s for s in d)
        assert any("ram_gib=3.8" in s for s in d)

    def test_zero_values_do_not_deficit(self):
        # If we cannot read a value we should not fabricate a deficit.
        assert HostSpecs(cpus=0, ram_gib=0.0).deficits() == []


class TestReadHostSpecs:
    def test_reads_process_visible_cpus(self):
        # sched_getaffinity is Linux-only; patch with create=True so this test
        # runs on macOS dev boxes as well as Linux CI.
        with patch.object(
            host_specs.os,
            "sched_getaffinity",
            create=True,
            return_value={0, 1, 2, 3},
        ):
            specs = read_host_specs()
            assert specs.cpus == 4

    def test_falls_back_to_cpu_count_when_affinity_unavailable(self):
        with patch.object(
            host_specs.os,
            "sched_getaffinity",
            create=True,
            side_effect=AttributeError(),
        ):
            with patch.object(host_specs.os, "cpu_count", return_value=7):
                specs = read_host_specs()
                assert specs.cpus == 7

    def test_reads_ram_from_proc_meminfo(self, tmp_path, monkeypatch):
        fake_meminfo = tmp_path / "meminfo"
        # 16 GiB in KiB
        fake_meminfo.write_text("MemTotal:       16777216 kB\nMemFree:        1000000 kB\n")
        real_open = open

        def fake_open(path, *args, **kwargs):
            if path == "/proc/meminfo":
                return real_open(fake_meminfo, *args, **kwargs)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        specs = read_host_specs()
        assert specs.ram_gib == pytest.approx(16.0, abs=0.01)

    def test_returns_zero_ram_on_read_failure(self, monkeypatch):
        def raise_ose(*args, **kwargs):
            raise OSError("no /proc here")

        monkeypatch.setattr("builtins.open", raise_ose)
        specs = read_host_specs()
        assert specs.ram_gib == 0.0


class TestCheckHostMinSpecsLogging:
    def test_warns_when_under_spec(self):
        with patch.object(host_specs, "logging") as mock_log, patch.object(
            host_specs, "read_host_specs", return_value=HostSpecs(cpus=2, ram_gib=3.8)
        ):
            check_host_min_specs()
        mock_log.warning.assert_called_once()
        mock_log.info.assert_not_called()
        args = mock_log.warning.call_args.args
        assert "HOST UNDER MIN SPEC" in args[0]
        assert "cpus=2" in args[1]
        assert "ram_gib=3.8" in args[1]

    def test_info_when_ok(self):
        with patch.object(host_specs, "logging") as mock_log, patch.object(
            host_specs, "read_host_specs", return_value=HostSpecs(cpus=8, ram_gib=16.0)
        ):
            check_host_min_specs()
        mock_log.info.assert_called_once()
        mock_log.warning.assert_not_called()
        args = mock_log.info.call_args.args
        assert "Host specs OK" in args[0]

    def test_returns_specs_regardless(self):
        with patch.object(host_specs, "read_host_specs", return_value=HostSpecs(cpus=2, ram_gib=3.8)):
            got = check_host_min_specs()
        assert got == HostSpecs(cpus=2, ram_gib=3.8)
