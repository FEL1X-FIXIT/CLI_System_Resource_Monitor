import psutil
import pynvml
import time
import argparse
import subprocess
from rich.live import Live
from rich.panel import Panel
from rich.console import Group
from rich.table import Table
import os
from collections import defaultdict

pynvml.nvmlInit()
gpu_count = pynvml.nvmlDeviceGetCount()
group = Group()

prev_disk = {}
def get_disk_delta():
    global prev_disk
    curr = {}
    stats = {}
    per_disk = psutil.disk_io_counters(perdisk=True)
    for disk, io in per_disk.items():
        curr[disk] = (io.read_bytes, io.write_bytes)
        if disk in prev_disk:
            read_diff = curr[disk][0] - prev_disk[disk][0]
            write_diff = curr[disk][1] - prev_disk[disk][1]
            stats[disk] = {
                "in": read_diff / 1024**2,
                "out": write_diff / 1024**2
            }
        else:
            stats[disk] = {"in": 0.0, "out": 0.0}
    prev_disk = curr
    return stats


def get_disk_size_gb(disk_name):
    """
    Get size of the block device in GB from /sys/block.
    This skips partitions like sda1 and only works for actual disk names like sda.
    """
    sys_block_path = f"/sys/block/{disk_name}/size"
    try:
        with open(sys_block_path, "r") as f:
            sectors = int(f.read().strip())
            size_bytes = sectors * 512  # 512 bytes per sector
            return size_bytes / (1024**3)
    except Exception:
        return 0  # Treat unreadable as 0 GB
    

def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        for name, entries in temps.items():
            for entry in entries:
                if 'cpu' in entry.label.lower() or 'package' in entry.label.lower():
                    return f"{entry.current:.1f}°C"
        return "N/A"
    except Exception:
        return "N/A"


def get_physical_cpu_info():
    cpuinfo = defaultdict(lambda: {"model": None, "cores": set(), "threads": 0, "logical_ids": []})
    with open("/proc/cpuinfo") as f:
        physical_id = core_id = model_name = None
        processor_id = None
        for line in f:
            if line.strip() == "":
                if physical_id is not None and processor_id is not None:
                    cpuinfo[physical_id]["threads"] += 1
                    cpuinfo[physical_id]["logical_ids"].append(int(processor_id))
                    if core_id is not None:
                        cpuinfo[physical_id]["cores"].add(core_id)
                    if model_name and not cpuinfo[physical_id]["model"]:
                        cpuinfo[physical_id]["model"] = model_name
                physical_id = core_id = model_name = processor_id = None
                continue
            if line.startswith("processor"):
                processor_id = line.strip().split(":")[1].strip()
            elif line.startswith("physical id"):
                physical_id = line.strip().split(":")[1].strip()
            elif line.startswith("model name"):
                model_name = line.strip().split(":")[1].strip()
            elif line.startswith("core id"):
                core_id = line.strip().split(":")[1].strip()
    return cpuinfo


def get_stats_cpu(per_core = False):
    stats = {}
    cpuinfo = get_physical_cpu_info()
    freqs = psutil.cpu_freq(percpu=True)
    cpu_percents = psutil.cpu_percent(percpu=True)
    temps = psutil.sensors_temperatures()
    cpu_temps = {}
    for name, entries in temps.items():
        for entry in entries:
            label_lower = entry.label.lower()
            if 'package id' in label_lower:
                # Extract CPU index from label, e.g. "Package id 0"
                try:
                    cpu_idx = int(label_lower.split('package id')[-1].strip())
                except ValueError:
                    cpu_idx = 0
                cpu_temps[cpu_idx] = f"{entry.current:.0f}°C"
            elif ('cpu' in label_lower or 'package' in label_lower) and 0 not in cpu_temps:
                # Fallback for generic labels, assign to CPU0 only once
                cpu_temps[0] = f"{entry.current:.0f}°C"

    for cpu_index, (pid, info) in enumerate(cpuinfo.items()):
        logical_ids = info['logical_ids']
        if not logical_ids:
            continue

        # Use only relevant logical CPUs for this socket
        usage = sum(cpu_percents[i] for i in logical_ids) / len(logical_ids)
        current_clock = sum(freqs[i].current for i in logical_ids) / len(logical_ids) / 1000

        freq_info = psutil.cpu_freq()
        max_clock = freq_info.max / 1000 if freq_info else 0

        raw_model = info['model'] or "Unknown"
        model_parts = raw_model.split(" @")
        model = model_parts[0]
        base_clock_str = model_parts[1] if len(model_parts) > 1 else ""
        base_clock = float(base_clock_str.replace("GHz", "").strip()) if base_clock_str else 0
        cores = len(info['cores'])
        threads = info['threads']

        try:
            temp = cpu_temps.get(cpu_index, "N/A")  # get temp per CPU index
        except:
            temp = "N/A"

        stats[f"CPU{cpu_index}:CPU{cpu_index} - {model}"] = f"{usage:.1f} %"
        stats[f"CPU{cpu_index}:{cores} Cores @{base_clock:.2f}GHz (max {max_clock:.2f}GHz)- ({threads} Threads)"] = f"{current_clock:.2f}GHz"
        stats[f"CPU{cpu_index}:Temp"] = temp
        if cpu_index < len(cpuinfo) - 1:
            stats[f"CPU{cpu_index}: "] = " "

    return stats       


def get_stats_fan():
    stats = {}
    try:
        fans = psutil.sensors_fans()
        fan_id = 0
        for name, entries in fans.items():
            for entry in entries:
                fan_speed = entry.current
                stats[f"FAN:Fan {fan_id}"] = f"{fan_speed}RPM"
                fan_id += 1
    except Exception:
        stats[f"FAN:Fan"] = "N/A"
    return stats


def get_stats_ram():
    stats = {}
    total_width = None
    data_width = None
    ecc_status = "N/A"
    cmd = "sudo dmidecode --type 17"  # Type 17 = Memory Device
    try:
        output = subprocess.check_output(cmd, shell=True, text=True)
        blocks = output.strip().split("Memory Device")

        for block in blocks:
            slot = size = speed = ram_type = ecc_status = "N/A"
            lines = block.strip().splitlines()
            if not lines:
                continue
            
            skip = False
            for line in lines:
                line = line.strip()
                if line.startswith("Locator:"):
                    slot = line.split(":", 1)[1].strip() 
                    if "NO DIMM" in slot:
                        skip = True   
                elif line.startswith("Size:") and "No Module Installed" not in line:
                    size = line.split(":", 1)[1].strip()
                    size_raw = line.split(":", 1)[1].strip()
                    try:
                        if "MB" in size_raw:
                            size = str(int(size_raw.replace("MB", "").strip()) // 1024)
                        elif "GB" in size_raw:
                            size = size_raw.replace("GB", "").strip()
                        else:
                            size = "N/A"
                    except:
                        size = "N/A"
                elif line.startswith("Speed:") and "Unknown" not in line:
                    speed = line.split(":", 1)[1].strip().replace(" MT/s", "")
                elif line.startswith("Type:") and "Unknown" not in line:
                    ram_type = line.split(":", 1)[1].strip()
                
                elif line.startswith("Total Width:"):
                    val = line.split(":", 1)[1].strip()
                    if val.lower() != "unknown":
                        try:
                            total_width = int(val.split()[0])  # get just the number of bits
                        except:
                            total_width = None
                elif line.startswith("Data Width:"):
                    val = line.split(":", 1)[1].strip()
                    if val.lower() != "unknown":
                        try:
                            data_width = int(val.split()[0])
                        except:
                            data_width = None
                elif line.startswith("Error Correction Type:"):
                    try:
                        ecc = line.split(":", 1)[1].strip()
                        ecc_status = "ECC" if "ecc" in ecc.lower() else "Non-ECC"
                    except:
                        ecc_status = "N/A"
                elif line.startswith("Configured Memory Speed:"):
                    configured_speed = line.split(":", 1)[1].strip().replace(" MT/s", "")
                if total_width is not None and data_width is not None:
                    if total_width == 72 or data_width == 72:
                        ecc_status = "ECC"
                    elif ecc_status == "N/A":  # Only overwrite if no explicit ECC info found
                        ecc_status = "Non-ECC"        
            if skip or size == "N/A":
                continue  # skip invalid or unpopulated slots
            
            stats[f"RAM:{slot}: {size}GB {speed}Mhz -{ram_type} -{ecc_status}"] = f"Cfg Speed: {configured_speed}"

        vm = psutil.virtual_memory()
        used = vm.used // 1024**2
        total = vm.total // 1024**2
        ram_stats = f"{used} / {total} MB"                
        stats["RAM: "] = " "
        stats["RAM:Total RAM Usage:"] = ram_stats
    
    except Exception as e:
            stats["RAM:RAM"] = f"N/A - {e}"
    return stats


def get_stats_gpu(show_fans = False):    # GPUs
    global gpu_count
    stats = {}
    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        label = name if gpu_count == 1 else f"{name} (GPU {i})"
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        except pynvml.NVMLError:
            power = 0.0
        try:
            clocks = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        except pynvml.NVMLError:
            clocks = 0
        try:
            fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
        except pynvml.NVMLError:
            fan_speed = None

        stats[f"GPU:{label}"] = f"{util.gpu} %"
        stats[f"GPU:VRAM {i}" if gpu_count > 1 else "GPU:VRAM"] = f"{mem.used // 1024**2} / {mem.total // 1024**2} MB"
        stats[f"GPU:TEMP {i}" if gpu_count > 1 else "GPU:TEMP"] = f"{temp}°C"
        stats[f"GPU:POWER {i}" if gpu_count > 1 else "GPU:POWER"] = f"{power:.0f} W"
        if i < (gpu_count -1):
            stats[f"GPU:GPU Clock {i}" if gpu_count > 1 else "GPU:GPU Clock"] = f"{clocks} MHz\n"
        else:
            stats[f"GPU:GPU Clock {i}" if gpu_count > 1 else "GPU:GPU Clock"] = f"{clocks} MHz"
        if show_fans and fan_speed is not None:
            stats[f"GPU:Fan {i}" if gpu_count > 1 else "GPU:Fan"] = f"{fan_speed} RPM"
        #stats[f"-\n"] = f"-\n"
    return stats


def get_stats(per_core=False, show_fans=False):
    stats = {}
   
    stats.update(get_stats_cpu(per_core))            # CPU
    if show_fans:                                    # Fans
        stats.update(get_stats_fans())    
    stats.update(get_stats_ram())                    # RAM
    stats.update(get_stats_gpu(show_fans))           # GPUs

# Disk I/O
    disk_deltas = get_disk_delta()

    for disk_name, delta in disk_deltas.items():
        # Skip partitions like sda1, nvme0n1p1
        if not os.path.exists(f"/sys/block/{disk_name}"):
            continue

        size_gb = get_disk_size_gb(disk_name)
        if size_gb < 50:
            continue

        stats[f"DISK:DISK I/O ({disk_name})"] = f"IN = {delta['in']:.2f} MB/s | OUT = {delta['out']:.2f} MB/s"

    
    return stats


def build_table(per_core=False, show_fans=False):
    stats = get_stats(per_core, show_fans)

    title_table = Table.grid(padding=(0, 1), expand=True)   
    cpu_table = Table.grid(padding=(0, 1), expand=True)  
    fan_table = Table.grid(padding=(0, 1), expand=True)
    ram_table = Table.grid(padding=(0, 1), expand=True)
    gpu_table = Table.grid(padding=(0, 1), expand=True)
    disk_table = Table.grid(padding=(0, 1), expand=True)
    
    # Set up columns
    for table in [title_table, cpu_table, ram_table, gpu_table, disk_table]:
        table.add_column("Metric", style="cyan", justify="left")
        table.add_column("Value", style="green", justify="right")
    
    # TITLE
    title_table.add_row("Metric","Value")
     
    for key, value in stats.items():
        # CPU
        if key.startswith("CPU"):
            split_key = key.split(":", 1)
            if len(split_key) == 2:
                label = split_key[1].strip()
            else:
                label = key.strip()
            cpu_table.add_row(label, value)
        # FAN
        if key.startswith("FAN:"):
            key = key[len("FAN:"):].strip()
            fan_table.add_row(key, value)
        # RAM
        elif key.startswith("RAM:"):
            key = key[len("RAM:"):].strip()
            ram_table.add_row(key, value)
        # GPU
        elif key.startswith("GPU:"):
            key = key[len("GPU:"):].strip()
            gpu_table.add_row(key, value)
        # Disk
        elif key.startswith("DISK"):
            key = key[len("DISK:"):].strip()
            disk_table.add_row(key, value)

    # Wrap the GPU and Disk tables in Panels
        # Wrap each in a panel
    title_panel = Panel(title_table, title="System Metrics",border_style="cyan", expand=True)
    cpu_panel = Panel(cpu_table, title="CPU Stats", border_style="blue", expand=True)
    if fan_table.rows:
        fan_panel = Panel(fan_table, title="FAN Stats", border_style="red", expand=True)
    ram_panel = Panel(ram_table, title="RAM Stats", border_style="green", expand=True)
    gpu_panel = Panel(gpu_table, title="GPU Stats", border_style="magenta", expand=True)
    disk_panel = Panel(disk_table, title="Disk I/O", border_style="yellow", expand=True)


    return Group(
        title_panel,
        cpu_panel,
        ram_panel,
        gpu_panel,
        disk_panel
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live System Monitor")
    parser.add_argument("--per-core", action="store_true", help="Show all CPU cores separately")
    parser.add_argument("--fan", action="store_true", help="Display GPU fan speeds if available")
    args = parser.parse_args()

    time.sleep(1)
    with Live(build_table(per_core=args.per_core, show_fans=args.fan), refresh_per_second=1, screen=False) as live:
        while True:
            time.sleep(1)
            live.update(build_table(per_core=args.per_core, show_fans=args.fan))
