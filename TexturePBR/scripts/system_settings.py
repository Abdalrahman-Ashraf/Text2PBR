import math
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetTemperature, nvmlShutdown

def throttle_gpu_hotspot_temp(throttle_time=0.2, throttle_temp=94, throttle=True, display_thermals=False, show_t=False):
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)

        core_temp = nvmlDeviceGetTemperature(handle, 0)
        hotspot_temp = 0
        if core_temp > 55:
            hotspot_temp = core_temp + math.sqrt(0.15 * (core_temp**2))
        else:
            hotspot_temp = core_temp * 1.1
        if display_thermals:
            print(f"Estimated GPU Hotspot Temp: {hotspot_temp:.1f} °C  ", end="\r", flush=True)

        if throttle:
            if throttle_temp < 75:
                throttle_temp = 75
            if throttle_time > 5:
                throttle_time = 5

            throttle_count = 1
            while hotspot_temp > throttle_temp - 1:
                time.sleep(throttle_time)
                core_temp = nvmlDeviceGetTemperature(handle, 0)
                hotspot_temp = core_temp + math.sqrt(0.15 * (core_temp**2))
                if display_thermals:
                    print(f"Estimated GPU Hotspot Temp: {hotspot_temp:.1f} °C, Throttle count: {int(throttle_count)}  ", end="\r", flush=True)
                elif show_t:
                    print("Throttling processes...")
                throttle_count += 1
    finally:
        nvmlShutdown()
