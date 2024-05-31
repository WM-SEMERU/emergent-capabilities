import timeit
from datetime import timedelta, datetime
from functools import wraps
import ipywidgets as widgets
from IPython.display import display
import time

def tobase(number, base, pad=None):
    if number == 0:
        digit_list = [0]
    else:
        digit_list = []
        while number > 0:
            remainder = number % base
            digit_list.insert(0, remainder)
            number //= base

    if pad is not None:
        while len(digit_list) < pad:
            digit_list.insert(0, 0)
    
    return digit_list

def fmt_delta(elapsed_time):
    ipart = int(elapsed_time)
    fpart = elapsed_time - ipart
    hr, mn, sec = tobase(ipart, 60, pad=3)
    ms = int(fpart * 1000)
    elapsed_parts = []
    if hr > 0:
        elapsed_parts.append(f"{hr}hr")
    if mn > 0:
        elapsed_parts.append(f"{mn}min")
    if sec > 0:
        elapsed_parts.append(f"{sec}s")

    if len(elapsed_parts) < 2 and ms > 0:
        elapsed_parts.append(f"{ms}ms")

    if len(elapsed_parts) == 0:
        return "~0s"
    
    return " ".join(elapsed_parts)

# only one at a time
timer_started = None
timer_label = None
def time_start(label=".timehelp"):
    global timer_started, timer_label
    timer_label = label
    display_now = datetime.today().strftime("%Y-%m-%d@%H:%M:%S")
    print(f"[{display_now}|{timer_label}] Starting timer.")
    timer_started = timeit.default_timer()

def time_end():
    now = timeit.default_timer()
    elapsed = now - timer_started
    display_now = datetime.today().strftime("%Y-%m-%d@%H:%M:%S")
    print(f"[{display_now}|{timer_label}] Time elapsed:", fmt_delta(elapsed))

def with_progress(steps=None, label="Progress"):
    assert steps is not None, "@with_progress: Missing required parameter steps"
    total_steps = steps
    def decorator(func):
        @wraps(func)
        def wrapper(*args, skip=None, **kwargs):
            # Create widgets
            progress = widgets.IntProgress(
                value=0, min=0, max=total_steps,
                **(
                    dict(description=f"{label}:") if label
                    else {}
                )
            )
            estimated_time = widgets.Label(value="Estimated time remaining: calculating...")
            
            hbox = widgets.VBox([progress, estimated_time])
            display(hbox)
            
            start_time = time.time()

            start = 0
            if skip is not None:
                assert skip >= 0, f"Cannot skip {skip} lines"
                start = skip

            my_total_steps = total_steps - start
            result = None
            progress.value = start
            for step in range(start, start + my_total_steps):
                # Call the wrapped function
                result = func(*args, **kwargs, step=step)
                
                # Update progress bar
                progress.value = step + 1
                
                # Calculate and update estimated time remaining
                elapsed_time = time.time() - start_time
                remaining_steps = total_steps - (step + 1)
                steps_so_far = step + 1 - start
                if steps_so_far > 1:
                    time_per_step = elapsed_time / steps_so_far
                    estimated_remaining_time = time_per_step * remaining_steps
                    estimated_time.value = f"[{step + 1}/{total_steps}] ETA: {fmt_delta(estimated_remaining_time)}, {fmt_delta(elapsed_time)} elapsed..."
                else:
                    estimated_time.value = f"[{step + 1}/{total_steps}] ETA: calculating, {fmt_delta(elapsed_time)} elapsed..."

            progress.value = total_steps
            estimated_time.value = f"Done, {fmt_delta(time.time() - start_time)} elapsed."
            print(estimated_time.value)
            return result
        
        return wrapper
    return decorator


def display_header(text):
    header = widgets.HTML(value=f"<h1 style='font-weight: bold;'>{text}</h1>")
    display(header)
    print(f"# {text}")
