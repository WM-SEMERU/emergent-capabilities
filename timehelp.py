import timeit
from datetime import timedelta, datetime

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