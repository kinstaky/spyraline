from spyral.trace.trace_reader import create_reader
from spyral.core.run_stacks import form_run_string
from spyral import GetParameters, FribParameters

from pathlib import Path
from tqdm import tqdm
from numpy.random import default_rng, SeedSequence
import matplotlib.pyplot as plt
import numpy as np


get_params = GetParameters(
    baseline_window_scale=20.0,
    peak_separation=50.0,
    peak_prominence=20.0,
    peak_max_width=50.0,
    peak_threshold=40.0,
)

frib_params = FribParameters(
    baseline_window_scale=100.0,
    peak_separation=50.0,
    peak_prominence=20.0,
    peak_max_width=500.0,
    peak_threshold=100.0,
    ic_delay_time_bucket=1100,
    ic_multiplicity=1,
)
def main():
    trace_path = Path("/data/tempMergedData/E565/")
    run = 1025
    trace_file = trace_path / f"{form_run_string(run)}.h5"
    reader = create_reader(trace_file, run)
    rng = default_rng(seed=SeedSequence())

    time_data = []
    for idx in tqdm(reader.event_range()):
        event = reader.read_event(idx, get_params, frib_params, rng)
        if event.get_pads is None:
            continue
        for trace_num, trace in enumerate(event.get_pads.traces):
            if trace.get_number_of_peaks() == 0 or trace.get_number_of_peaks() > 5:
                continue
            for peak_num, peak in enumerate(trace.get_peaks()):
                time_data.append([idx, trace_num, peak_num, peak.centroid])

    time_array = np.array(time_data)
    np.save(f"/data/sustech/spyraline/check_time_range_{run:04d}.npy", time_array)
    fig, ax = plt.subplots(1, 1)
    ax.hist(time_array[:, 3], bins=512, range=(0, 512))
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    plt.show()

if __name__ == "__main__":
    main()