import h5py as h5
from pathlib import Path

run_min = 1025
run_max = 1045
runs_to_skip = [1052, 1069]

workspace_path = Path("/data/pwl/spyraline")
phase = "Pointcloud"
event_type = "silicon"

def main():
	vds_filename = \
		workspace_path \
		/ phase \
		/ f"run_{run_min:04d}_{run_max:04d}_{event_type}.h5"
	with h5.File(vds_filename, 'w') as v_file:
		vidx = 0
		v_cloud_group = v_file.create_group("cloud")
		for run in range(run_min, run_max+1):
			if run in runs_to_skip:
				continue
			prefix = f"run_{run:04d}"
			filename = workspace_path / phase / f"{prefix}.h5"
			with h5.File(filename, 'r') as origin_file:
				# 遍历原始文件中的 cloud 组
				cloud_group: h5.Group = origin_file.get("cloud")
				min_event = cloud_group.attrs["min_event"]
				max_event = cloud_group.attrs["max_event"]
				for idx in range(min_event, max_event+1):
					event_name = f"{event_type}_{idx}"
					if event_name not in cloud_group:
						continue
					vsource = h5.VirtualSource(
						filename,
						f"cloud/{event_name}",
						shape=cloud_group[event_name].shape,
					)
					layout = h5.VirtualLayout(
						shape=cloud_group[event_name].shape,
						dtype=cloud_group[event_name].dtype,
					)
					layout[:] = vsource
					v_cloud_group.create_virtual_dataset(
						f"{event_type}_{vidx}", layout
					)
					vidx += 1
			print(f"Finish run {run}")
		v_cloud_group.attrs["min_event"] = 0
		v_cloud_group.attrs["max_event"] = vidx-1

if __name__ == "__main__":
	main()