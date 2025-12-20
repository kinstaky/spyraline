from spyral import ClusterParameters, LinearJoinParameters
from spyral.core.point_cloud import PointCloud
from spyral.core.clusterize import form_clusters

from pathlib import Path
import h5py as h5
from tqdm import tqdm
import numpy as np

cluster_params = ClusterParameters(
    min_cloud_size=50,
    min_points=5,
    min_size_scale_factor=0.0,
    min_size_lower_cutoff=5,
    cluster_selection_epsilon=5.0,
    overlap_join = None,
    continuity_join = None,
    linear_join = LinearJoinParameters(
        position_threshold = 0.0,
        slope_threshold = 5.0,
    ),
    outlier_scale_factor=0.05,
)

workspace = workspace_path = Path("/data/sustech/spyraline/")

def get_slope(x, y) -> float:
    t = np.sum((x - x.mean())**2)
    if t == 0:
        return -1000
    return np.sum((x - x.mean())*(y - y.mean())) / np.sum((x - x.mean())**2)

def main():
    pointcloud_path = workspace_path / "Pointcloud"
    run_number = 1025

    point_file_path = pointcloud_path / f"run_{run_number:04d}.h5"
    point_file = h5.File(point_file_path, "r")
    cloud_group: h5.Group = point_file["cloud"]
    min_event = cloud_group.attrs["min_event"]
    max_event = cloud_group.attrs["max_event"]

    min_pos_diff = []
    min_cluster_size = []
    min_compare_cluster_size = []
    xz_slope = []
    yz_slope = []
    compare_xz_slope = []
    compare_yz_slope = []
    smallest_range = []
    slope_point_x_diff = []
    slope_point_y_diff = []
    for idx in tqdm(range(min_event, max_event+1)):
        event_name = f"cloud_{idx}"
        if event_name not in cloud_group:
            continue
        cloud_data = cloud_group[event_name]
        if cloud_data is None:
            continue
        cloud = PointCloud(idx, cloud_data[:].copy())
        clusters, labels = form_clusters(cloud, cluster_params)
        jclusters = [cluster for cluster in clusters if cluster.label != -1]
        jclusters = sorted(
            jclusters,
            key=lambda x:x.point_cloud.data[-1,6]*np.mean(x.point_cloud.data[:,7])
        )
        current_min_pos_diff = None
        min_pos_diff_idx = None
        min_pos_diff_cidx = None
        slope_point_x = 10000
        slope_point_y = 10000
        for idx, cluster in enumerate(jclusters):
            if cluster.label == -1:
                continue
            avg_pos = np.median(cluster.point_cloud.data[:3, :3], axis=0)
            for cidx, compare_cluster in enumerate(jclusters):
                if compare_cluster.label == -1 or cidx == idx:
                    continue
                compare_avg_pos = np.median(compare_cluster.point_cloud.data[-3:, :3], axis=0)
                pos_diff = np.linalg.norm(avg_pos - compare_avg_pos)
                if current_min_pos_diff != None and pos_diff >= current_min_pos_diff:
                    continue
                current_min_pos_diff = pos_diff
                min_pos_diff_idx = idx
                min_pos_diff_cidx = cidx
                cluster_start_point_pos = np.sqrt(
                    cluster.point_cloud.data[-1, 0]**2
                    + cluster.point_cloud.data[-1, 1]**2
                )
                if cluster_start_point_pos > 50:
                    continue
                compare_start_point_pos = np.sqrt(
                    compare_cluster.point_cloud.data[-1, 0]**2
                    + compare_cluster.point_cloud.data[-1, 1]**2
                )
                if compare_start_point_pos < 50:
                    continue
                xz_fit_pars = np.polyfit(
                    cluster.point_cloud.data[:, 2].flatten(),
                    cluster.point_cloud.data[:, 0].flatten(),
                    1
                )
                yz_fit_pars = np.polyfit(
                    cluster.point_cloud.data[:, 2].flatten(),
                    cluster.point_cloud.data[:, 1].flatten(),
                    1
                )
                compare_mean: float = np.mean(compare_cluster.point_cloud.data[:, :3], axis=0)
                slope_point_x: float = xz_fit_pars[0] * compare_mean[2] + xz_fit_pars[1] - compare_mean[0]
                slope_point_y: float = yz_fit_pars[0] * compare_mean[2] + yz_fit_pars[1] - compare_mean[1]
        if current_min_pos_diff is not None:
            min_pos_diff.append(current_min_pos_diff)
            idx = min_pos_diff_idx
            cidx = min_pos_diff_cidx
            min_cluster_size.append(len(jclusters[idx].point_cloud.data))
            min_compare_cluster_size.append(len(jclusters[cidx].point_cloud.data))
            # get slope under the minimum position difference
            xz_slope.append(get_slope(
                jclusters[idx].point_cloud.data[:, 0].flatten(),
                jclusters[idx].point_cloud.data[:, 2].flatten()
            ))
            yz_slope.append(get_slope(
                jclusters[idx].point_cloud.data[:, 1].flatten(),
                jclusters[idx].point_cloud.data[:, 2].flatten()
            ))
            compare_xz_slope.append(get_slope(
                jclusters[cidx].point_cloud.data[:, 0].flatten(),
                jclusters[cidx].point_cloud.data[:, 2].flatten()
            ))
            compare_yz_slope.append(get_slope(
                jclusters[cidx].point_cloud.data[:, 1].flatten(),
                jclusters[cidx].point_cloud.data[:, 2].flatten()
            ))
            smallest_range.append(
                np.sqrt(
                    cluster.point_cloud.data[-1, 0]**2
                    +cluster.point_cloud.data[-1,1]**2
                )
            )
            slope_point_x_diff.append(slope_point_x)
            slope_point_y_diff.append(slope_point_y)
            

    combined = np.column_stack((
        min_cluster_size,
        min_compare_cluster_size,
        min_pos_diff,
        xz_slope,
        yz_slope,
        compare_xz_slope,
        compare_yz_slope,
        smallest_range,
        slope_point_x_diff,
        slope_point_y_diff,
    ))
    save_path = workspace_path / "Linearjoin"
    if not save_path.exists():
        save_path.mkdir()
    np.save(save_path / f"threshold_{run_number:04d}.npy", combined)            


if __name__ == "__main__":
    main()