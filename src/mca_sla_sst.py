"""
Maximum Covariance Analysis (MCA) between Sea Level Anomalies (SLA)
and Sea Surface Temperature (SST) over the African Atlantic coast.

The workflow:
- Aligns SSH and SST datasets in space and time
- Subsamples spatially for computational efficiency
- Computes anomalies and standardizes fields
- Performs MCA using SVD of the cross-covariance matrix
- Produces spatial patterns, expansion coefficients, and SCF diagnostics

Outputs:
- Figure: figures/Figure4_MCA_SLA_SST_corrected.png
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.sparse.linalg import svds
import warnings

warnings.filterwarnings("ignore")


# ======================================================
# Configuration
# ======================================================
SSH_FILE = "data/SSHafr.nc"
SST_FILE = "data/SSTafr.nc"
SUBSAMPLE_FACTOR = 4
N_MAX_MODES = 5
OUTPUT_FIG = "figures/Figure4_MCA_SLA_SST_corrected.png"


# ======================================================
# Utility functions
# ======================================================
def setup_map_ax(ax, lon, lat, title):
    ax.coastlines(resolution="50m", linewidth=0.5)
    ax.add_feature(cfeature.LAND, color="lightgray", alpha=0.7)
    ax.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.7)

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_extent(
        [lon.min(), lon.max(), lat.min(), lat.max()],
        crs=ccrs.PlateCarree(),
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    return ax


# ======================================================
# Main workflow
# ======================================================
def main():

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    ds_ssh = xr.open_dataset(SSH_FILE)
    ds_sst = xr.open_dataset(SST_FILE)

    ssh = ds_ssh[list(ds_ssh.data_vars)[0]]
    sst = ds_sst[list(ds_sst.data_vars)[0]]

    # --------------------------------------------------
    # 2. Align coordinates
    # --------------------------------------------------
    if not ssh.latitude.equals(sst.latitude) or not ssh.longitude.equals(sst.longitude):
        sst = sst.interp(
            latitude=ssh.latitude,
            longitude=ssh.longitude,
            method="nearest",
        )

    common_time = np.intersect1d(ssh.time.values, sst.time.values)
    if len(common_time) == 0:
        raise ValueError("No overlapping time periods found.")

    ssh = ssh.sel(time=common_time)
    sst = sst.sel(time=common_time)

    nt, ny, nx = ssh.shape

    # --------------------------------------------------
    # 3. Spatial subsampling
    # --------------------------------------------------
    ssh = ssh[:, ::SUBSAMPLE_FACTOR, ::SUBSAMPLE_FACTOR]
    sst = sst[:, ::SUBSAMPLE_FACTOR, ::SUBSAMPLE_FACTOR]

    ny_sub, nx_sub = ssh.shape[1:]

    # --------------------------------------------------
    # 4. Anomalies and masking
    # --------------------------------------------------
    mask = np.isfinite(ssh).all("time") & np.isfinite(sst).all("time")

    ssh = ssh.where(mask, 0) - ssh.mean("time")
    sst = sst.where(mask, 0) - sst.mean("time")

    ssh_2d = ssh.values.reshape(nt, ny_sub * nx_sub)
    sst_2d = sst.values.reshape(nt, ny_sub * nx_sub)

    std_mask = (ssh_2d.std(axis=0) > 0) & (sst_2d.std(axis=0) > 0)
    ssh_2d = ssh_2d[:, std_mask]
    sst_2d = sst_2d[:, std_mask]

    ssh_std = StandardScaler().fit_transform(ssh_2d)
    sst_std = StandardScaler().fit_transform(sst_2d)

    # --------------------------------------------------
    # 5. MCA via cross-covariance SVD
    # --------------------------------------------------
    n_modes = min(N_MAX_MODES, nt - 1, ssh_std.shape[1])
    cross_cov = (ssh_std.T @ sst_std) / (nt - 1)

    U, s, Vt = svds(cross_cov, k=n_modes)
    idx = np.argsort(s)[::-1]

    s = s[idx]
    U = U[:, idx]
    Vt = Vt[idx, :]

    scf = (s ** 2) / np.sum(s ** 2) * 100

    ssh_pcs = ssh_std @ U
    sst_pcs = sst_std @ Vt.T

    # --------------------------------------------------
    # 6. Reconstruct spatial patterns
    # --------------------------------------------------
    ssh_maps = np.zeros((n_modes, ny_sub * nx_sub))
    sst_maps = np.zeros((n_modes, ny_sub * nx_sub))

    ssh_maps[:, std_mask] = U.T
    sst_maps[:, std_mask] = Vt

    ssh_maps = ssh_maps.reshape(n_modes, ny_sub, nx_sub)
    sst_maps = sst_maps.reshape(n_modes, ny_sub, nx_sub)

    lat = ssh.latitude.values
    lon = ssh.longitude.values

    # --------------------------------------------------
    # 7. Plot
    # --------------------------------------------------
    fig = plt.figure(figsize=(14, 12))

    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    im1 = ax1.pcolormesh(lon, lat, ssh_maps[0], cmap="RdBu_r", shading="auto")
    setup_map_ax(ax1, lon, lat, "Panel A: Leading SLA MCA Pattern")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
    im2 = ax2.pcolormesh(lon, lat, sst_maps[0], cmap="RdBu_r", shading="auto")
    setup_map_ax(ax2, lon, lat, "Panel B: Leading SST MCA Pattern")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    ax3 = fig.add_subplot(2, 2, 3)
    pc1_ssh = (ssh_pcs[:, 0] - ssh_pcs[:, 0].mean()) / ssh_pcs[:, 0].std()
    pc1_sst = (sst_pcs[:, 0] - sst_pcs[:, 0].mean()) / sst_pcs[:, 0].std()
    r = pearsonr(ssh_pcs[:, 0], sst_pcs[:, 0])[0]

    ax3.plot(ssh.time, pc1_ssh, label="SLA PC1")
    ax3.plot(ssh.time, pc1_sst, label="SST PC1")
    ax3.set_title(f"Panel C: Expansion Coefficients (r = {r:.3f})")
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.bar(np.arange(1, n_modes + 1), scf)
    ax4.set_xlabel("Mode")
    ax4.set_ylabel("SCF (%)")
    ax4.set_title("Panel D: Squared Covariance Fraction")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
