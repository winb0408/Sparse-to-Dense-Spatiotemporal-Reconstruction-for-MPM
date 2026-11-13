# GPU 2 fell off the bus, 2026-11-13 ~21:44 — E026 `real` interrupted

**What happened:** the E026 `real` group was running its third floor evaluation
(`P3_Wlegacy`, test split) on GPU 2 when the device disappeared from the driver.
`nvidia-smi -L` now lists GPUs 0, 1, 3 and, in place of GPU 2, the line

```
Unable to determine the device handle for gpu 0000:86:00.0: Unknown Error
```

`nvidia-smi --query-gpu=...` fails outright on the same device. Snapshot with
timestamps: `snapshot.txt` in this directory.

**Timeline** (from the run log and file mtimes, all `+08:00`):

| time | event |
|---|---|
| 21:27 | `E026 ... SOURCE_KIND=real STAGE=all`, `GPU=2` launched; GPU 2 measured at 20569/49140 MiB = **41.9 %**, below the 50 % gate |
| 21:35 | `I0` floor eval written (7 min, normal) |
| 21:44 | `W0_full3d` floor eval written (9 min, normal); `P3_Wlegacy` eval started |
| ~21:44 | device falls off the bus; the eval process makes no further progress |
| 22:00 | noticed: log mtime stuck at 21:44, `nvidia-smi` reports the dead handle, the eval process is in `R` state burning CPU with its dataloader workers asleep |
| 22:02 | `SIGTERM` to the eval and its wrapper; no `spur_mpm` process of this group remains |

**What survived:** `E026_real_yz_k2_kz2_fold0_s0_I0_test` and
`..._W0_full3d_test`. Their `summary.json` files differ in exactly one field —
`arm` — which is the identity-kernel collapse this group was expected to show
(`wiener_fill_all` short-circuits to `interp_linear` when `psf.is_identity`).

**What was lost:** only the `P3_Wlegacy` floor eval (re-runnable, no partial
output directory was left behind, so a rerun will not hit the overwrite guard)
and everything after it: the three trained arms of the `real` group, the
`STAGE=analysis` passes for both E026 groups, and E025, which had not started.

**Cause:** not determinable from a user account. It is a device-level fault, not
an out-of-memory in our process: two floor evaluations of identical shape had
just completed normally on the same device, our job was within the launch gate,
and the failing device is the one this box has previously crashed when filled
(`Project log/Project_log.md`, 2026-11-11 entry). `nvidia-smi --gpu-reset` needs
root and a device the driver can still see; a device that has fallen off the bus
usually needs the machine power-cycled. Neither is available to this session.

**Containment taken:** GPU work stopped on GPU 2 (no further launch attempted);
the spinning process was terminated so it stops holding the device; GPUs 0/1/3
are other users' jobs and were not touched. No experiment will be re-launched on
GPU 2 until `nvidia-smi -L` lists it again.

**Standing rule this incident adds:** the run scripts check the *memory* gate at
launch but not whether the device is *visible*. A launch against a dead handle
would have hung the same way. Both `run_e026_historical.sh` and `run_e025.sh`
therefore now refuse to start when `nvidia-smi -L` does not list the requested
device, and every GPU stage carries a `timeout` so a stall fails instead of
spinning forever (`spur_mpm/experiments/_gpu_gate.sh`).

**Occupancy ceiling after the incident:** GPUs 0/1/3 are authorised for this work
below 60 % occupancy (owner's instruction, 2026-11-13; the 50 % figure quoted
above was for GPU 2 before it died). 60 % is justified by the measured footprint
— an E026 training arm peaked at 5979 MiB allocated / 6658 MiB reserved, so a
60 %-occupied 49 GB card still leaves ~19.6 GB, about three times our peak. A job
with a larger footprint must lower the ceiling rather than inherit it. The queue
polls every 10 minutes rather than every 2, so waiting does not cost the box.
