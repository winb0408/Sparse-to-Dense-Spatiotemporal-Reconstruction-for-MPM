# Shared launch gate for the experiment scripts.  Source it after `cd` to the
# repository root, then call `gpu_guard <index>` on every non-DRY launch.
#
# Two checks, both learned from incidents rather than from theory:
#
# 1. **The device must be visible to the driver.** On 2026-11-13 GPU 2 fell off
#    the bus mid-run ("Unable to determine the device handle for
#    0000:86:00.0"). A launch against a dead handle does not fail — it hangs,
#    burning CPU on a device that will never answer, and the wrapper looks like
#    it is still working. `CUDA_VISIBLE_DEVICES=2` is not a check.
#    Evidence: docs/evidence/gpu2_falloff_20261113/INCIDENT.md.
#
# 2. **The card must not be more than 60 % occupied at launch.** Filling GPU 2
#    has hung this server and triggered an unattended restart that took other
#    users' jobs down with ours. The occupancy is measured at launch time, not
#    remembered: "it was free an hour ago" is exactly the shape of the last
#    incident. An unreadable card counts as *not free*, because silence is not
#    evidence.
#
#    GPU 2 fell off the bus on 2026-11-13, so the owner authorised the other
#    three cards for this work, at up to 60 % occupancy (raised from 50 % after
#    measuring what we actually ask for). The same rule applies to every index,
#    which is also why it is checked per index — one dead device breaks a
#    whole-card-list query and would look like "everything unreadable".
#
#    Why 60 % is defensible here and would not be for a bigger job: an E026
#    training arm peaked at 5979 MiB allocated / 6658 MiB reserved and a test
#    evaluation is smaller; at 60 % of 49140 MiB the card still has ~19.6 GB
#    free, about three times our peak. A job with a larger footprint must lower
#    this ceiling rather than inherit it.
#
#   GPU_MAX_FRACTION=0.6   ceiling on used/total before a launch, every index
#   ALLOW_BUSY=1           skip check 2; a visible act, not a forgotten one

GPU_MAX_FRACTION="${GPU_MAX_FRACTION:-0.6}"

gpu_guard () {   # gpu_guard <physical index>; returns 2 after printing why
  local idx="$1" nvsmi="${NVIDIA_SMI:-nvidia-smi}" listed mem used total
  listed="$("$nvsmi" -L 2>&1)" || true
  if ! grep -q "^GPU ${idx}:" <<< "$listed"; then
    printf 'refusing GPU %s: the driver does not list it.\n' "$idx" >&2
    printf 'A device that has fallen off the bus accepts CUDA_VISIBLE_DEVICES\n' >&2
    printf 'and then hangs. `%s -L` said:\n' "$nvsmi" >&2
    printf '%s\n' "$listed" | sed -n '1,8p' >&2
    return 2
  fi
  [[ "${ALLOW_BUSY:-0}" != "1" ]] || return 0
  mem="$("$nvsmi" --query-gpu=memory.used,memory.total \
        --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
  used="${mem%%,*}"; total="${mem##*,}"
  if ! [[ "$used" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ ]] || (( total == 0 )); then
    printf 'refusing GPU %s: could not read its occupancy (got %s).\n' \
      "$idx" "'$mem'" >&2
    printf 'An unreadable card is not evidence that it is free.\n' >&2
    return 2
  fi
  if awk -v u="$used" -v t="$total" -v f="$GPU_MAX_FRACTION" \
       'BEGIN{exit !(u/t >= f)}'; then
    printf 'refusing GPU %s: %s/%s MiB is %s%%, at or above the %s%% ceiling.\n' \
      "$idx" "$used" "$total" \
      "$(awk -v u="$used" -v t="$total" 'BEGIN{printf "%.1f", 100*u/t}')" \
      "$(awk -v f="$GPU_MAX_FRACTION" 'BEGIN{printf "%.0f", 100*f}')" >&2
    printf 'The card must be under the occupancy ceiling; ALLOW_BUSY=1 overrides.\n' >&2
    return 2
  fi
  printf 'GPU %s at %s/%s MiB -- under the ceiling, allowed this launch\n' \
    "$idx" "$used" "$total" >&2
}

gpu_free_index () {   # gpu_free_index <idx>... -> first index passing gpu_guard
  local i
  for i in "$@"; do
    if gpu_guard "$i" 2>/dev/null; then printf '%s\n' "$i"; return 0; fi
  done
  return 1
}
