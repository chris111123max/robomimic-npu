# Stage 1 raw datasets

Each policy HDF5 file declares `progress_observation_schema`. It maps
`payload_in_target_bin` and `trash_in_trash_bin` to their runtime-derived flat indices in the
canonical `object` observation. The fields are already embedded there and are not duplicated as
transition datasets. They are analysis-only and do not affect reward or rollout behavior.

Each collection creates a timestamped subdirectory. Generated HDF5 and JSON files are runtime
artifacts and should not be committed. No success-only filtering or equal-budget subsampling is
performed in Stage 1.
