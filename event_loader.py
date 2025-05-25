# event_loader.py

import numpy as np

def read_aedat(file_path: str) -> np.ndarray:
    """
    Read a DVS128 AEDAT 3.1 file and return an [N,4] array:
      columns = [x, y, timestamp, polarity]
    """
    with open(file_path, 'rb') as f:
        # Skip all header/comment lines starting with '#'
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b'#'):
                f.seek(pos)
                break

        # Read the remaining uint32 stream
        raw = np.fromfile(f, dtype=np.uint32)

    # Only keep full (data, ts) pairs
    pair_count = raw.shape[0] // 2
    raw = raw[: pair_count * 2]

    data_words = raw[0::2]
    timestamps = raw[1::2]

    # Decode coordinates & polarity
    x =  (data_words >> 17) & 0x1FFF
    y =  (data_words >>  2) & 0x1FFF
    p =  (data_words      ) & 0x1       # polarity bit

    # Stack into float32 so your transforms can run
    events = np.stack([x, y, timestamps, p], axis=1).astype(np.float32)
    return events

def load_events(event_file: str, start_usec: int, end_usec: int) -> np.ndarray:
    """
    Load events from `event_file` between `start_usec` and `end_usec` (both in µs).
    Returns an (M,4) float32 array.
    """
    all_events = read_aedat(event_file)
    ts = all_events[:, 2]
    mask = (ts >= start_usec) & (ts < end_usec)
    return all_events[mask]
